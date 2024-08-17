import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
from torch_geometric.nn import ChebConv, GCNConv, global_mean_pool, global_max_pool, GATConv, Set2Set, SAGEConv
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.data import Batch
import pdb


class RQGNN(nn.Module):
    def __init__(self, featuredim, hdim, nclass, width, depth, dropout, normalize):
        super(RQGNN, self).__init__()

        # ChebConv def
        # ChebConv(in_channels, out_channels, K)
        # in_channels: Number of input features per node.
        # out_channels: Number of output features per node.
        # K: Order of the Chebyshev polynomial (depth).
        self.conv = nn.ModuleList([ChebConv(featuredim, featuredim, depth) for _ in range(width)])

        # Linear layer def
        # nn.Linear(in_features, out_features)
        # in_features: Size of each input sample.
        # out_features: Size of each output sample.
        # for feature transformation hence the feature dim as input
        self.linear = nn.Linear(featuredim, featuredim)
        self.linear2 = nn.Linear(featuredim, featuredim)

        # further transfor the concatenated conv outputs to the hidden dims
        self.linear3 = nn.Linear(featuredim*len(self.conv), hdim)
        self.linear4 = nn.Linear(hdim, hdim)

        # used after each linear transf
        self.act = nn.LeakyReLU()
        #self.act = nn.ReLU()

        # Transform features derived from the rayleigh quotient computations
        self.linear5 = nn.Linear(featuredim, hdim)
        self.linear6 = nn.Linear(hdim, hdim)
        
        # final linear layer producing output logits with dimentions equal to the nimber of classes
        self.linear7 = nn.Linear(hdim * 2, nclass)
        #self.linear7 = nn.Linear(hdim, nclass)

        #self.attpool = nn.Linear(hdim, 1)

        self.bn = torch.nn.BatchNorm1d(hdim * 2)
        #self.bn = torch.nn.BatchNorm1d(hdim)

        self.dp = nn.Dropout(p=dropout)
        self.normalize = normalize

        self.linear8 = nn.Linear(featuredim, hdim)
        self.linear9 = nn.Linear(hdim, hdim)
        self.residual = nn.Linear(featuredim, featuredim)  # For residual connections

    def forward(self, data):
        h = self.linear(data.features_list)
        h = self.act(h)
        residual = h

        h = self.linear2(h)
        h = self.act(h)
        h += residual  # Applying residual connection

        # self.conv represent the use of Chebyshev polynomials.
        h_final = []
        for conv in self.conv:
            h0 = conv(h, data.edge_index)
            h_final.append(h0)
        h_final = torch.cat(h_final, dim=-1)

        h = self.linear3(h_final)
        h = self.act(h)
        
        h = self.linear4(h)
        h = self.act(h)

        tmpscores = self.linear8(data.xLx_batch)
        tmpscores = self.act(tmpscores)

        tmpscores = self.linear9(tmpscores)
        tmpscores = self.act(tmpscores)

        scores = torch.zeros(h.size(0), 1, device=h.device)
        for i, node_belong in enumerate(data.node_belong):
            # node_belong is a list of indices indicating which nodes belong to each graph
            # torch.mv(h[node_belong], tmpscores[i]) performs a matrix-vector multiplication between the node features (h[node_belong]) and the transformed scores (tmpscores[i]).
            # torch.unsqueeze(..., 1) adds an additional dimension to the tensor to make it compatible for further processing.
            scores[node_belong] = torch.unsqueeze(torch.mv(h[node_belong], tmpscores[i]), 1)

        temp = torch.mul(data.graphpool_list.to_dense().T, scores).T

        h = torch.mm(temp, h)
        #h = torch.spmm(data.graphpool_list, h)

        xLx = self.linear5(data.xLx_batch)
        
        xLx = self.linear6(xLx)
        xLx = self.act(xLx)

        h = torch.cat([h, xLx], -1)

        if self.normalize:
            h = self.bn(h)

        h = self.dp(h)
        embeddings = self.linear7(h)

        return embeddings

class Graph2Vec(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pooling="mean"):
        super(Graph2Vec, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.pooling = pooling

    def forward(self, x, edge_index, batch):
        # Node-level GNN encoding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Graph-level pooling
        if self.pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_embedding = global_max_pool(x, batch)
        else:
            raise ValueError("Unknown pooling method")
        
        return graph_embedding

class Graph2VecSet2Set(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Graph2VecSet2Set, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.set2set = Set2Set(out_channels, processing_steps=3)

    def forward(self, x, edge_index, batch):
        # Node-level GNN encoding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Set2Set pooling
        graph_embedding = self.set2set(x, batch)
        return graph_embedding

class Graph2VecSortPooling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k):
        super(Graph2VecSortPooling, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.sort_pool = SortAggregation(k)

    def forward(self, x, edge_index, batch):
        # Node-level GNN encoding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # SortPooling
        graph_embedding = self.sort_pool(x, batch)
        return graph_embedding

class Graph2VecGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pooling="max"):
        super(Graph2VecGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.pooling = pooling

    def forward(self, x, edge_index, batch):
        # Node-level GNN encoding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Graph-level pooling
        if self.pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_embedding = global_max_pool(x, batch)
        else:
            raise ValueError("Unknown pooling method")

        return graph_embedding

class EnhancedRQGNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_class, gnn_width, gnn_depth, dropout, normalize, embedding_dim=128, inter_graph_pooling="mean"):
        super(EnhancedRQGNN, self).__init__()
        self.intra_analyzer = RQGNN(feature_dim, hidden_dim, n_class, gnn_width, gnn_depth, dropout, normalize)
        if inter_graph_pooling == "set2set":
            self.inter_analyzer = Graph2VecSet2Set(in_channels=feature_dim, hidden_channels=gnn_width, out_channels=embedding_dim)
            self.projection = nn.Linear(2 * embedding_dim, n_class)
        elif inter_graph_pooling == "sort":
            k = 30
            self.inter_analyzer = Graph2VecSortPooling(in_channels=feature_dim, hidden_channels=gnn_width, out_channels=embedding_dim, k=k)
            self.projection = nn.Linear(k * embedding_dim, n_class)
        elif inter_graph_pooling == "sage":
            self.inter_analyzer = Graph2VecGraphSAGE(in_channels=feature_dim, hidden_channels=gnn_width, out_channels=embedding_dim)
            self.projection = nn.Linear(embedding_dim, n_class)
        else:
            self.inter_analyzer = Graph2Vec(in_channels=feature_dim, hidden_channels=gnn_width, out_channels=embedding_dim, pooling=inter_graph_pooling)
            self.projection = nn.Linear(embedding_dim, n_class)
        self.fc = nn.Linear(n_class + n_class, n_class)  # Combine intra and inter outputs
        

    def forward(self, batch_data):
        intra_output = self.intra_analyzer(batch_data)
        batch_graphs = Batch.from_data_list(batch_data.graphs)
        graph_embeddings = self.inter_analyzer(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)

        # Project Graph2Vec embeddings to match nclass dimension
        graph_embeddings = self.projection(graph_embeddings)
        
        combined_output = torch.cat((intra_output, graph_embeddings), dim=1)
        final_output = self.fc(combined_output)
        return final_output