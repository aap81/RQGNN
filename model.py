import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
from torch_geometric.nn import ChebConv, GCNConv, global_mean_pool, global_max_pool

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

    def forward(self, data):
        h = self.linear(data.features_list)
        h = self.act(h)

        h = self.linear2(h)
        h = self.act(h)

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

# class Graph2Vec(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, pooling="mean"):
#         super(Graph2Vec, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#         self.pooling = pooling

#     def forward(self, data):
#         # Node-level GNN encoding
#         x = self.conv1(data.x, data.edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, data.edge_index)

#         # Graph-level pooling
#         if self.pooling == "mean":
#             graph_embedding = global_mean_pool(x, data.batch)
#         elif self.pooling == "max":
#             graph_embedding = global_max_pool(x, data.batch)
#         else:
#             raise ValueError("Unknown pooling method")
        
#         return graph_embedding

# class EnhancedRQGNN(torch.nn.Module):
#     def __init__(self, featuredim, hdim, nclass, width, depth, dropout, normalize, embedding_dim, pooling="mean"):
#         super(EnhancedRQGNN, self).__init__()
        
#         self.intra_gnn = RQGNN(featuredim, hdim, nclass, width, depth, dropout, normalize)
        
#         # Graph2Vec module for inter-graph embedding
#         self.graph2vec = Graph2Vec(featuredim, hdim, embedding_dim, pooling=pooling)
        
#         # Combining intra-graph and inter-graph features
#         self.combination_layer = nn.Linear(hdim + embedding_dim, hdim)
        
#         # Final classification layer
#         self.classifier = nn.Linear(hdim, nclass)
    
#     def forward(self, data):
#         # Intra-graph feature extraction using RQGNN
#         intra_features = self.intra_gnn(data)
        
#         # Inter-graph embedding using Graph2Vec
#         graph_embedding = self.graph2vec(data)
        
#         # Combine both sets of features
#         combined_features = torch.cat((intra_features, graph_embedding), dim=1)
#         combined_features = F.relu(self.combination_layer(combined_features))
        
#         # Final classification
#         output = self.classifier(combined_features)
        
#         return output

class GADGNN(nn.Module):
    def __init__(self, featuredim, hdim, nclass, width, depth, dropout, normalize):
        super(GADGNN, self).__init__()

        self.conv = []
        for i in range(width):
            self.conv.append(ChebConv(featuredim, featuredim, depth))

        self.linear = nn.Linear(featuredim, featuredim)
        self.linear2 = nn.Linear(featuredim, featuredim)
        self.linear3 = nn.Linear(featuredim*len(self.conv), hdim)
        self.linear4 = nn.Linear(hdim, hdim)
        self.act = nn.LeakyReLU()
        #self.act = nn.ReLU()


        self.linear5 = nn.Linear(featuredim, hdim)
        self.linear6 = nn.Linear(hdim, hdim)
        
        self.linear7 = nn.Linear(hdim * 2, nclass)
        #self.linear7 = nn.Linear(hdim, nclass)

        #self.attpool = nn.Linear(hdim, 1)

        self.bn = torch.nn.BatchNorm1d(hdim * 2)
        #self.bn = torch.nn.BatchNorm1d(hdim)

        self.dp = nn.Dropout(p=dropout)
        self.normalize = normalize

        self.linear8 = nn.Linear(featuredim, hdim)
        self.linear9 = nn.Linear(hdim, hdim)

    def forward(self, data):
        h = self.linear(data.features_list)
        h = self.act(h)

        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(data.features_list), 0])
        for conv in self.conv:
            h0 = conv(h, data.edge_index)
            h_final = torch.cat([h_final, h0], -1)

        h = self.linear3(h_final)
        h = self.act(h)
        
        h = self.linear4(h)
        h = self.act(h)


        tmpscores = self.linear8(data.xLx_batch)
        tmpscores = self.act(tmpscores)
        tmpscores = self.linear9(tmpscores)
        tmpscores = self.act(tmpscores)
        scores = torch.zeros([len(data.features_list), 1])
        for i, node_belong in enumerate(data.node_belong):
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
