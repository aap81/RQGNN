import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
from torch_geometric.nn import ChebConv

#bm
class GADGNN(nn.Module):
    def __init__(self, featuredim, hdim, nclass, width, depth, dropout, normalize):
        super(GADGNN, self).__init__()

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
        residual = h  # Adding residual connection

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
