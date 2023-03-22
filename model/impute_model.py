import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphConvolution

import torch
from torch_geometric.nn import GCNConv


class coGCN(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(coGCN, self).__init__()
        nhid1=2048
        nhid2=256
        self.gc1 = GraphConvolution(in_feat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, out_feat)

    def forward(self, x, adj):
        x = nn.LeakyReLU()(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        return x


class pygGCN(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(pygGCN, self).__init__()
        nhid1=2048
        nhid2=256
        self.gc1 = GCNConv(in_feat, nhid1)
        self.gc2 = GCNConv(nhid1, nhid2)
        self.gc3 = GCNConv(nhid2, out_feat)

    def forward(self, x, edge_index,edge_weight):
        x =nn.LeakyReLU()(self.gc1(x, edge_index,edge_weight))
        x = F.relu(self.gc2(x,edge_index,edge_weight))
        x = F.relu(self.gc3(x, edge_index,edge_weight))
        return x