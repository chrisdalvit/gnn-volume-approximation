import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class ConvexHullModel(torch.nn.Module):
    def __init__(self, point_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(point_dim, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.regression = Linear(2, 1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        return self.regression(h).sum()