import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_add_pool

class ConvexHullModel(torch.nn.Module):
    def __init__(self, point_dim):
        super().__init__()
        #torch.manual_seed(1234)
        self.conv1 = GCNConv(point_dim, 2)
        self.conv2 = GCNConv(2, 2)
        self.conv3 = GCNConv(2, 2)
        self.regression = Linear(2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()

        h = global_add_pool(h, batch)
        
        return self.regression(h).squeeze(1)