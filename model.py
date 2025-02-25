import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_max_pool
import torch.nn.init as init

class ConvexHullModel(torch.nn.Module):
    def __init__(self, point_dim):
        super().__init__()
        hidden_dim = 32
        self.conv1 = GraphConv(point_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GraphConv(hidden_dim // 2, hidden_dim // 4)
        self.lin = Linear(hidden_dim // 4, 1)
        
    def _initialize_weights(self): # Initialization is important for training
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)
        init.xavier_uniform_(self.conv3.weight)
        init.xavier_uniform_(self.lin.weight)
        if self.conv1.bias is not None:
            init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            init.zeros_(self.conv2.bias)
        if self.conv3.bias is not None:
            init.zeros_(self.conv3.bias)
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        
        x = global_max_pool(x, batch)
        x = self.lin(x)
        
        return x.squeeze(1)