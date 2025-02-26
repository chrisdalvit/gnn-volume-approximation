import json
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected


class ConvexHullDataset(Dataset):
    
    def __init__(self, root, undirected, normalize):
        super().__init__(root)
        self.root = root
        self.undirected = undirected
        self.normalize = normalize
        with open(self.root) as f:
            self.data = json.loads(f.read())
        
    @property
    def dimension(self):
        return len(self.data[0]['points'][0])    
    
    def len(self):
        return len(self.data)

    def get(self, idx):
        vs = torch.index_select(torch.tensor(self.data[idx]['points']), 0, torch.tensor(self.data[idx]['vertices']))
        volume = torch.tensor(self.data[idx]['volume'])
        edge_index = torch.tensor([[0, 1, 2], 
                                   [1, 2, 0]])
        
        if self.undirected:
            edge_index = to_undirected(edge_index)
        if self.normalize:
            vs -= vs.mean(dim=0)
        return Data(x=vs, y=volume, edge_index=edge_index)