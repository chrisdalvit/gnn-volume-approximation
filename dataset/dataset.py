import json
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data


class ConvexHullDataset(Dataset):
    
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        with open(self.root) as f:
            self.data = json.loads(f.read())
        
    @property
    def dimension(self):
        return len(self.data[0]['points'][0])    
    
    def len(self):
        return len(self.data)

    def get(self, idx):
        return Data(
            x=torch.index_select(torch.tensor(self.data[idx]['points']), 0, torch.tensor(self.data[idx]['vertices'])) ,
            y=torch.tensor(self.data[idx]['area']),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], 
                                     [1, 0, 2, 1, 0, 2]])
        )