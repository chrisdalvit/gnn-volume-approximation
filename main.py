import torch
from torch.nn import MSELoss
from dataset.dataset import ConvexHullDataset
from model import ConvexHullModel

dataset = ConvexHullDataset(root='dataset/dataset.json')
model = ConvexHullModel(dataset.dimension)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = MSELoss()

def construct_edge_information(graph):
    # Construct edge information
    # Current implementation just connects all vertices 
    # 0 -> 1, 1 -> 2, 2 -> 0
    num_edges = len(graph.x)
    indices = torch.tensor([[0, 1, 2], 
                            [1, 2, 0]])
    values = torch.ones(num_edges)
    return torch.sparse_coo_tensor(indices, values)

for idx, sample in enumerate(dataset[:2000]):
    optimizer.zero_grad()
    sample.edge_index = construct_edge_information(sample)
    pred = model(sample.x, sample.edge_index)
    loss = loss_fn(pred, sample.y)
    loss.backward()
    optimizer.step()
    if idx % 100 == 0:
        print(f'Loss: {loss.item()}')
    