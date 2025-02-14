import torch
from torch.nn import L1Loss
from torch_geometric.loader import DataLoader
from dataset.dataset import ConvexHullDataset
from model import ConvexHullModel

dataset = ConvexHullDataset(root='dataset/dataset.json')
dataloader = DataLoader(dataset, batch_size=64)
model = ConvexHullModel(dataset.dimension)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_fn = L1Loss()

for epoch in range(15):
    losses = []
    for sample in dataloader:
        optimizer.zero_grad()
        pred = model(sample)
        loss = loss_fn(pred, sample.y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    print(f'Loss {epoch}: {torch.stack(losses, dim=0).mean().item():.2f}')

for sample in dataloader:
    pred = model(sample)
    print(pred)
    print(sample.y)
    print(sample.y.mean())
    break