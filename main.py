import torch
from torch.nn import L1Loss
from torch_geometric.loader import DataLoader
from dataset.dataset import ConvexHullDataset
from model import ConvexHullModel

train_dataset = ConvexHullDataset(root='dataset/train.json')
test_dataset = ConvexHullDataset(root='dataset/test.json')
train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

model = ConvexHullModel(train_dataset.dimension)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
criterion = L1Loss()

def compute_eval_metric(model, dataloader, criterion):
    model.eval()
    losses = []
    for sample in dataloader:
        pred = model(sample)
        loss = criterion(pred, sample.y)
        losses.append(loss)
    return torch.stack(losses, dim=0).mean().item()

for epoch in range(300):
    model.train()
    losses = []
    for sample in train_loader:
        optimizer.zero_grad()
        pred = model(sample)
        loss = criterion(pred, sample.y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    if epoch % 10 == 0:
        eval_loss = compute_eval_metric(model, test_loader, criterion)
        train_loss = torch.stack(losses, dim=0).mean().item()
        print(f'Loss {epoch} -> Train: {train_loss:.2f} / Test: {eval_loss:.2f}')