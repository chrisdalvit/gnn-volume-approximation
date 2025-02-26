import torch
from torch.nn import L1Loss
from torch_geometric.loader import DataLoader
from dataset.dataset import ConvexHullDataset
from model import ConvexHullModel
import matplotlib.pyplot as plt

undirected = False
normalize = True
batch_size = 32
train_dataset = ConvexHullDataset(root='dataset/train.json', undirected=undirected, normalize=normalize)
test_dataset = ConvexHullDataset(root='dataset/test.json', undirected=undirected, normalize=normalize) 
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

train_losses = []
eval_losses = []
for epoch in range(250):
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
        eval_losses.append(eval_loss)
        train_losses.append(train_loss)
        print(f'Loss {epoch} -> Train: {train_loss:.2f} / Test: {eval_loss:.2f}')
        
torch.save(model.state_dict(), 'model.pth')
plt.plot(train_losses, label='Train')
plt.plot(eval_losses, label='Eval')
plt.show()