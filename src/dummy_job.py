#dummy job for slurm
import time
import torch
import torch.utils.data as utils
from torch import optim
from torch import nn
import numpy as np


device = torch.device('cpu')
early_stopping = False
num_epochs = 10000
data = torch.randn(100,100).float()

dataset = utils.TensorDataset(data,data)
dataloader = utils.DataLoader(dataset, 10, shuffle=True, pin_memory=True)
patience = 10

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.hidden = nn.Linear(100, 20)
        self.out = nn.Linear(20, 100)
    def forward(self, X):
        X = self.hidden(X)
        X = self.out(X)
        return X

model = model1().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

model.train()
val_loss_hist = []
patience_vec = []
for epoch in range(num_epochs):
    epoch_hist = []
    for i, (sounds, truth) in enumerate(dataloader):
        sounds = sounds.to(device)
        truth = truth.to(device)
        optimizer.zero_grad()
        outputs = model(sounds)
        loss = criterion(outputs, truth)
        loss.backward()
        epoch_hist.append(loss.detach().numpy())
        optimizer.step()

    print (np.mean(epoch_hist))
    val_loss_hist.append(np.mean(epoch_hist))
    #val_loss_hist.append(1.)


    if early_stopping and epoch >= patience+1:
        patience_vec = val_loss_hist[-patience+1:]
        best_l = np.argmin(patience_vec)
        if best_l == 0:
            print ('Training early-stopped')
            break
