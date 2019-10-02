#dummy job for slurm
import time
import torch
import torch.utils.data as utils
from torch import optim
from torch import nn

device = torch.device('cpu')

num_epochs = 10
data = torch.randn(100,100).float()

dataset = utils.TensorDataset(data,data)
dataloader = utils.DataLoader(dataset, 10, shuffle=True, pin_memory=True)

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
for epoch in range(num_epochs):
    for i, (sounds, truth) in enumerate(dataloader):
        sounds = sounds.to(device)
        truth = truth.to(device)
        optimizer.zero_grad()
        outputs = model(sounds)
        loss = criterion(outputs, truth)
        loss.backward()
        optimizer.step()
        print (loss)
    time.sleep(5)
