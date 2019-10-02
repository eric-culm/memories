#dummy job for slurm
import time
import torch

a = torch.empty.cuda()
while True:
    time.sleep(1000)
