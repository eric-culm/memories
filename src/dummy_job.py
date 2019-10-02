#dummy job for slurm
import time
import torch

a = torch.empty(100).cuda()
while True:
    a = torch.empty(100).cuda()
    time.sleep(10)
