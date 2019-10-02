#dummy job for slurm
import time
import torch

a = torch.ones(100000).cuda()
for i in range(10):
    a = torch.add(a,a)
    print (a)
    time.sleep(1)
