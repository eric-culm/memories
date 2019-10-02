#dummy job for slurm
import time
import torch

a = torch.empty.cuda()
while True:
    print ('hey')
    time.sleep(1000)
