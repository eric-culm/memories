import define_models as def_mod
import numpy as np
import torch

device = torch.device('cuda:0')

encoder, p = def_mod.WAVE_encoder(1,1)
encoder = encoder.to(device)

input = np.random.rand(100)
input = input.reshape(1,100)
input = torch.tensor(input).cuda()

y = encoder(input)
print (y.shape)
