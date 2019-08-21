import define_models as def_mod
import numpy as np
import torch

encoder, p = def_mod.WAVE_encoder(1,1)

input = np.random.rand(100)
input = torch.tensor(input).cuda()

y = encoder(input)
print (y.shape)
