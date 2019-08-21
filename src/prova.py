import define_models as def_mod
import numpy as np
import torch

device = torch.device('cuda:0')

encoder, p = def_mod.WAVE_encoder(1,1)
encoder = encoder.to(device)
decoder, p = def_mod.WAVE_decoder(1,1)
decoder = decoder.to(device)


input = np.random.rand(100)
input = input.reshape(1,100)
input = torch.tensor(input).float().to(device)
y = encoder(input)
print ('ENCODED')
print (y.shape)



input1 = np.random.rand(16348)
input1 = input1.reshape(1,1,16348)
input1 = torch.tensor(input1).float().to(device)
y1 = decoder(input1)
print ('DECODED')
print (y1.shape)
