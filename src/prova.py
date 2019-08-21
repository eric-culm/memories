import define_models as def_mod
import numpy as np
import torch

device = torch.device('cuda:0')

encoder, p = def_mod.WAVE_encoder(1,1)
encoder = encoder.to(device)
decoder, p = def_mod.WAVE_decoder(1,1)
decoder = decoder.to(device)
reparametrize, p = def_mod.reparametrize(1,1)
reparametrize = reparametrize.to(device)

vae, p = def_mod.vae(1,1)
vae = vae.to(device)




input1 = np.random.rand(16348)
input1 = input1.reshape(1,1,16348)
input1 = torch.tensor(input1).float().to(device)
y1 = vae(input1)
print ('RECONSTRUCTED')
print (y1.shape)

'''
input1 = np.random.rand(16348)
input1 = input1.reshape(1,1,16348)
input1 = torch.tensor(input1).float().to(device)
y1 = decoder(input1)
print ('DECODED')
print (y1.shape)

input = np.random.rand(100)
input = input.reshape(1,100)
input = torch.tensor(input).float().to(device)
y = encoder(y1)
print ('ENCODED')
print (y.shape)

input2 = np.random.rand(100)
input2 = input2.reshape(1,100)
input2 = torch.tensor(input2).float().to(device)

input3 = np.random.rand(100)
input3 = input3.reshape(1,100)
input3 = torch.tensor(input3).float().to(device)

r = reparametrize(input2, input3)
print ('REPARAMETRIZED')
print (r.shape)
'''
