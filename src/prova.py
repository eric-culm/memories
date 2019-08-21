import define_models as def_mod
import numpy as np

encoder, p = def_mod.WAVE_encoder(1,1)

input = np.random.rand(100)

y = encoder(input)
print (y.shape)
