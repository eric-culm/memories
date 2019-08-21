import define_models as def_mod

encoder, p = def_mod.WAVE_encoder(1,1)

in = np.random.rand(100)

y = encoder(in)
print (y.shape)
