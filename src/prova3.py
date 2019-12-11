import numpy as np
import matplotlib.pyplot as plt

input = np.arange(100) / 100

input = input * 2

bound = 8

input = input ** np.sqrt(bound)

plt.plot(input)
plt.show()
