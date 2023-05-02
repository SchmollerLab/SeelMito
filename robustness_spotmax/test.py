import numpy as np

arr = np.zeros((20,10,50,50))

flipped = np.flip(arr, (0,1))

print(flipped.shape)