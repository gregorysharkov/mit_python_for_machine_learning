import numpy as np

f = np.array([1,3,-1,1,-3])
g = np.array([1,0,-1])

print(np.convolve(g, f, "same"))

from scipy.ndimage import convolve
data = np.array([
    [1,2,1],
    [2,1,1],
    [1,1,1]
])

kernel = np.array([
    [1,0.5],
    [0.5,1]
])

print(np.sum(convolve(data, kernel, mode="reflect")))


