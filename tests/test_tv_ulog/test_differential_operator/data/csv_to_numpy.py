import numpy as np
from skimage.filters import gaussian


f_map = np.loadtxt("test.csv", delimiter=",")

np.save("example_map.npy", f_map)