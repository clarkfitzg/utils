import numpy as np

npts = 100

norm = np.random.randn(npts)

np.savetxt('normal.csv', norm, delimiter=',')
