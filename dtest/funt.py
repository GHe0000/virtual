import numpy as np

def perspective():
    a = np.eye(4, dtype = np.float32)
    a[2, 2] = 1 / 1000
    a[3, 2] = -0.0001
    a[2, 3] = 1
    a[3, 3] = 0
    return a

print(perspective())