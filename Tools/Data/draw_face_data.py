import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt("./t3.txt",dtype = np.float32)
t = np.linspace(1,A.shape[0],A.shape[0])
# feature定义
# x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth
# 0, 1, 2  , 3    , 4   , 5   , 6    , 7    , 8     , 9     , 10   
plt.plot(t,A[:,0],label="x")
plt.plot(t,A[:,1],label="y")

plt.legend()
plt.show()