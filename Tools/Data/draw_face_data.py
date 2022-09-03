import numpy as np
import matplotlib.pyplot as plt

# 滤波
S = np.zeros((5,11))
def SlidingAverage(i):
    global S
    S = np.delete(S,0,axis=0)
    S = np.vstack((S,i))
    return np.average(S,axis=0)

def fun(input):
    O = []
    for i in range(input.shape[0]):
        #print(np.array(input[i]))
        ans = SlidingAverage(np.array(input[i]))
        O.append(ans)
    return O

A = np.loadtxt("./t2.txt",dtype = np.float32)
t = np.linspace(1,A.shape[0],A.shape[0])

B = np.array(fun(A))

# feature定义
# x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth
# 0, 1, 2  , 3    , 4   , 5   , 6    , 7    , 8     , 9     , 10   
plt.plot(t,B[:,0],label="x")
plt.plot(t,B[:,1],label="y")
# plt.plot(t,B[:,5],label="face")

plt.legend()
plt.show()
