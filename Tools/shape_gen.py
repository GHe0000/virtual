import numpy as np

R = 100
z = 0.34

A = np.ones([5,5])
B = np.array([[2*np.sqrt(2),np.sqrt(3),2,np.sqrt(3),2*np.sqrt(2)],
			  [  np.sqrt(3),np.sqrt(2),1,np.sqrt(2),np.sqrt(3)  ],
			  [           2,         1,0,        1,            2],
			  [  np.sqrt(3),np.sqrt(2),1,np.sqrt(2),np.sqrt(3)  ],
			  [2*np.sqrt(2),np.sqrt(3),2,np.sqrt(3),2*np.sqrt(2)]])

def func(i):
	o = R - np.sqrt(R ** 2 - i ** 2)
	return o

ans = z * A + func(B)
print(ans)