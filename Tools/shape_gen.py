import numpy as np

R = 100
z = 0.45

A = np.ones([5,5])
B = np.array([[2*np.sqrt(2),np.sqrt(3),2,np.sqrt(3),2*np.sqrt(2)],
			  [  np.sqrt(3),np.sqrt(2),1,np.sqrt(2),np.sqrt(3)  ],
			  [           2,         1,0,        1,            2],
			  [  np.sqrt(3),np.sqrt(2),1,np.sqrt(2),np.sqrt(3)  ],
			  [2*np.sqrt(2),np.sqrt(3),2,np.sqrt(3),2*np.sqrt(2)]])
# B = np.array([[np.sqrt(20),np.sqrt(17),4,np.sqrt(17),np.sqrt(20)],
# 			  [np.sqrt(13),2*np.sqrt(2),3,2*np.sqrt(2),np.sqrt(13)],
# 			  [2*np.sqrt(2),np.sqrt(3),2,np.sqrt(3),2*np.sqrt(2)],
# 			  [  np.sqrt(3),np.sqrt(2),1,np.sqrt(2),np.sqrt(3)  ],
# 			  [           2,         1,0,        1,            2],
# 			  [  np.sqrt(3),np.sqrt(2),1,np.sqrt(2),np.sqrt(3)  ],
# 			  [2*np.sqrt(2),np.sqrt(3),2,np.sqrt(3),2*np.sqrt(2)],
# 			  [np.sqrt(13),2*np.sqrt(2),3,2*np.sqrt(2),np.sqrt(13)],
# 			  [np.sqrt(20),np.sqrt(17),4,np.sqrt(17),np.sqrt(20)],])

def func(i):
	o = R - np.sqrt(R ** 2 - i ** 2)
	return o

ans = z * A + func(B)
print(ans)