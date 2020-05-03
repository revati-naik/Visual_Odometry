import cv2
import numpy as np 

def estimateFundamentalMatrix(x1, x2):
	m1 = np.mean(x1,axis=0)
	m2 = np.mean(x2,axis=0)

	# m1 = [640,480]
	# m2 = [640,480]


	x1_m = x1[:,:2] - m1[:2]
	x2_m = x2[:,:2] - m2[:2]

	s1 = np.mean(np.sqrt(np.sum(x1_m*x1_m,axis=1)))/np.sqrt(2)
	s2 = np.mean(np.sqrt(np.sum(x2_m*x2_m,axis=1)))/np.sqrt(2)


	T1 = np.array([[(1/s1),0,-(m1[0]/s1)],
				  [0,(1/s1),-(m1[1]/s1)],
				  [0,0,1]])

	T2 = np.array([[(1/s2),0,-(m2[0]/s2)],
				  [0,(1/s2),-(m2[1]/s2)],
				  [0,0,1]])

	x1 = np.matmul(T1,x1.T)
	x2 = np.matmul(T2,x2.T)

	x1 = x1.T
	x2 = x2.T

	A = []
	for i in range(x1.shape[0]):
		A.append([x1[i,0]*x2[i,0],
				  x1[i,0]*x2[i,1],
				  x1[i,0],
				  x1[i,1]*x2[i,0],
				  x1[i,1]*x2[i,1],
				  x1[i,1],
				  x2[i,0],
				  x2[i,1],
				  1])
	A =  np.array(A)

	U,S,V = np.linalg.svd(A)

	F = V.T[:,-1]
	F = np.reshape(F,(3,3))
	U_f,S_f,V_f = np.linalg.svd(F)
	S_f[-1]=0
	F = np.matmul(U_f,np.matmul(np.diag(S_f),V_f))
	
	F_n = np.matmul(T2.T,np.matmul(F,T1))
	
	return F_n/F_n[2,2]
