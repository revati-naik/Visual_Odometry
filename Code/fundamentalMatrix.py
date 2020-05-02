import cv2
import numpy as np 

def estimateFundamentalMatrix(x1, x2):
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
	# print("A shape:",A.shape)

	U,S,V = np.linalg.svd(A)

	# Last column 
	F = V[:,-1]
	F = np.reshape(F,(3,3))
	U_f,S_f,V_f = np.linalg.svd(F)
	S_f[-1]=0
	F = np.matmul(U_f,np.matmul(np.diag(S_f),V_f.T))

	return F