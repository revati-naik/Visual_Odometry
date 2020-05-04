import numpy as np 

def extractCameraPose(essential_matrix):
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

	U,S,V = np.linalg.svd(essential_matrix)

	C1 = U[:,-1]
	C2 = -U[:,-1]
	C3 = U[:,-1]
	C4 = -U[:,-1]

	# print(V.shape)

	R1 = np.matmul(U,np.matmul(W,V))
	R2 = np.matmul(U,np.matmul(W,V))
	R3 = np.matmul(U,np.matmul(W.T,V))
	R4 = np.matmul(U,np.matmul(W.T,V))

	if np.linalg.det(R1)<0:
		R1 = R1*-1
		C1 = C1*-1
	if np.linalg.det(R2)<0:
		R2 = R2*-1
		C2 = C2*-1
	if np.linalg.det(R3)<0:
		R3 = R3*-1
		C3 = C3*-1
	if np.linalg.det(R4)<0:
		R4 = R4*-1
		C4 = C4*-1

	C = np.array([C1,C2,C3,C4])
	R = np.array([R1,R2,R3,R4])

	return C, R