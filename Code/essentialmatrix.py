import numpy as np 

def estimateEssentialMatrix(fundamental_matrix, camera_matrix):
	E = np.matmul(camera_matrix.T,np.matmul(fundamental_matrix,camera_matrix))
	U,S,V = np.linalg.svd(E)
	S = [1,1,0]
	S = np.diag(S)
	E = np.matmul(U,np.matmul(S,V))
	return E