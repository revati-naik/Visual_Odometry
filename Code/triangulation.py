import numpy as np 


def chiralityCondition(x):
	X = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
	return X

def getTrinagulation(camera_matrix, c_1, c_2, r_1, r_2, x1, x2):
	# C1,C2  = C_set[0],C_set[1]
	# R1,R2  = R_set[0],R_set[1]

	c_1 = np.reshape(c_1,(3,1))
	c_2 = np.reshape(c_2,(3,1))

	X = []
	# Ic_1 = np.append(np.eye(3),-1*c_1,axis=1)
	# Ic_2 = np.append(np.eye(3),-1*c_2,axis=1)

	P1 = np.matmul(camera_matrix,np.hstack([r_1, np.matmul(-r_1,c_1)]))
	P2 = np.matmul(camera_matrix,np.hstack([r_2, np.matmul(-r_2,c_2)]))

	# print("IC1 Shape:",IC1)
	# print("IC2 Shape:",IC2)

	# print("P1 Shape :",P1.shape)
	# print("P2 Shape :",P2.shape)

	# A1 = np.matmul(chiralityCondition(x1),P1)
	# A2 = np.matmul(chiralityCondition(x2),P2)

	# print("A1 :",A1)
	# print("A2 :",A2)
	for i in range(x1.shape[0]):
		A1 = x1[i,0]*P1[2,:]-P1[0,:]
		A2 = x1[i,1]*P1[2,:]-P1[1,:]
		A3 = x1[i,0]*P1[2,:]-P1[0,:]
		A4 = x1[i,1]*P1[2,:]-P1[1,:]

		A = [A1, A2, A3, A4]
		U,S,V = np.linalg.svd(A)
		V = V[-1]
		

		V = V/V[-1]
		X.append(V)
	X = np.array(X)
	return X