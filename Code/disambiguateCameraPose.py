import numpy as np 


def getDisambiguateCameraPose(world_points, C_set, R_set):
	num_positives = []
	for i in range(4):
		r3 = R_set[i][:,2]
		r3 = np.reshape(r3,(1,3))
		C = C_set[i]
		C = np.reshape(C,(1,3))
		X = world_points[i][:,:3]
		# print("r3 shape",r3.shape)
		# print("C shape",C.shape)
		# print("X shape",X.shape)

		condition = np.matmul(r3,(X-C).T)
		num_positives.append(condition[np.where(condition>0)].shape[0] )
	num_positives = np.array(num_positives)
	index = np.argmax(num_positives)
	return C_set[index],R_set[index]