import cv2
import numpy as np
import sys 
import matplotlib.pyplot as plt

import fundamentalMatrix



def getInliersRansac(features, threshold, size, num_inliers, num_iters):
	inliers = []
	inliers = np.array(inliers)

	for i in range(0,num_iters):
		random_sample = np.random.random_integers(low=0, high=features.shape[0]-1, size=size)
		# print("random_sample",random_sample)
		temp_features = features[random_sample]
		# print("temp_features", temp_features)
		# print("random_sample", random_sample)
		# print("random_sample.shape", random_sample.shape)
		
		x1_temp = temp_features[:,:3]
		x2_temp = temp_features[:,3:]

		F = fundamentalMatrix.estimateFundamentalMatrix(x1=x1_temp, x2=x2_temp)

		x1 = features[:,:3]
		x2 = features[:,3:]

		error = np.diag(np.matmul(x2, np.matmul(F,x1.T)))
		# print(error)
		# print(len(error))
		# sys.exit(0)

		error = abs(error)
		# plt.plot(error)
		# plt.show()
		

		# print(features[8])
		# sys.exit(0)
		# print("len(error)", len(error[error<threshold]))
		# print("inliers.shape[0]", inliers.shape[0])
		if len(error[error<threshold])>inliers.shape[0]:
			# print("np.where(error<threshold)", np.where(error<threshold))
			inliers = features[np.where(error<threshold)]
			# print(inliers.shape)
			F_in = F

		# print(inliers.shape)

	x1_inlier = inliers[:,:3]
	x2_inlier = inliers[:,3:]

	return x1_inlier, x2_inlier




