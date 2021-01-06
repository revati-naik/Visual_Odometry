import matplotlib.pyplot as plt
import numpy as np 
import os
import cv2
import utils
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

np.set_printoptions(precision=6,suppress=True)

# def test():
# 	for :
# 		print(i)
# 		i=i+3


def EstimateFundamentalMatrix(x1,x2):

	m1 = np.mean(x1,axis=0)
	m2 = np.mean(x2,axis=0)

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

	# # Check Normalization
	# m1_m = np.mean(x1,axis=0)
	# m2_m = np.mean(x2,axis=0)

	# print("Centroid 1:",m1_m )
	# print("Centroid 2:",m2_m )

	# x1_m = x1[:,:2] - m1_m[:2]
	# x2_m = x2[:,:2] - m2_m[:2]

	# print("Average Distance to Centroid x1 :", np.mean(np.sqrt(np.sum(x1_m*x1_m,axis=1))))
	# print("Average Distance to Centroid x2 :", np.mean(np.sqrt(np.sum(x2_m*x2_m,axis=1))))

	# A = []
	# for i in range(x1.shape[0]):
	# 	A.append([x1[i,0]*x2[i,0],
	# 			  x1[i,0]*x2[i,1],
	# 			  x1[i,0],
	# 			  x1[i,1]*x2[i,0],
	# 			  x1[i,1]*x2[i,1],
	# 			  x1[i,1],
	# 			  x2[i,0],
	# 			  x2[i,1],
	# 			  1])
	# A =  np.array(A)
	A = np.vstack([x1[:,0]*x2[:,0],
					x1[:,0]*x2[:,1],
					x1[:,0],
					x1[:,1]*x2[:,0],
					x1[:,1]*x2[:,1],
					x1[:,1],
					x2[:,0],
					x2[:,1],
					np.ones(x1.shape[0])])
	
	A = A.T


	U,S,V = np.linalg.svd(A)
	F = V.T[:,-1]
	F = np.reshape(F,(3,3))
	U_f,S_f,V_f = np.linalg.svd(F)
	S_f[2]=0
	F = np.matmul(U_f,np.matmul(np.diag(S_f),V_f))
	F_n = np.matmul(T2.T,np.matmul(F,T1))
	return F_n/F_n[-1,-1]

def main():

	img_path = '../Data/stereo/centre/'
	imgs = sorted(os.listdir(img_path))
	fx,fy,cx,cy,G_camera_image,LUT = ReadCameraModel('../Data/model')
	i = 100
	img1 = cv2.imread(img_path+imgs[i],-1)
	img1 = cv2.cvtColor(img1,cv2.COLOR_BAYER_GR2BGR)
	img1 = UndistortImage(img1,LUT)

	img2 = cv2.imread(img_path+imgs[i+1],-1)
	img2 = cv2.cvtColor(img2,cv2.COLOR_BAYER_GR2BGR)
	img2 = UndistortImage(img2,LUT)


	x1,x2 = utils.getMatchingFeaturePoints(img1,img2)

	x1 = np.hstack([x1,np.ones((x1.shape[0],1))])
	x2 = np.hstack([x2,np.ones((x2.shape[0],1))])

	print(x1.shape)
	print(x2.shape)

	F= EstimateFundamentalMatrix(x1,x2)
	print("Calculated")
	print(F)
	# print(F_n)
	print("CV2")
	F,_ = cv2.findFundamentalMat(x1,x2)
	print(F)


if __name__ == '__main__':
	# main()
	test()

