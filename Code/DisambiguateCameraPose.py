import matplotlib.pyplot as plt
import numpy as np 
import os
import cv2
import utils
import EstimateFundamentalMatrix as fundamental
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import sys
import GetInlierRANSAC as RANSAC
import EssentialMatrixFromFundamentalMatrix as essential
import ExtractCameraPose as cameraPose
import LinearTriangulation as triangulation

def disambiguateCameraPose(X_set,C_set,R_set):
	num_positives = []
	# C1 = np.array([0,0,1])
	# R1 = np.eye(3)
	for i in range(4):
		r3 = R_set[i][:,2]
		r3 = np.reshape(r3,(1,3))
		C = C_set[i]
		C = np.reshape(C,(1,3))
		X = X_set[i][:,:3]
		condition = np.matmul(r3,(X-C).T)
		# print("r3 shape",r3.shape)
		# print("C shape",C.shape)
		# print("X shape",X.shape)
		n1 = condition[np.where(condition>0)].shape[0] 
		# print(n1)
		# n2=0
		r3 = [0,0,1]
		r3 = np.reshape(r3,(1,3))
		X = X_set[i][:,:3]
		condition = np.matmul(r3,X.T)
		n2 = condition[np.where(condition>0)].shape[0] 
		# print(n2)

		num_positives.append(n1+n2)
	num_positives = np.array(num_positives)
	# print(num_positives)
	index = np.argmax(num_positives)
	# print(index)
	return C_set[index],R_set[index]


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

	features = np.hstack([x1,x2])
	x1_in,x2_in = RANSAC.getInliersRANSAC(features,threshold=(0.005),size=8,num_inliers=0.6*features.shape[0],num_iters=1000)
	print(x1_in.shape)
	fund_mtx = fundamental.EstimateFundamentalMatrix(x1_in,x2_in)
	K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

	ess_mtx = essential.EssentialMatrixFromFundamentalMatrix(fund_mtx,K)

	
	C,R = cameraPose.ExtractCameraPose(ess_mtx)
	# print("Available C :")
	# print(C)

	print("Available R")
	print(R)

	X_tri = []
	for j in range(4):
		X_tri.append(triangulation.linearTriangulation(K,np.zeros(3),C[j],np.eye(3),R[j],x1_in,x2_in))
	X_tri = np.array(X_tri)
	# print(X_tri.shape)
	C_c,R_c = disambiguateCameraPose(X_tri,C,R)

	print("Pose Position :")
	print(C_c)
	print("Pose Orientation :")
	print(R_c)	

	# angle = utils.rotationMatrixToEulerAngles(R_c)
	# print("Angle :")
	# print(angle)

	E_act = cv2.findEssentialMat(x1_in[:,:2],x2_in[:,:2],K)
	_,R,T,_ = cv2.recoverPose(E_act[0],x1_in[:,:2],x2_in[:,:2])

	print("---")
	print("CV2 Pose Position :")
	print(T.T)
	print("CV2 Pose Orientation :")
	print(R)	
	# angle = utils.rotationMatrixToEulerAngles(R)
	# print("CV2 Ang;e:")
	# print(angle)
	print("<----------------->")



if __name__=='__main__':
	main()