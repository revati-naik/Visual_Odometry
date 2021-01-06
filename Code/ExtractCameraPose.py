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

def ExtractCameraPose(E):
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

	U,S,V = np.linalg.svd(E)
	# print("U :",U)
	# print("S :",S)
	# print("V :",V)
	C1 = U[:,2]
	C2 = -U[:,2]
	C3 = U[:,2]
	C4 = -U[:,2]

	R1 = np.matmul(U,np.matmul(W,V))
	R2= np.matmul(U,np.matmul(W,V))
	R3 = np.matmul(U,np.matmul(W.T,V))
	R4 = np.matmul(U,np.matmul(W.T,V))

	# print("det :",np.linalg.det(R1),np.linalg.det(R2),np.linalg.det(R3),np.linalg.det(R4))
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

	

	return C,R

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

	# print(x1.shape)
	# print(x2.shape)
	features = np.hstack([x1,x2])
	# getInliersRANSAC(features,threshold=(0.07),size=8,num_inliers=0.6*features.shape[0],num_iters=1000)
	x1_in,x2_in = RANSAC.getInliersRANSAC(features,threshold=(0.005),size=8,num_inliers=0.6*features.shape[0],num_iters=200)
	fund_mtx = fundamental.EstimateFundamentalMatrix(x1_in,x2_in)
	K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
	
	ess_mtx = essential.EssentialMatrixFromFundamentalMatrix(fund_mtx,K)
	print("Essential Matrix")
	print(ess_mtx)
	
	C,R = ExtractCameraPose(ess_mtx)
	print("Pose Orientation :")
	print(R)

	E_act = cv2.findEssentialMat(x1_in[:,:2],x2_in[:,:2],K)
	# _,R,T,_ = cv2.recoverPose(E_act[0],x1_in[:,:2],x2_in[:,:2])

	# print("Pose Position :")
	# print(T.T)
	# print("Pose Orientation :")
	# print(R)

	R1, R2, T = cv2.decomposeEssentialMat(E_act[0])
	print("OpenCV R1")
	print(R1)
	print("OpenCV R2")
	print(R2)

	print("Calculated Pose Position :")
	print(C)
	print("Opencv T")
	print(T.T)



if __name__=='__main__':
	main()