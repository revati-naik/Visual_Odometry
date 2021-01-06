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

sys.dont_write_bytecode=True


def chi(x):
	X = np.array([	[0,-x[2],x[1]],
					[x[2],0,-x[0]],
					[-x[1],x[0],0]])
	return X

def linearTriangulation(K,C1,C2,R1,R2,x1,x2):
	C1 = np.reshape(C1,(3,1))
	C2 = np.reshape(C2,(3,1))

	X=[]
	P1 = np.matmul(K,np.hstack([R1,np.matmul(-R1,C1)]))
	P2 = np.matmul(K,np.hstack([R2,np.matmul(-R2,C2)]))

	for i in range(x1.shape[0]):
		A1 = x1[i,0]*P1[2,:]-P1[0,:]
		A2 = x1[i,1]*P1[2,:]-P1[1,:]
		A3 = x2[i,0]*P2[2,:]-P2[0,:]
		A4 = x2[i,1]*P2[2,:]-P2[1,:]
		A = [A1, A2, A3, A4]
		U,S,V = np.linalg.svd(A)
		V = V.T[:,-1]
		V = V/V[-1]
		X.append(V)

	X = np.array(X)
	return X

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
	x1_in,x2_in = RANSAC.getInliersRANSAC(features,threshold=(0.005),size=8,num_inliers=0.6*features.shape[0],num_iters=200)
	fund_mtx = fundamental.EstimateFundamentalMatrix(x1_in,x2_in)
	K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

	ess_mtx = essential.EssentialMatrixFromFundamentalMatrix(fund_mtx,K)

	
	C,R = cameraPose.ExtractCameraPose(ess_mtx)

	X_tri = []
	for j in range(4):
		X_tri.append(linearTriangulation(K,np.zeros(3),C[j],np.eye(3),R[j],x1_in,x2_in))
	X_tri = np.array(X_tri)
	print(X_tri.shape)

	sys.exit(0)




if __name__=='__main__':
	main()