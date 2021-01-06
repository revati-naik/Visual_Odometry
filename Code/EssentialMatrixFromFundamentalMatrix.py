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

def EssentialMatrixFromFundamentalMatrix(F,K):
	E = np.matmul(K.T,np.matmul(F,K))
	# print(E)
	U,S,V = np.linalg.svd(E)
	S = [1,1,0]
	S = np.diag(S)
	E = np.matmul(U,np.matmul(S,V))
	return E

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
	x1_in,x2_in = RANSAC.getInliersRANSAC(features,threshold=(0.005),size=8,num_inliers=0.6*features.shape[0],num_iters=1000)
	fund_mtx = fundamental.EstimateFundamentalMatrix(x1_in,x2_in)
	K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

	E = EssentialMatrixFromFundamentalMatrix(fund_mtx,K)
	print("Essential Matrix")
	print(E)

	E_act = cv2.findEssentialMat(x1_in[:,:2],x2_in[:,:2],K)
	print("E actual")
	print(E_act[0])





if __name__=='__main__':
	main()
