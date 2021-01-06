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
import DisambiguateCameraPose as disambiguateCameraPose
import copy
# import math
plt.ion()


def main():
	img_path = '../Data/stereo/centre/'
	imgs = sorted(os.listdir(img_path))
	fx,fy,cx,cy,G_camera_image,LUT = ReadCameraModel('../Data/model')
	i = 100
	pt_old = np.zeros((3,1))
	pt_old_cv = np.zeros((3,1))
	i=15
	while i<len(imgs):
	# for i in range(15,len(imgs)):
	# for i in range(500,600):
		i=i+3
		
		print(i)
		img1 = cv2.imread(img_path+imgs[i])
		img1 = cv2.cvtColor(img1,cv2.COLOR_BAYER_GR2BGR)
		img1 = UndistortImage(img1,LUT)

		img2 = cv2.imread(img_path+imgs[i+1])
		img2 = cv2.cvtColor(img2,cv2.COLOR_BAYER_GR2BGR)
		img2 = UndistortImage(img2,LUT)


		x1,x2 = utils.getMatchingFeaturePoints(img1,img2)

		x1 = np.hstack([x1,np.ones((x1.shape[0],1))])
		x2 = np.hstack([x2,np.ones((x2.shape[0],1))])

		features = np.hstack([x1,x2])
		x1_in,x2_in = RANSAC.getInliersRANSAC(features,threshold=(0.002),size=8,num_inliers=0.6*features.shape[0],num_iters=200)
		# print("Inliers :",x1_in.shape[0])
		fund_mtx = fundamental.EstimateFundamentalMatrix(x1_in,x2_in)
		K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

		ess_mtx = essential.EssentialMatrixFromFundamentalMatrix(fund_mtx,K)

		
		C,R = cameraPose.ExtractCameraPose(ess_mtx)

		X_tri = []
		for j in range(4):
			X_tri.append(triangulation.linearTriangulation(K,np.zeros(3),C[j],np.eye(3),R[j],x1_in,x2_in))
		X_tri = np.array(X_tri)
		# print(X_tri.shape)
		C_c,R_c = disambiguateCameraPose.disambiguateCameraPose(X_tri,C,R)
		C_c = np.reshape(C_c,(3,1))
		if (C_c[2]<0):
			C_c*=-1

		# print("Pose Position :")
		# print(C_c.T)
		# print("Pose Orientation :")
		# print(R_c)


		pt_new = np.matmul(R_c,pt_old)
		pt_new += C_c

		if abs(pt_new[0]-pt_old[0]) >2:
			pt_new[0] = copy.copy(pt_old[0])
		if abs(pt_new[1]-pt_old[1]) >2:
			pt_new[1] = copy.copy(pt_old[1])
		if abs(pt_new[2]-pt_old[2]) >2:
			pt_new[2] = copy.copy(pt_old[2])

		print("Old point :",pt_old.T)
		print("New point :",pt_new.T)

		plt.figure("Calculated")
		plt.plot([pt_old[0],pt_new[0]],[pt_old[2],pt_new[2]]) 
		pt_old = copy.copy(pt_new)


		# E_act = cv2.findEssentialMat(x1[:,:2],x2[:,:2],K)
		# _,R,T,_ = cv2.recoverPose(E_act[0],x1[:,:2],x2[:,:2])

		# pt_new_cv = np.matmul(R,pt_old_cv)
		# pt_new_cv += T

		# if abs(pt_new_cv[0]-pt_old_cv[0]) >0.5:
		# 	pt_new_cv[0] = copy.copy(pt_old_cv[0])
		# if abs(pt_new_cv[1]-pt_old_cv[1]) >0.5:
		# 	pt_new_cv[1] = copy.copy(pt_old_cv[1])

		# print("Act :",plot_act)
		# print("Calc :",plot_calc)
		# print("---")
		# print("CV2 Pose Position :")
		# print(T.T)
		# print("CV2 Pose Orientation :")
		# print(R)	
		# 
		# print("Pose Position cv2:")
		# print(T.T)
		# print("Pose Orientation cv2 :")
		# print(R)

		# print("Old point :",pt_old_cv.T)
		# print("New point :",pt_new_cv.T)
		# plt.figure("OpenCV")
		# plt.plot([pt_old_cv[0],pt_new_cv[0]],[pt_old_cv[2],pt_new_cv[2]]) 
		# pt_old_cv = copy.copy(pt_new_cv)

		plt.show()
		plt.pause(0.00001)


		print("!-----------------!")


if __name__=='__main__':
	main()
	# genImages() 