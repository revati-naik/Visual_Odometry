
import sys
import cv2
import numpy as np 
import matplotlib.pyplot as plt

import ReadCameraModel
import UndistortImage
import fundamentalMatrix
import inliersRansac
import essentialMatrix
import cameraPose
import triangulation
import drawLines
import disambiguateCameraPose
import utils

sys.dont_write_bytecode=True

def main():
	########## DATA PREPARATION ##########################
	fig, ax = plt.subplots(3,2)
	# Read the image in Bayer form


	img_bayer = cv2.imread('../Oxford_dataset/stereo/centre/1399381445767267.png')
	img_bayer_next =  cv2.imread('../Oxford_dataset/stereo/centre/1399381445829757.png')
	# cv2.imshow("img_bayer", img_bayer)
	ax[0][0].imshow(img_bayer)
	ax[0][1].imshow(img_bayer_next)


	img_gray = cv2.cvtColor(img_bayer, cv2.COLOR_BGR2GRAY)
	img_gray_next = cv2.cvtColor(img_bayer_next, cv2.COLOR_BGR2GRAY)
	img_color = cv2.cvtColor(img_gray, cv2.COLOR_BayerGR2BGR)
	img_color_next = cv2.cvtColor(img_gray_next, cv2.COLOR_BayerGR2BGR)
	# cv2.imshow("img_color", img_color)
	ax[1][0].imshow(img_color)
	ax[1][1].imshow(img_color_next)


	# Extract camera parameters
	fx, fy, cx, cy, G_cam_img, lut = ReadCameraModel.ReadCameraModel(models_dir='../Oxford_dataset/model')
	print("cam_params: ", fx, fy, cx, cy, G_cam_img, lut)
	# Undistort Image
	img_undistort = UndistortImage.UndistortImage(image=img_color,LUT=lut)
	img_undistort_next = UndistortImage.UndistortImage(image=img_color_next,LUT=lut)
	# cv2.imshow("img_undistort", img_undistort)
	ax[2][0].imshow(img_undistort)
	ax[2][1].imshow(img_undistort_next)
	plt.show()
	sys.exit(0)

	######### PIPELINE ##################################

	# Feature matching: point correspondence 
	features = utils.featureMatch(img_1=img_undistort, img_2=img_undistort_next)
	x1 = features[:,:3]
	x2 = features[:,3:]

	# print("x1", x1[0])
	# print("x2", x2[0])

	pts1 = np.int32(x1[:,:2])
	# pts1 = pts1.reshape(-1,1,2)
	pts2 = np.int32(x2[:,:2])
	# pts2 = pts2.reshape(-1,1,2)

	# print("pts1", pts1[0]) 
	# print("pts2", pts2[0]) 



	x1_inlier, x2_inlier = inliersRansac.getInliersRansac(features=features, threshold=0.02, size=8, num_inliers=0.6*features.shape[0], num_iters=500)

	f = fundamentalMatrix.estimateFundamentalMatrix(x1=x1_inlier, x2=x2_inlier)
	print("Fundamnetal Matrix Normalised: ", f)



	# F,_ = cv2.findFundamentalMat(x1, x2, cv2.RANSAC)
	# print("Fundamental Matrix Direct: ", F)
	
	# for i in range(0,):
	# x2 = x2[3]
	# val_1 = np.matmul(x2.T, np.matmul(F,x1[3]))
	# val_2 = np.matmul(x2.T, np.matmul(f,x1[3]))
	# print("val_1: ", val_1)
	# print("val_2: ", val_2)

	k = np.array([[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]])
	# print("k", k)
	# sys.exit(0)

	e = essentialMatrix.estimateEssentialMatrix(fundamental_matrix=f, camera_matrix=k)

	# print("e", e)
#
	C, R = cameraPose.extractCameraPose(essential_matrix=e)
	print("Position: ", C)
	print("Orientation: ", R)

	world_points = []

	# for i in range(x1_inlier.shape[0]):
	# 	x1 = x1_inlier[i]
	# 	x2 = x2_inlier[i]

	for j in range(0,4):
			X = triangulation.getTrinagulation(camera_matrix=k, c_1=np.zeros(3), c_2=C[j], r_1=np.eye(3), r_2=R[j], x1=x1_inlier, x2=x2_inlier)
			world_points.append(X)

	world_points = np.array(world_points)
	# print("world_points", world_points)
 	

 	C_c, R_c = disambiguateCameraPose.getDisambiguateCameraPose(world_points=world_points, C_set=C, R_set=R)

 	print("Pose Position: ", C_c)
 	print("Pose Orientation: ", R_c)

 	_,R,T,_ = cv2.recoverPose(e,x1_inlier[:,:2],x2_inlier[:,:2])

	print("CV2 Pose Position :", T)
	print("CV2 Pose Orientation :", R)

	###########Epipolar Lines###############

	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,f)
	# lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	# lines2 = lines2.reshape(-1,3)
	img3,img4 = utils.drawlines(img_color,img_color_next,lines1,pts1,pts2)
	# img5,img6 = utils.drawlines(img_color,img_color_next,lines2,pts1,pts2)
	plt.subplot(121),plt.imshow(img3)
	plt.subplot(122),plt.imshow(img4)
	plt.show()
	# plt.subplot(221),plt.imshow(img5)
	# plt.subplot(222),plt.imshow(img6)
	# plt.show()


if __name__ == '__main__':
	 main()