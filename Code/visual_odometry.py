import cv2
import numpy as np 
import matplotlib.pyplot as plt

import ReadCameraModel
import UndistortImage
import featureMatch
import fundamentalMatrix
import inliersRansac


def main():
	########## DATA PREPARATION ##########################
	fig, ax = plt.subplots(3,2)
	# Read the image in Bayer form
	img_bayer = cv2.imread('../Oxford_dataset/stereo/centre/1399381509571009.png')
	img_bayer_next =  cv2.imread('../Oxford_dataset/stereo/centre/1399381571937696.png')
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
	# print(fx, fy, cx, cy, G_cam_img, lut)

	# Undistort Image
	img_undistort = UndistortImage.UndistortImage(image=img_bayer,LUT=lut)
	img_undistort_next = UndistortImage.UndistortImage(image=img_bayer_next,LUT=lut)
	# cv2.imshow("img_undistort", img_undistort)
	ax[2][0].imshow(img_undistort)
	ax[2][1].imshow(img_undistort_next)
	# plt.show()

	######### PIPELINE ##################################

	# Feature matching: point correspondence 
	features = featureMatch.featureMatch(img_1=img_undistort, img_2=img_undistort_next)
	x1 = features[:,:3]
	x2 = features[:,3:]
	# 
	f = fundamentalMatrix.estimateFundamentalMatrix(x1=x1, x2=x2)

	print("Fundamnetal Matrix", f)



if __name__ == '__main__':
	 main()