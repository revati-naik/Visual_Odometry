import cv2
import numpy as np 
import ReadCameraModel
import UndistortImage

# Read the image in Bayer form
img_bayer = cv2.imread('../Oxford_dataset/stereo/centre/1399381444704913.png')
cv2.imshow("img_bayer", img_bayer)

img_gray = cv2.cvtColor(img_bayer, cv2.COLOR_BGR2GRAY)
img_color = cv2.cvtColor(img_gray, cv2.COLOR_BayerGR2BGR)
cv2.imshow("img_color", img_color)


# Extract camera parameters
fx, fy, cx, cy, G_cam_img, lut = ReadCameraModel.ReadCameraModel(models_dir="../Oxford_dataset/model")
print(fx, fy, cx, cy, G_cam_img, lut)

# Undistort Image
img_undistort = UndistortImage.UndistortImage(image=img_color,LUT=lut)
cv2.imshow("img_undistort", img_undistort)

cv2.waitKey(0)
cv2.waitKey(0)