from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def randomGrid():
	i = np.random.randint(0,64,8)
	r = i/8
	r = r.astype(np.int64)
	c = i%8
	print(r,c)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def getMatchingFeaturePoints(img1,img2):


	img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	good = []
	pts1 = []
	pts2 = []

	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.8*n.distance:
	        good.append(m)
	        pts2.append(kp2[m.trainIdx].pt)
	        pts1.append(kp1[m.queryIdx].pt)


	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)



	return pts1,pts2

def main():
	randomGrid()
	# img_path = '../Data/stereo/centre/'
	# imgs = sorted(os.listdir(img_path))
	# fx,fy,cx,cy,G_camera_image,LUT = ReadCameraModel('../Data/model')
	
	# i=100
	# img1 = cv2.imread(img_path+imgs[i],-1)
	# img1 = cv2.cvtColor(img1,cv2.COLOR_BAYER_GR2BGR)
	# img1 = UndistortImage(img1,LUT)

	# img2 = cv2.imread(img_path+imgs[i+1],-1)
	# img2 = cv2.cvtColor(img2,cv2.COLOR_BAYER_GR2BGR)
	# img2 = UndistortImage(img2,LUT)

	# pts1,pts2 = getMatchingFeaturePoints(img1,img2)

	# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
	# 	# We select only inlier points
	# pts1 = pts1[mask.ravel()==1]
	# pts2 = pts2[mask.ravel()==1]

	# # Find epilines corresponding to points in right image (second image) and
	# # drawing its lines on left image
	# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	# lines1 = lines1.reshape(-1,3)
	# img3,img4 = drawlines(img1,img2,lines1,pts1,pts2)

	# # Find epilines corresponding to points in left image (first image) and
	# # drawing its lines on right image
	# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	# lines2 = lines2.reshape(-1,3)
	# img5,img6 = drawlines(img2,img1,lines2,pts2,pts1)

	# plt.subplot(121),plt.imshow(img5)
	# plt.subplot(122),plt.imshow(img3)
	# plt.show()

if __name__=='__main__':
	main()
