import matplotlib.pyplot as plt
import numpy as np 
import os
import cv2
import utils
import EstimateFundamentalMatrix as fundamental
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import sys


# plt.ion()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        # color = tuple(np.random.randint(0,255,3).tolist())
        color = (255,0,0)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
        img1 = cv2.circle(img1,tuple(pt1),10,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),10,color,-1)
    return img1,img2


def drawFeatures(img1,img2,x1,x2,c,lw):
	img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
	img = np.hstack([img1,img2])
	for i in range(x1.shape[0]):
		x_values = [x1[i,0],x2[i,0]+img1.shape[1]]
		y_values = [x1[i,1],x2[i,1]]
		plt.plot(x_values,y_values,c=str(c),linewidth = lw)
	plt.imshow(img,cmap='gray')


def getInliersRANSAC(features,threshold,size,num_inliers,num_iters=100):
	S_in = []
	S_in = np.array(S_in)

	for i in range(0,num_iters):
	# for i in range(0,1):
		random_sample = np.random.random_integers(low=0,high=features.shape[0]-1,size=size)
		temp_features = features[random_sample]
		# print("Random random_sample",features[random_sample])
		x1_r = temp_features[:,:3]
		x2_r = temp_features[:,3:]
		F = fundamental.EstimateFundamentalMatrix(x1_r,x2_r)
		# print(F)

		x1 = features[:,:3]
		x2 = features[:,3:]
		# print(x1.shape)
		error = np.diag(np.matmul(x2,np.matmul(F,x1.T)))
		error = abs(error)
		# error = np.sort(error)
		# print(error)
		# plt.plot(error)
		# plt.show()
		# plt.pause(0.05)

		# print(len(error[error<threshold]))
		if len(error[error<threshold])>S_in.shape[0]:
			S_in = features[np.where(error<threshold)]
			F_in = F
	# print(S_in.shape)
	# print(F_in)

	x1_in = S_in[:,:3]
	x2_in = S_in[:,3:]

	return x1_in,x2_in



def main():

	img_path = '../Data/stereo/centre/'
	imgs = sorted(os.listdir(img_path))
	fx,fy,cx,cy,G_camera_image,LUT = ReadCameraModel('../Data/model')

	i = 150
	img1 = cv2.imread(img_path+imgs[i],-1)
	img1 = cv2.cvtColor(img1,cv2.COLOR_BAYER_GR2BGR)
	img1 = UndistortImage(img1,LUT)

	img2 = cv2.imread(img_path+imgs[i+1],-1)
	img2 = cv2.cvtColor(img2,cv2.COLOR_BAYER_GR2BGR)
	img2 = UndistortImage(img2,LUT)

	print(img1.shape)

	x1,x2 = utils.getMatchingFeaturePoints(img1,img2)

	x1 = np.hstack([x1,np.ones((x1.shape[0],1))])
	x2 = np.hstack([x2,np.ones((x2.shape[0],1))])

	# print(x1.shape)
	# print(x2.shape)
	features = np.hstack([x1,x2])
	# getInliersRANSAC(features,threshold=(0.07),size=8,num_inliers=0.6*features.shape[0],num_iters=1000)
	x1_in,x2_in = getInliersRANSAC(features,threshold=(0.005),size=8,num_inliers=0.6*features.shape[0],num_iters=500)
	print(x1_in.shape)
	# drawFeatures(img1,img2,x1,x2,'r',0.3)
	# drawFeatures(img1,img2,x1_in,x2_in,'g',1)
	# plt.show()

	# sys.exit(0)

	fund_mtx = fundamental.EstimateFundamentalMatrix(x1_in,x2_in)

	# x1 = x1_in
	# x2 = x2_in

	# print("CV2 Fundamental Matrix :",correct_fund_mtx[0])

	print("Fundamental Matrix RANSAC:")
	print(fund_mtx)

	# img1 =cv2.imread('../Data/1.jpg')
	# img2 =cv2.imread('../Data/2.jpg')

	pts1 = np.int32(x1[:,:2])
	pts2 = np.int32(x2[:,:2])
	F,_ = cv2.findFundamentalMat(x1,x2,cv2.RANSAC)
	
	drawFeatures(img1,img2,x1,x2,'r',0.1)
	drawFeatures(img1,img2,x1_in,x2_in,'g',0.5)
	plt.show()

	print("F")
	print(F)
	
	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	# lines1 = [lines1[15]]
	# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
	img5,img6 = drawlines(img1,img2,lines1[25:30],pts1[25:30],pts2[25:30])

	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	# lines2 = [lines2[25]]
	# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
	img3,img4 = drawlines(img2,img1,lines2[25:30],pts2[25:30],pts1[25:30])

	plt.subplot(221),plt.imshow(img5)
	plt.subplot(222),plt.imshow(img3)
	# plt.show()

	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,fund_mtx)
	lines1 = lines1.reshape(-1,3)
	# lines1 = [lines1[25]]
	# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
	img5,img6 = drawlines(img1,img2,lines1[25:30],pts1[25:30],pts2[25:30])


	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,fund_mtx)
	lines2 = lines2.reshape(-1,3)
	# lines2 = [lines2[25]]
	# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
	img3,img4 = drawlines(img2,img1,lines2[25:30],pts2[25:30],pts1[25:30])

	plt.subplot(223),plt.imshow(img5)
	plt.subplot(224),plt.imshow(img3)
	plt.show()


if __name__ =="__main__":
	main()