import sys
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

sys.dont_write_bytecode=True


def featureMatch(img_1, img_2):
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with orb
	kp1, des1 = orb.detectAndCompute(img_1,None)
	kp2, des2 = orb.detectAndCompute(img_2,None)

	# bf object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)
	# print("des1",des1)
	# print("des2",des2)
	# print("size", des1.shape)
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	# print("matches", matches)
	# print(type(matches))


	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)
	features = []

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt
		# print("points1",points1)
		features.append([points1[i][0], points1[i][1], 1, points2[i][0], points2[i][1], 1])
		# sys.exit(0)

	# features.append(points1, points2)
	features = np.array(features).astype(np.float32)

	# print("features", features[1])
	#Draw first 10 matches.
	# img3 = cv2.drawMatches(img_1,kp1,img_2,kp2,matches[:50], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	# plt.figure()
	# plt.imshow(img3)
	# plt.show()
	# sys.exit(0)
	return features

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
	    
	return img1,img2