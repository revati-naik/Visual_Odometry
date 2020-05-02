import numpy as np 
import cv2


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
	# Draw first 10 matches.
	# img3 = cv2.drawMatches(img_color,kp1,img_color_next,kp2,matches[:50], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	# plt.figure()
	# plt.imshow(img3)
	# plt.show()

	return features