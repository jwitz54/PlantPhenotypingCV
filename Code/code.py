"""
Script that applies morphological filtering and watershed algorithm to an image.
Input image is specified as a command line input and stored in argv

An arbitrary number of morphological filtering (and any other preprocessing algorithm)
can be applied to the image before the watershed algorithm is applied

TODO: incorporate scoring functionality and output scoring statistics to .txt file
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import csv
import os

np.set_printoptions(threshold=np.nan)
def main(fid):
	# -------------SEGMENTATION--------------
	# Read files
	img = cv2.imread('../Images/orig/orig' + fid + '.png')
	img_orig = np.copy(img)

	img_ground = cv2.imread('../Images/truth/truth' + fid + '.png')
	
	# Set lower and upper green values
	lower_green = np.array([0, 150, 0])
	upper_green = np.array([150, 255, 150])

	# Remove pixels not in range
	bw_image = cv2.inRange(img, lower_green, upper_green)

	# -------- DEFINE MORPHOLOGICAL FILTERING COMBOS BELOW ------- #

	# -------------- MORPH 1 --------------
	# dilation, erosion, opening
	# Define kernels
	kernelOpen1 = np.ones((6,6),np.uint8)
	kernelDilate1 = np.ones((5,5),np.uint8)
	kernelErode1 = np.ones((3,3),np.uint8)
	dilation1 = cv2.dilate(bw_image,kernelDilate1,iterations = 1) 
	erosion1 = cv2.erode(dilation1,kernelErode1,iterations = 4)
	opening1 = cv2.morphologyEx(erosion1, cv2.MORPH_OPEN, kernelOpen1, iterations = 1)

	morphed_img1 = opening1

	# -------------- MORPH 2 --------------
	# opening, dilation, erosion
	# Define kernels
	kernelOpen2 = np.ones((6,6),np.uint8)
	kernelDilate2 = np.ones((5,5),np.uint8)
	kernelErode2 = np.ones((3,3),np.uint8)
	opening2 = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernelOpen2, iterations = 1)
	dilation2 = cv2.dilate(opening2,kernelDilate2,iterations = 1) 
	erosion2 = cv2.erode(dilation2,kernelErode2,iterations = 4)

	morphed_img2 = erosion2

	# -------------- MORPH 3 --------------
	# dilation, opening, erosion
	# Define kernels
	kernelOpen3 = np.ones((12,12),np.uint8)
	kernelDilate3 = np.ones((3,3),np.uint8)
	kernelErode3 = np.ones((3,3),np.uint8)
	dilation3 = cv2.dilate(bw_image,kernelDilate3,iterations = 1)
	opening3 = cv2.morphologyEx(dilation3, cv2.MORPH_OPEN, kernelOpen3, iterations = 1)
	erosion3 = cv2.erode(opening3,kernelErode3,iterations = 4)

	morphed_img3 = erosion3

	# -------------- MORPH 4 --------------
	kernelMean4 = np.ones((3,3),np.float32)/9	#low pass
	kernelDilate4 = np.ones((5,5),np.uint8)
	kernelErode4 = np.ones((3,3),np.uint8)
	mean4 = cv2.filter2D(bw_image,-1,kernelMean4)
	dilation4 = cv2.dilate(mean4,kernelDilate4,iterations = 1) 
	erosion4 = cv2.erode(dilation4,kernelErode4,iterations = 4)

	morphed_img4 = erosion4

	# get outputs for each combo
	markers1, img1, leave_centers1, contours1 = get_centers(img_orig,morphed_img1,bw_image)
	markers2, img2, leave_centers2, contours2 = get_centers(img_orig,morphed_img2,bw_image)
	markers3, img3, leave_centers3, contours3 = get_centers(img_orig,morphed_img3,bw_image)
	markers4, img4, leave_centers4, contours4 = get_centers(img_orig,morphed_img4,bw_image)
	
	# score each image and record
	score1 = scoreImg(markers1, img_ground)
	score2 = scoreImg(markers2, img_ground)
	score3 = scoreImg(markers3, img_ground)
	score4 = scoreImg(markers4, img_ground)
	
	scores = [score1,score2,score3,score4]
	if type(scores[0]) == list:
		scores = [j for i in scores for j in i] #flatten 2D list
	scores = [str(x) for x in scores]
	data = [fid] + scores

	#append score to scores.csv
	with open('../scores.csv','a') as fin:
		writer = csv.writer(fin)
		writer.writerow(data)

	# Display and save images
	if sys.argv[2] == '-w':
		directory = '../images' + sys.argv[1]
		if not os.path.exists(directory):
			os.makedirs(directory)

		
		cv2.imshow('original', img_orig)
		cv2.waitKey(0)
		cv2.imshow('original bw', bw_image)
		cv2.waitKey(0)
		cv2.imshow('img D-E-O', img1)
		cv2.imwrite(directory + '/im1_1.png',dilation1)
		cv2.imwrite(directory + '/im1_2.png',erosion1)
		cv2.imwrite(directory + '/im1_3.png',opening1)
		cv2.imwrite(directory + '/im1_4.png',img1)
		cv2.waitKey(0)
		cv2.imshow('img O-D-E', img2)
		cv2.imwrite(directory + '/im2_1.png',opening2)
		cv2.imwrite(directory + '/im2_2.png',dilation2)
		cv2.imwrite(directory + '/im2_3.png',erosion2)
		cv2.imwrite(directory + '/im2_4.png',img2)
		cv2.waitKey(0)
		cv2.imshow('img D-O-E', img3)
		cv2.imwrite(directory + '/im3_1.png',dilation3)
		cv2.imwrite(directory + '/im3_2.png',opening3)
		cv2.imwrite(directory + '/im3_3.png',erosion3)
		cv2.imwrite(directory + '/im3_4.png',img3)
		cv2.waitKey(0)
		cv2.imshow('img M-D-E', img4)
		cv2.imwrite(directory + '/im4_1.png',mean4)
		cv2.imwrite(directory + '/im4_2.png',dilation4)
		cv2.imwrite(directory + '/im4_3.png',erosion4)
		cv2.imwrite(directory + '/im4_4.png',img4)
		cv2.waitKey(0)
	
	
	sys.exit()

def get_centers(img,morphed_img,bw_image):
	"""
	Find contours and apply Watershed algorithm to morphologically filtered binary image
	"""
	# Find contours
	im2, contours, hierarchy = cv2.findContours(morphed_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Separate and fill contours
	img_contours = []
	for i, c in enumerate(contours):
		img_contour = np.zeros(img.shape)
		img_contour = cv2.drawContours(img_contour, contours, i, (255,255,255), 3)
		img_contour = cv2.fillPoly(img_contour, pts = [contours[i]], color = (255,255,255))
		img_contours.append(img_contour)

	# Distance transform and then threshold individual leaves
	leave_centers = np.zeros(bw_image.shape)
	leave_centers = leave_centers.astype(np.uint8)
	for i, ic in enumerate(img_contours):

		# Convert to greyscale unsigned int
		ic = ic.astype(np.uint8)
		ic = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)

		# Perform and normalize distance transform
		dist_transform = cv2.distanceTransform(ic,cv2.DIST_L2,3)
		dist_transform_norm = dist_transform
		cv2.normalize(dist_transform, dist_transform_norm, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
		dist_transform_norm = dist_transform_norm.astype(np.uint8)

		# Threshold and OR images
		ret,thresh1 = cv2.threshold(dist_transform_norm,210,255,cv2.THRESH_BINARY)
		thresh1 = thresh1.astype(np.uint8)
		leave_centers = cv2.bitwise_or(leave_centers, thresh1)

	# Perform region growing
	ret, markers = cv2.connectedComponents(leave_centers)
	
	markers = cv2.watershed(img,markers)

	# print (markers)
	# markers = markers+1
	# markers = markers*20
	# markers = markers.astype(np.uint8)
	# cv2.imshow("marker", markers)


	# cv2.applyColorMap(markers,imC, cv2.COLORMAP_JET)
	# print (markers)
	# cv2.imshow("yay", img)
	# cv2.waitKey(0)
	# print (markers)
	

	#copy to new np array to preserve original image and allow for multiple combinations
	img_marked = np.copy(img)	
	img_marked[markers == -1] = [255,0,0]

	return markers, img_marked, leave_centers, contours

def scoreImg(markers, truth):
	"""
	Takes in labeled image and ground truth image and computes the Dice coeffeicient between them
	1: Returns the mean
	"""
	set1 = set( tuple(v) for m2d in truth for v in m2d ) # change this line/understand it better
	# print (set1)
	set1.discard((0,0,0))

	maxDiceArray = []
	for color in set1:
		newLeaf = np.all(truth == color, axis=-1)

		# img debug 
		"""
		newLeaf1 = newLeaf
		newLeaf1 = newLeaf1 * 250
		newLeaf1 = newLeaf1 + 1
		newLeaf1 = newLeaf1.astype(np.uint8)
		cv2.imshow("wow", newLeaf1)
		cv2.waitKey(0)

		"""


		# truthBoolLeaf = np.asarray(newLeaf).astype(np.bool)
		truthBoolLeaf = newLeaf.astype(np.bool)
		# print("truth bool leaf")
		# print (truthBoolLeaf)

		data = np.unique(markers)
		diceArray = []

		# print ("YAYAYAYAYAYYAY")
		# print (markers)
		for label in np.unique(markers):
			# print("label: {}", label)
			testLeaf = (markers == label)

			# img debug######
			# testLeaf1 = testLeaf
			# testLeaf1 = testLeaf1 * 250
			# testLeaf1 = testLeaf1 + 1
			# testLeaf1 = testLeaf1.astype(np.uint8)
			# cv2.imshow("test test", testLeaf1)
			# cv2.waitKey(0)
			########

			testBoolLeaf = testLeaf.astype(np.bool)

			union = np.logical_and(truthBoolLeaf, testBoolLeaf)

			# print ("UNION WOOH")
			# print (union.sum())
			# print (truthBoolLeaf.sum())
			# print(testBoolLeaf.sum())

			# total = (2*union.sum())/float((truthBoolLeaf.sum() + testBoolLeaf.sum()))
			# print(total)

			diceArray.append(2*union.sum()/float((truthBoolLeaf.sum() + testBoolLeaf.sum())))

		# print ("color: {} and dice".format(color))
		# print (diceArray)
		# print (max(diceArray))
		maxDiceArray.append(max(diceArray))

		# print(newLeaf)

		# newLeaf = np.copy(truth)
		# for i in range(len(newLeaf)):
		# 	# print (newLeaf[i])
		# 	# print(color)
		# 	if (cmp(newLeaf[i],color) == 1):
		# 		newLeaf[i] = [255,255,255]
		# 	else:
		# 		newLeaf[i] = [0,0,0]
		
		# print(np.where(newLeaf == color))
		# newLeaf[np.where(newLeaf == color)] = [255, 255, 255]
		# newLeaf[color] = [255,0,0]
	# print (maxDiceArray)
	print (np.mean(maxDiceArray))
	return np.mean(maxDiceArray)



if __name__ == "__main__":
	"""Driver"""
	if len(sys.argv) < 2:
		print 'Please specify an image ID'
		sys.exit()
	try:
		int(sys.argv[1])
	except:
		print 'Specify the image ID as a positive integer between 1 and 186'

	main(sys.argv[1])