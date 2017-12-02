"""
Script that applies morphological filtering and watershed algorithm to an image 
to separate leaves in the image. Each leaf is individually scored against ground
truth. The mean of all the leaves in the image is taken as the image's score.abs

Command line usage: "python code.py id -w"
						  where id specified the ID of the input and ground truth image
						  to be read from ../Images/orig and ../Images/truth, 
						  respectively. The -w flag writes all of the intermediate
						  images to a folder corresponding to the image ID.

Developed for ECE 5470 Fall 2017 by Jeff Witz, Curran Sinha, and Cameron Schultz
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import csv
import os

np.set_printoptions(threshold=np.nan)
def main(fid):
	"""
	main driver function
	"""
	# -------------SEGMENTATION--------------
	# Read files
	img = cv2.imread('../Images/orig/orig' + fid + '.png')
	img_orig = np.copy(img)

	img_ground = cv2.imread('../Images/truth/truth' + fid + '.png')
	
	# Set lower and upper green values
	lower_green = np.array([0, 160, 0])
	upper_green = np.array([60, 255, 130])

	# Remove pixels not in range
	bw_image = cv2.inRange(img, lower_green, upper_green)

	# -------- DEFINE MORPHOLOGICAL FILTERING COMBOS BELOW ------- #

	# -------------- MORPH 1 --------------
	# dilation, erosion, opening
	# Define kernels
	kernelDilate1 = np.ones((5,5),np.uint8)
	kernelErode1 = np.ones((3,3),np.uint8)
	kernelOpen1 = np.ones((6,6),np.uint8)
	dilation1 = cv2.dilate(bw_image,kernelDilate1,iterations = 1) 
	erosion1 = cv2.erode(dilation1,kernelErode1,iterations = 4)
	opening1 = cv2.morphologyEx(erosion1, cv2.MORPH_OPEN, kernelOpen1, iterations = 1)

	morphed_img1 = opening1

	# -------------- MORPH 2 --------------
	# opening, dilation, erosion
	# Define kernels
	kernelOpen2 = np.ones((2,2),np.uint8)
	kernelDilate2 = np.ones((3,3),np.uint8)
	kernelErode2 = np.ones((3,3),np.uint8)
	opening2 = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernelOpen2, iterations = 2)
	dilation2 = cv2.dilate(opening2,kernelDilate2,iterations = 2) 
	erosion2 = cv2.erode(dilation2,kernelErode2,iterations = 4)
	
	morphed_img2 = erosion2

	# -------------- MORPH 3 --------------
	# dilation x 2, opening, erosion
	# Define kernels
	kernelOpen3 = np.ones((12,12),np.uint8)
	kernelDilate3 = np.ones((5,5),np.uint8)
	kernelErode3 = np.ones((3,3),np.uint8)
	dilation3 = cv2.dilate(bw_image,kernelDilate3,iterations = 1)
	dilation3 = cv2.dilate(dilation3,np.ones((2,2),np.uint8),iterations = 2)
	opening3 = cv2.morphologyEx(dilation3, cv2.MORPH_OPEN, kernelOpen3, iterations = 1)
	erosion3 = cv2.erode(opening3,kernelErode3,iterations = 4)
	
	morphed_img3 = erosion3

	# -------------- MORPH 4 --------------
	# mean filter, dilation, erosion
	# Define kernels
	kernelMean4 = np.ones((3,3),np.float32)/9	#low pass
	kernelDilate4 = np.ones((5,5),np.uint8)
	kernelErode4 = np.ones((3,3),np.uint8)
	mean4 = cv2.filter2D(bw_image,-1,kernelMean4)
	dilation4 = cv2.dilate(mean4,kernelDilate4,iterations = 1) 
	erosion4 = cv2.erode(dilation4,kernelErode4,iterations = 4)

	morphed_img4 = erosion4

	# get centers for each combo
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

	# append score to scores.csv
	with open('../scores.csv','a') as fin:
		writer = csv.writer(fin)
		writer.writerow(data)

	# Display and save images if -w flag set
	if len(sys.argv) > 2 and sys.argv[2] == '-w':
		directory = '../images' + sys.argv[1]
		# create unique directory if DNE
		if not os.path.exists(directory):
			os.makedirs(directory)

		cv2.imshow('original', img_orig)
		#cv2.waitKey(0)
		cv2.imshow('original bw', bw_image)
		#cv2.waitKey(0)
		cv2.imshow('img D-E-O', img1)
		cv2.imwrite(directory + '/im1_1.png',dilation1)
		cv2.imwrite(directory + '/im1_2.png',erosion1)
		cv2.imwrite(directory + '/im1_3.png',opening1)
		cv2.imwrite(directory + '/im1_4.png',img1)
		#cv2.waitKey(0)
		cv2.imshow('img O-D-E', img2)
		cv2.imwrite(directory + '/im2_1.png',opening2)
		cv2.imwrite(directory + '/im2_2.png',dilation2)
		cv2.imwrite(directory + '/im2_3.png',erosion2)
		cv2.imwrite(directory + '/im2_4.png',img2)
		#cv2.waitKey(0)
		cv2.imshow('img D-O-E', img3)
		cv2.imwrite(directory + '/im3_1.png',dilation3)
		cv2.imwrite(directory + '/im3_2.png',opening3)
		cv2.imwrite(directory + '/im3_3.png',erosion3)
		cv2.imwrite(directory + '/im3_4.png',img3)
		#cv2.waitKey(0)
		cv2.imshow('img M-D-E', img4)
		cv2.imwrite(directory + '/im4_1.png',mean4)
		cv2.imwrite(directory + '/im4_2.png',dilation4)
		cv2.imwrite(directory + '/im4_3.png',erosion4)
		cv2.imwrite(directory + '/im4_4.png',img4)
		#cv2.waitKey(0)
	
	# exit Python otherwise figure windows 'hang'
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
		ret,thresh1 = cv2.threshold(dist_transform_norm,170,255,cv2.THRESH_BINARY)
		# cv2.imshow("thresh1", dist_transform_norm)
		thresh1 = thresh1.astype(np.uint8)
		leave_centers = cv2.bitwise_or(leave_centers, thresh1)

	# Perform region growing
	ret, markers = cv2.connectedComponents(leave_centers)
	
	# Reduce noise of input image via thresholding
	lower_green = np.array([0, 120, 0])
	upper_green = np.array([100, 255, 142])
	bw = cv2.inRange(img, lower_green, upper_green)
	img_noise_reduction = cv2.bitwise_and(img, img, mask = bw)
	# cv2.imshow("noise", img_noise_reduction)

	markers[2][2] = 30 #set dummy marker to prevent outline of image from being marked

	#pass markers into watershed
	markers = cv2.watershed(img_noise_reduction,markers)
	
	#copy to new np array to preserve original image and allow for multiple combinations
	img_marked = np.copy(img)	
	img_marked[markers == -1] = [255,0,0]
	
	return markers, img_marked, leave_centers, contours

def scoreImg(markers, truth):
	"""
	Takes in labeled image and ground truth image and computes the Dice coeffeicient between them
	1: Returns the mean
	"""
	# get set of all unique color tuples in ground truth image
	set1 = set( tuple(j) for i in truth for j in i )
	set1.discard((0,0,0))

	maxDiceArray = []
	# compute Dice coeff for each unique color in the ground truth image
	for color in set1:
		# isolate color from iamge
		newLeaf = np.all(truth == color, axis=-1)

		# convert to boolean type
		truthBoolLeaf = newLeaf.astype(np.bool)
		
		# get set of all unique grayscale colors in markers
		data = np.unique(markers)
		diceArray = []

		# for each unique detected leaf		
		for label in np.unique(markers):
			# isolate detected leaf from image
			testLeaf = (markers == label)

			# convert to boolean type
			testBoolLeaf = testLeaf.astype(np.bool)

			# compute union between ground truth leaf and detected leaf
			union = np.logical_and(truthBoolLeaf, testBoolLeaf)

			# append Dice coeff for this leaf to diceArray
			diceArray.append(2*union.sum()/float((truthBoolLeaf.sum() + testBoolLeaf.sum())))

		# The detected leaf with the highest Dice coefficient with the 
		# ground truth leaf is the most likely match
		maxDiceArray.append(max(diceArray))

	return np.mean(maxDiceArray)

if __name__ == "__main__":
	"""Driver"""
	if len(sys.argv) < 2:
		print 'Please specify an image ID'
		sys.exit()
	try:
		int(sys.argv[1])
		if int(sys.argv[1] > 159 or sys.argv[1] < 1):
			print 'Specify the image ID as a positive integer between 1 and 159'
	except:
		print 'Specify the image ID as a positive integer between 1 and 159'

	main(sys.argv[1])