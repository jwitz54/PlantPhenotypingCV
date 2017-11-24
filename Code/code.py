import numpy as np
import cv2
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)

# -------------SEGMENTATION--------------
# Read file
img = cv2.imread('../Images/plant2.png')

# Set lower and upper green values
lower_green = np.array([0, 150, 0])
upper_green = np.array([150, 255, 150])

# Remove pixels not in range
bw_image = cv2.inRange(img, lower_green, upper_green)

# Define kernels
kernelOpen = np.ones((6,6),np.uint8)
kernelDilate = np.ones((5,5),np.uint8)
kernelErode = np.ones((3,3),np.uint8)

# Morph to remove noise
openning = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernelOpen, iterations = 1)
dilation = cv2.dilate(openning,kernelDilate,iterations = 1) 
erosion = cv2.erode(dilation,kernelErode,iterations = 4)

# Find contours
im2, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

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
markers = markers+1
# markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
markers_norm = markers
cv2.normalize(markers, markers_norm, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
markers_norm = markers_norm.astype(np.uint8)

# Display images
cv2.imshow('di', markers)
# cv2.imshow('res', thresh)
cv2.waitKey(0)
