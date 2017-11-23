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

# Morphological operations
# 	Open to remove background noise
#	Dilate to remove foreground noise
#	Erode to decrease size
#	Dist tranform + normalize
#	Adaptive threshold to find centers

# Define kernels
kernelOpen = np.ones((6,6),np.uint8)
kernelDilate = np.ones((5,5),np.uint8)
kernelErode = np.ones((3,3),np.uint8)

# Morph to remove noisee
openning = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernelOpen, iterations = 1)
dilation = cv2.dilate(openning,kernelDilate,iterations = 2) 
erosion = cv2.erode(dilation,kernelErode,iterations = 3)

# Distance transform to get centers 
dist_transform = cv2.distanceTransform(openning,cv2.DIST_L2,3)
dist_transform_norm = dist_transform
cv2.normalize(dist_transform, dist_transform_norm, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
print(dist_transform_norm)
dist_transform_norm = dist_transform_norm.astype(np.uint8)
thresh = cv2.adaptiveThreshold(dist_transform_norm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)

# Display images
cv2.imshow('res', thresh)
cv2.waitKey(0)
