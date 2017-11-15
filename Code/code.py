import numpy as np
import cv2
from matplotlib import pyplot as plt

# -------------SEGMENTATION--------------
# Read file
img = cv2.imread('../Images/plant2.png')

# Set lower and upper green values
lower_green = np.array([0, 150, 0])
upper_green = np.array([150, 255, 150])

# Remove pixels not in range
green_image = cv2.inRange(img, lower_green, upper_green)

# Morphological operations
# 	Open to remove background noise
#	Dilate to remove foreground noise
#	Erode to decrease size
kernelOpen = np.ones((6,6),np.uint8)
kernelDilate = np.ones((5,5),np.uint8)
kernelErode = np.ones((3,3),np.uint8)
openning = cv2.morphologyEx(green_image, cv2.MORPH_OPEN, kernelOpen)
dilation = cv2.dilate(openning,kernelDilate,iterations = 1) 
erosion = cv2.erode(dilation,kernelErode,iterations = 3)
res = cv2.bitwise_and(img, img, mask = erosion)

# Display images
cv2.imshow('orignal',img)
cv2.imshow('res',res)
cv2.waitKey(0)
