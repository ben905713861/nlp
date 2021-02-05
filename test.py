import numpy as np
import os
import cv2
import copy
import matplotlib.image as mpimg

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=320)

originImage = mpimg.imread('1.jpg')
print(originImage)
im2, contours = cv2.findContours(originImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
