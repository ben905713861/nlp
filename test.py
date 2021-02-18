import numpy as np
import os
import cv2
import copy
import matplotlib.image as mpimg
import uuid
from concurrent.futures import ProcessPoolExecutor,wait,as_completed
import time
import random


f = open("C:/Users/wuxb/Desktop/identity_CelebA.txt")

id2count = {}
while True:
    line = f.readline()
    if not line:
        break
    array = line.split()
    name = array[1]
    if not id2count.__contains__(name):
        id2count[name] = 0
    id2count[name] += 1

count = 0
for key, value in id2count.items():
    if value > 1:
        count += 1
        print(key, value)

print("count", count)
print("mancount", len(id2count))


