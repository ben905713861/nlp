import numpy as np
import os
import cv2
import copy
import matplotlib.image as mpimg
import uuid
from concurrent.futures import ProcessPoolExecutor,wait,as_completed
import time
import random

a = np.arange(2000)
a = a.tolist()
random.shuffle(a)
print(a[0])