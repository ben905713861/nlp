import numpy as np
import os
import cv2
import copy
import matplotlib.image as mpimg
import uuid
from concurrent.futures import ProcessPoolExecutor,wait,as_completed
import time

def test(counter, name):
    counter += 1
    print(counter, name)
    time.sleep(len(name))
    return counter

if __name__ == "__main__":
    executor = ProcessPoolExecutor(max_workers=4)
    
    counter = 0
    threads = []
    thread = executor.submit(test, counter, "ben")
    threads.append(thread)
    thread = executor.submit(test, counter, "leedd")
    threads.append(thread)
#     wait(threads)
    for thread in as_completed(threads):
        print(thread.result())
    
#     results = executor.map(test, [100,200], "ben")
#     executor.shutdown(wait=True)
#     for result in results:
#         print(result)
    
    print("end")
