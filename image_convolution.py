import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import ndarray

originImg = mpimg.imread('2.jpg')
originImg = originImg/255.0
originImgHeight = originImg.shape[0]
originImgWidth = originImg.shape[1]
newImg = np.zeros(originImg.shape)
# 卷积核
# kernel = np.array([
#     [1./16, 2./16, 1./16,],
#     [2./16, 4./16, 2./16,],
#     [1./16, 2./16, 1./16,],
# ])
kernel = np.array([
    [0., 1., 0.,],
    [1., -4., 1.,],
    [0., 1., 0.,],
])

# 卷积核直径与半径
kernelDiameter = kernel.shape[0]
kernelRadius = (int)((kernel.shape[0] - 1) / 2)

# 边缘插值
originImg = np.append(originImg, np.zeros([kernelRadius, originImgWidth, 3]), axis = 0)
originImg = np.append(originImg, np.zeros([originImgHeight + kernelRadius, kernelRadius, 3]), axis = 1)


def gray():
    for rowIndex in range(originImgHeight):
        for colIndex in range(originImgWidth):
            cell = np.zeros([kernelDiameter, kernelDiameter])
            cellConvolution = 0.
            
            for kernel1 in range(kernelDiameter):
                for kernel2 in range(kernelDiameter):
                    cell[kernel1][kernel2] = originImg[rowIndex + kernel1 - kernelRadius][colIndex + kernel2 - kernelRadius]
            
            for kernel1 in range(kernelDiameter):
                for kernel2 in range(kernelDiameter):
                    cellConvolution += cell[kernel1][kernel2] * kernel[kernel1][kernel2]
            
            newImg[rowIndex][colIndex] = cellConvolution
            
    plt.subplot(1, 2, 1)
    plt.imshow(originImg, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.imshow(newImg, cmap='gray')
    
    plt.show()
    

def color():
    for rowIndex in range(originImgHeight):
        for colIndex in range(originImgWidth):
            for rgbIndex in range(3):
                cell = np.zeros([kernelDiameter, kernelDiameter])
                cellConvolution = 0.
                
                for kernel1 in range(kernelDiameter):
                    for kernel2 in range(kernelDiameter):
                        cell[kernel1][kernel2] = originImg[rowIndex + kernel1 - kernelRadius][colIndex + kernel2 - kernelRadius][rgbIndex]
                
                for kernel1 in range(kernelDiameter):
                    for kernel2 in range(kernelDiameter):
                        cellConvolution += cell[kernel1][kernel2] * kernel[kernel1][kernel2]
                        
                newImg[rowIndex][colIndex][rgbIndex] = cellConvolution
            
    plt.subplot(1, 2, 1)
    plt.imshow(originImg)
    
    plt.subplot(1, 2, 2)
    plt.imshow(newImg)
    
    plt.show()


color()


