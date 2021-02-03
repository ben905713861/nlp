import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import ndarray

originImg = mpimg.imread('2.jpg')
originImg = originImg / 255.0
originImg = np.array([originImg])


# 卷积核
# kernel = np.array([
#     [1./16, 2./16, 1./16,],
#     [2./16, 4./16, 2./16,],
#     [1./16, 2./16, 1./16,],
# ])
# kernel = np.array([
#     [[[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],], [[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],], [[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],],],
#     [[[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],], [[9.,9.,9.], [9.,9.,9.], [9.,9.,9.],], [[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],],],
#     [[[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],], [[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],], [[-1.,-1.,-1.], [-1.,-1.,-1.], [-1.,-1.,-1.],],],
# ])
# kernel = tf.constant(kernel, shape=(3,3, 3,3))

kernel = tf.Variable(tf.constant([
                                    [-1., -1., -1.], [-1., -1., -1.], [-1., -1., -1.],
                                    [-1., -1., -1.], [9.0, 9.0, 9.0], [-1., -1., -1.],
                                    [-1., -1., -1.], [-1., -1., -1.], [-1., -1., -1.],
                                  ], shape=[3, 3, 3, 1]))
# kernel = tf.Variable(tf.constant([
#                                     [0., 0., 0.,], [0., 0., 0.,], [0., 0., 0.,],
#                                     [0., 0., 0.,], [1., 1., 1.,], [0., 0., 0.,],
#                                     [0., 0., 0.,], [0., 0., 0.,], [0., 0., 0.,],
#                                   ], shape=[3, 3, 3, 1]))

edge = tf.nn.conv2d(originImg, kernel, strides=[1, 1, 1, 1], padding='SAME')  # 3个通道输入，生成1个feature map
# print()
# print(op)
# exit()


# edge = tf.nn.relu(edge) #与o比较，opp用relu函数，未做归一化处理
# edge = tf.cast(((edge - tf.reduce_min(edge)) / (tf.reduce_max(edge) - tf.reduce_min(edge))) * 255, tf.uint8)

# edge = tf.nn.conv2d(originImg, kernel, [1,1,1,1], "SAME")

print(originImg[0][0])
print(edge[0][0])

plt.subplot(1, 2, 1)
plt.imshow(originImg[0])
plt.subplot(1, 2, 2)
plt.imshow(edge[0], cmap='gray')
plt.show()

