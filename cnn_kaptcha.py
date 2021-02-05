import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os
import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import dtype


# 图片去噪
def filterNoise(originImg):
    # 图片去除干扰线，二值化
    newImg = np.zeros([originImg.shape[0], originImg.shape[1],], dtype='uint8')
    for rowIndex, row in enumerate(originImg):
        for colIndex, col in enumerate(row):
            if col[0] == col[1] and col[1] == col[2]:
                newImg[rowIndex][colIndex] = 0
            else:
                # 变为灰色
                gray = np.mean(col)
                # 对近似白色区域一律改为纯黑
                if gray > 0.5:
                    gray = 0
                else:
                    gray = 1
                newImg[rowIndex][colIndex] = gray
    # 九宫格除椒盐噪点
    newImg2 = copy.deepcopy(newImg)
    for rowIndex in range(1, newImg.shape[0] - 1):
        for colIndex in range(1, newImg.shape[1] - 1):
            blackCount = 0
            if newImg[rowIndex - 1][colIndex - 1] == 0:
                blackCount += 1
            if newImg[rowIndex - 1][colIndex] == 0:
                blackCount += 1
            if newImg[rowIndex - 1][colIndex + 1] == 0:
                blackCount += 1
            if newImg[rowIndex][colIndex - 1] == 0:
                blackCount += 1
            if newImg[rowIndex][colIndex + 1] == 0:
                blackCount += 1
            if newImg[rowIndex + 1][colIndex - 1] == 0:
                blackCount += 1
            if newImg[rowIndex + 1][colIndex] == 0:
                blackCount += 1
            if newImg[rowIndex + 1][colIndex + 1] == 0:
                blackCount += 1
            whiteCount = 8 - blackCount
            # 当前是黑格子，周边的白格子大于等于5个则当前格子是噪点（白格子反之亦然）
            col = newImg[rowIndex][colIndex]
            if col == 0:
                if whiteCount >= 5:
                    newImg2[rowIndex][colIndex] = 1
            else:
                if blackCount >= 5:
                    newImg2[rowIndex][colIndex] = 0
#     plt.subplot(1, 2, 1)
#     plt.imshow(newImg, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.imshow(newImg2, cmap='gray')
#     plt.show()
#     exit()
    return newImg2


# 切割图片上的字符
def cutWords(originImage):
    # 第一步 用opencv的findContours来提取轮廓
    # 第一个参数是寻找轮廓的图像，第二个参数表示轮廓的检索模式，第三个参数method为轮廓的近似办法
    contours, hierarchys = cv2.findContours(originImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 第4个参数为线条颜色，第5个参数线条粗度
#     cv2.drawContours(originImage, contours, -1, (125,))
#     plt.imshow(originImage, cmap='gray')
#     plt.show()
#     exit()
    if len(contours) < 2:
        print("图片粘连2处以上，无法分割")
        return None
    if len(contours) == 3:
        maxWidth = 0
        maxIndex = -1
        for index, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > maxWidth:
                maxWidth = w
                maxIndex = index
        contour = contours[maxIndex]
    draws = []
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if index == maxIndex:
            draw1 = originImage[y:y+h, x:x+(int)(w/2)]
            draw2 = originImage[y:y+h, x+(int)(w/2):x+w]
            draws.append(draw1)
            draws.append(draw2)
        else:
            draw = originImage[y:y+h, x:x+w]
            draws.append(draw)
            
    plt.imshow(draws[2], cmap='gray')
    plt.show()
        
    exit()
    return originImage


# 获得图片集
def getImagesAndLabels(imageDir, labelPath):
    test_images = []
    pathDir =  os.listdir(imageDir)
    for fileDir in pathDir:
        img = mpimg.imread(imageDir + "/" + fileDir)
        img = filterNoise(img)
        img = cutWords(img)
        test_images.append(img)
        print("初始图处理：%s" % fileDir)
        break
    test_images = np.array(test_images)
    with open(labelPath, "r") as f:
        test_labels = json.load(f)
    new_test_labels = []
    for test_label in test_labels:
        test_label = np.array(list(test_label))
        test_label = keras.utils.to_categorical(test_label, 10) 
        new_test_labels.append(test_label)
    test_labels = np.array(new_test_labels)
    return (test_images, test_labels)


(train_images, train_labels) = getImagesAndLabels("C:/Users/wuxb/Desktop/kaptcha_cnn/test", "C:/Users/wuxb/Desktop/kaptcha_cnn/test_answer.json")
(test_images, test_labels) = getImagesAndLabels("C:/Users/wuxb/Desktop/kaptcha_cnn/test", "C:/Users/wuxb/Desktop/kaptcha_cnn/test_answer.json")
# for i in range(10):
#     plt.imshow(test_images[i], cmap='gray')
#     plt.show()
print(test_labels)


# cnn
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3,), activation=tf.nn.relu, input_shape=(160, 40, 1)),
    layers.MaxPooling2D(pool_size=(2, 2,)),
    layers.Conv2D(filters=64, kernel_size=(3, 3,), activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=(2, 2,)),
    # 输入层
    layers.Flatten(),
    # units该层的神经元数; activation激活函数
    layers.Dense(units=256, activation=tf.nn.relu),
    layers.Dropout(0.25),
    # 输出层有10个。分别为0-9的数字，因为是多分类任务，我们选择softmax作为激活函数
    layers.Dense(units=10, activation=tf.nn.softmax)
])
model.compile(optimizer = keras.optimizers.Adadelta(1),
              loss = keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])

# 查看模型
model.summary()

# 训练10轮，每轮60张图
history = model.fit(train_images, train_labels, batch_size=1, epochs=1, validation_data=(test_images, test_labels))

# 存储
# model.save_weights('data/cnn')
keras.models.save_model(model, 'data/cnn_kaptcha')


# 测试模型
# verbose输出日志等级，0=不开启日志
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test acc:', test_acc)


