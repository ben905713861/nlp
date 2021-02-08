import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os
import copy
import cv2
import matplotlib.image as mpimg
import uuid
from concurrent.futures import ProcessPoolExecutor,wait,as_completed
import time
from _io import open
import random
from cnn_kaptcha import filterNoise
import matplotlib.pyplot as plt


# 保存处理后的图片用于以后训练
def saveCuttedImage(img, originIndex):
    baseDir = "C:/Users/wuxb/Desktop/kaptcha_cnn/train_nocut/"
    filePath = baseDir + str(originIndex) + ".png"
    cv2.imwrite(filePath, img * 255)


# 子进程中执行图片处理
def excuteImageFilter(imageDir, fileDir, labels, save):
    img = mpimg.imread(imageDir + "/" + fileDir)
    img = filterNoise(img)
    index = int(fileDir[0: fileDir.rindex(".")])
    label = labels[index]
    if save:
        saveCuttedImage(img, index)
    print("初始图处理：%s" % fileDir)
    return (img, label)


# 获得图片集
def getImagesAndLabels(imageDir, labelPath, save=True):
    with open(labelPath, "r") as f:
        originLabels = json.load(f)
    
    images = []
    labels = []
    pathDir =  os.listdir(imageDir)
    executor = ProcessPoolExecutor(max_workers=7)
    
    childProcesses = []
    for fileDir in pathDir:
        childProcess = executor.submit(excuteImageFilter, imageDir, fileDir, originLabels, save)
        childProcesses.append(childProcess)
        ################################### TODO
#         break
    for childProcess in as_completed(childProcesses):
        result = childProcess.result()
        images.extend(result[0])
        labels.append(result[1])
#         plt.imshow(images, cmap='gray')
#         plt.show()
        
    images = np.array(images)
    labels = np.array(labels)
    return (images, labels)


def prepareTrainImage():
    getImagesAndLabels("C:/Users/wuxb/Desktop/kaptcha_cnn/train", "C:/Users/wuxb/Desktop/kaptcha_cnn/train_answer.json", save=True)


def loadPreparedIamge():
    labelPath = "C:/Users/wuxb/Desktop/kaptcha_cnn/train_answer.json"
    with open(labelPath, "r") as f:
        origin_labels = json.load(f)
    
    baseDir = "C:/Users/wuxb/Desktop/kaptcha_cnn/train_nocut/"
    train_images = []
    train_labels = []
    pathDir =  os.listdir(baseDir)
    for fileDir in pathDir:
        img = mpimg.imread(baseDir + fileDir)
        train_images.append(img)
        index = int(fileDir[0: fileDir.rindex(".")])
        label = origin_labels[index]
        train_labels.append(label)
        ############################################
#         break
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    return (train_images, train_labels)


def train():
    (train_images, train_labels) = loadPreparedIamge()
    (test_images, test_labels) = getImagesAndLabels("C:/Users/wuxb/Desktop/kaptcha_cnn/test", "C:/Users/wuxb/Desktop/kaptcha_cnn/test_answer.json", save=False)
    train_images = train_images.reshape(-1, 160, 40, 1)
    test_images = test_images.reshape(-1, 160, 40, 1)
    
    def stringList2wordListList(inputList):
        outputList = []
        for item in inputList:
            outputList.append(list(item))
        return outputList
    
    train_labels = keras.utils.to_categorical(stringList2wordListList(train_labels), 10)   #将整形数组转化为二元类型矩阵
    test_labels = keras.utils.to_categorical(stringList2wordListList(test_labels), 10)
    
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)
    
    # cnn
    model = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3,), activation=tf.nn.relu, input_shape=(160, 40, 1)),
        layers.MaxPooling2D(pool_size=(2, 2,)),
        layers.Conv2D(filters=64, kernel_size=(3, 3,), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2,)),
        # 输入层
        layers.Flatten(),
        # units该层的神经元数; activation激活函数
        layers.Dense(units=128, activation=tf.nn.relu),
        layers.Dropout(0.25),
        # 输出层有4*10个。每层分别为0-9的数字，因为是多分类任务，我们选择softmax作为激活函数
        layers.Dense(units=4*10),
        # 将输出层分为4小层
        layers.Reshape([4,10]),
        layers.Softmax(),
    ])
    model.compile(optimizer = keras.optimizers.Adam(),
                  loss = keras.losses.categorical_crossentropy,
                  metrics = ['accuracy'])
    
    # 查看模型
    model.summary()
    
    # 训练10轮，每轮60张图
    model.fit(train_images, train_labels, batch_size=60, epochs=10, validation_data=(test_images, test_labels))
    
    # 存储
    keras.models.save_model(model, 'data/cnn_kaptcha_nocut')
    
    # 测试模型
    # verbose输出日志等级，0=不开启日志
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test acc:', test_acc)


def run(imgPath):
    model = keras.models.load_model('data/cnn_kaptcha_nocut')
    img = mpimg.imread(imgPath)
    img = filterNoise(img)
    test_images = img.reshape(-1, 160, 40, 1)
    answerOneHot = model(test_images)[0]
    answerWords = tf.argmax(answerOneHot, axis=1)
    answerWords = answerWords.numpy()
    answerWords = answerWords.astype(np.str)
    answerWords = ''.join(answerWords)
    return answerWords


def test():
    with open("C:/Users/wuxb/Desktop/kaptcha_cnn/test_answer.json", "r") as f:
        labels = json.load(f)
    baseDir = "C:/Users/wuxb/Desktop/kaptcha_cnn/test/"
    pathDir = os.listdir(baseDir)
    successCount = 0
    errorCount = 0
    for fileDir in pathDir:
        answer = run(baseDir + fileDir)
        index = int(fileDir[0: fileDir.rindex(".")])
        label = labels[index]
        if answer == label:
            successCount += 1
        else:
            errorCount += 1
            print(fileDir, answer)
        total = successCount + errorCount
        rate = successCount * 1. / total
        print("total: %d" % total)
        print("rate: %f" % rate)


if __name__ == "__main__":
#     prepareTrainImage()
#     train()
    test()



