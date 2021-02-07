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
import uuid
from concurrent.futures import ProcessPoolExecutor,wait,as_completed
from numpy import save

# 九宫格除椒盐噪点
def convFilter(originImg):
    newImg = copy.deepcopy(originImg)
    for rowIndex in range(1, originImg.shape[0] - 1):
        for colIndex in range(1, originImg.shape[1] - 1):
            blackCount = 0
            if originImg[rowIndex - 1][colIndex - 1] == 0:
                blackCount += 1
            if originImg[rowIndex - 1][colIndex] == 0:
                blackCount += 1
            if originImg[rowIndex - 1][colIndex + 1] == 0:
                blackCount += 1
            if originImg[rowIndex][colIndex - 1] == 0:
                blackCount += 1
            if originImg[rowIndex][colIndex + 1] == 0:
                blackCount += 1
            if originImg[rowIndex + 1][colIndex - 1] == 0:
                blackCount += 1
            if originImg[rowIndex + 1][colIndex] == 0:
                blackCount += 1
            if originImg[rowIndex + 1][colIndex + 1] == 0:
                blackCount += 1
            whiteCount = 8 - blackCount
            # 当前是黑格子，周边的白格子大于等于5个则当前格子是噪点（白格子反之亦然）
            col = originImg[rowIndex][colIndex]
            if col == 0:
                if whiteCount >= 5:
                    newImg[rowIndex][colIndex] = 1
            else:
                if blackCount >= 5:
                    newImg[rowIndex][colIndex] = 0
    return newImg

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
    newImg2 = convFilter(newImg)
    newImg3 = convFilter(newImg2)
#     print(newImg)
#     plt.subplot(1, 2, 1)
#     plt.imshow(newImg, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.imshow(newImg2, cmap='gray')
#     plt.subplot(2, 2, 3)
#     plt.imshow(newImg3, cmap='gray')
#     plt.show()
#     exit()
    return newImg3


# 切割图片上的字符
def cutImageWords(originImage):
    # 第一步 用opencv的findContours来提取轮廓
    # 第一个参数是寻找轮廓的图像，第二个参数表示轮廓的检索模式，第三个参数method为轮廓的近似办法
    contours, hierarchys = cv2.findContours(originImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    del hierarchys
    if len(contours) <= 2:
        print("图片粘连2处以上，无法分割")
        return None
    if len(contours) > 4:
        print("图片分割多于4份，无法分割")
        return None
    # 根据x坐标横向排序
    x2position = {}
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x2position[x] = (x, y, w, h)
    x2positionSorted = sorted(x2position)
    
    maxIndex = -1
    if len(contours) == 3:
        maxWidth = 0
        for key in x2positionSorted:
            x, y, w, h = x2position[key]
            if w > maxWidth:
                maxWidth = w
                maxIndex = key
    draws = []
    for key in x2positionSorted:
        x, y, w, h = x2position[key]
        if key == maxIndex:
            draw1 = originImage[y:y+h, x:x+(int)(w/2)]
            draw2 = originImage[y:y+h, x+(int)(w/2):x+w]
            draws.append(draw1)
            draws.append(draw2)
        else:
            draw = originImage[y:y+h, x:x+w]
            draws.append(draw)
            
    # 边缘扩充，使得图片等大
    draws2 = []
    fixedHeight = 30
    fixedWidth = 24
    for draw in draws:
        height = draw.shape[0]
        width = draw.shape[1]
        if height > fixedHeight or width > fixedWidth:
            print("图片分割后尺寸过大，无法分割")
            return None
        top = int((fixedHeight - height)/2)
        bottom = fixedHeight - height - top
        left = int((fixedWidth - width)/2)
        right = fixedWidth - width - left
        draw = np.pad(draw, ((top,bottom),(left,right)))
        draws2.append(draw)
#     plt.imshow(draws[1], cmap='gray')
#     plt.show()
#     exit()
    return draws2


# 保存分割后的图片用于以后训练
def saveCuttedImage(draws, label):
    baseDir = "C:/Users/wuxb/Desktop/kaptcha_cnn/train_decompose/"
    for index, draw in enumerate(draws):
        word = label[index]
        fileDir = baseDir + word
        if os.path.exists(fileDir) == False:
            os.makedirs(fileDir)
        filePath = fileDir + "/" + str(uuid.uuid4()) + ".png"
        cv2.imwrite(filePath, draw * 255)


# 子进程中执行图片处理
def excuteImageFilter(imageDir, fileDir, labels, save):
    img = mpimg.imread(imageDir + "/" + fileDir)
    img = filterNoise(img)
    draws = cutImageWords(img)
    if draws == None:
        return
    index = int(fileDir[0: fileDir.rindex(".")])
    label = labels[index]
    words = list(label)
    if save:
        saveCuttedImage(draws, label)
    print("初始图处理：%s" % fileDir)
    return (draws, words)


# 获得图片集
def getImagesAndLabels(imageDir, labelPath, save=True):
    with open(labelPath, "r") as f:
        labels = json.load(f)
    
    decompose_images = []
    decompose_labels = []
    pathDir =  os.listdir(imageDir)
    executor = ProcessPoolExecutor(max_workers=7)
    
    childProcesses = []
    for fileDir in pathDir:
        childProcess = executor.submit(excuteImageFilter, imageDir, fileDir, labels, save)
        childProcesses.append(childProcess)
    for childProcess in as_completed(childProcesses):
        result = childProcess.result()
        if result == None:
            continue
        decompose_images.extend(result[0])
        decompose_labels.extend(result[1])
    decompose_images = np.array(decompose_images)
    decompose_labels = np.array(decompose_labels)
    return (decompose_images, decompose_labels)


def prepareTrainImage():
    getImagesAndLabels("C:/Users/wuxb/Desktop/kaptcha_cnn/train", "C:/Users/wuxb/Desktop/kaptcha_cnn/train_answer.json", save=True)


def loadPreparedIamge():
    keys = ["0","1","2","3","4","5","6","7","8",]
    baseDir = "C:/Users/wuxb/Desktop/kaptcha_cnn/train_decompose/"
    train_images = []
    train_labels = []
    for key in keys:
        pathDir = baseDir + key + "/"
        imagePathDir = os.listdir(pathDir)
        for imageName in imagePathDir:
            img = mpimg.imread(pathDir + imageName)
            train_images.append(img)
            train_labels.append(int(key))
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    return (train_images, train_labels)


def train():
    (train_images, train_labels) = loadPreparedIamge()
#     (test_images, test_labels) = getImagesAndLabels("C:/Users/wuxb/Desktop/kaptcha_cnn/test", "C:/Users/wuxb/Desktop/kaptcha_cnn/test_answer.json", save=False)
    train_images = train_images.reshape(-1, 30, 24, 1)
#     test_images = test_images.reshape(-1, 30, 24, 1)
    train_labels = keras.utils.to_categorical(train_labels, 10)   #将整形数组转化为二元类型矩阵
#     test_labels = keras.utils.to_categorical(test_labels, 10)
    
    print(train_images.shape)
    print(train_labels.shape)
#     print(test_images.shape)
#     print(test_labels.shape)
    
    # cnn
    model = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3,), activation=tf.nn.relu, input_shape=(30, 24, 1)),
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
    history = model.fit(train_images, train_labels, batch_size=60, epochs=1)
    
    # 存储
    # model.save_weights('data/cnn')
    keras.models.save_model(model, 'data/cnn_kaptcha')
    
    # 测试模型
    # verbose输出日志等级，0=不开启日志
#     test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
#     print('Test acc:', test_acc)


if __name__ == "__main__":  
#     prepareTrainImage()
    train()



