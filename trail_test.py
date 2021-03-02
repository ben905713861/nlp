import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import copy
import matplotlib.image as mpimg
import uuid
import time
import random
import pymysql
import matplotlib.pyplot as plt
import json


# 整理、归一化轨迹
def trailFilter(originTrail):
    trail = np.zeros([100, 3])
    for index, point in enumerate(originTrail):
        trail[index][0] = point[0] / 300.
        trail[index][1] = point[1] / 100.
        trail[index][2] = point[2] / 2000.
        if index >= 100:
            break
    return trail


def makeWrongTrail():
    # 创建1-20个随机点
    pointNum = random.randint(1, 40)
    # 随机斜率
    rate = random.random()
    timeStep = 1. / pointNum
    trail = np.zeros([100, 3])
    for pointIndex in range(1, pointNum + 1):
        time = pointIndex * timeStep
        x = time * rate
        trail[pointIndex - 1] = np.array([x, 0., time])
    return trail


conn = pymysql.connect(host='127.0.0.1', user='root', password='21316002', db='huodong')
cursor = conn.cursor()


trails = []
# 添加真实轨迹
cursor.execute('select * from verifyimg_trail')
while True:
    res = cursor.fetchone()
    if res is None:
        break
    trail = json.loads(res[2])
    trail = trailFilter(trail)
    trails.append(trail)
cursor.close()
conn.close()
# 添加机器轨迹
for i in range(23):
    trail = makeWrongTrail()
    trails.append(trail)
# 添加标签
train_trails = np.array(trails)
train_labels = np.ones((60, 2,))
for index, label in enumerate(train_labels):
    if index < 37:
        label[0] = 1
        label[1] = 0
    else:
        label[0] = 0
        label[1] = 1
print(train_trails.shape)
print(train_labels.shape)


# cnn
model = keras.Sequential([
    # 输入层
    layers.Flatten(input_shape=(100, 3,)),
    # units该层的神经元数; activation激活函数
    layers.Dense(units=256, activation=tf.nn.relu),
    # 输出层有2个
    layers.Dense(units=2, activation=tf.nn.softmax)
])

# 查看模型
model.summary()


# optimizer优化器=adam
# loss损失函数=categorical_crossentropy交叉熵损失函数
# metrics评估标准=sparse_categorical_accuracy稀疏分类准确率函数
model.compile(optimizer = keras.optimizers.Adadelta(1),
              loss = keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])

# 读取模型
# model.load_weights('data/cnn')
# model = keras.models.load_model('data/cnn')

# 训练10轮，每轮60张图
history = model.fit(train_trails, train_labels, batch_size=32, epochs=10, validation_data=(train_trails, train_labels))






