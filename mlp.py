import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 60000张图片（每张图片是28*28的矩阵），转换为60000个1*784矩阵
train_images = train_images.reshape(60000, -1)
# 10000张图片（每张图片是28*28的矩阵），转换为10000个1*784矩阵
test_images = test_images.reshape(10000, -1)


# MLP
model = keras.Sequential([
    # 输入层
    layers.Flatten(input_shape=(784, )),
    # units该层的神经元数; activation激活函数
    layers.Dense(units=256, activation=tf.nn.relu),
    # 输出层有10个，分别为0-9的数字，因为是多分类任务，我们选择softmax作为激活函数
    layers.Dense(units=10, activation=tf.nn.softmax)
])

# 查看模型
model.summary()

# optimizer优化器=adam
# loss损失函数=sparse_categorical_crossentropy交叉熵损失函数
# metrics评估标准=sparse_categorical_accuracy稀疏分类准确率函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


# 读取模型
model.load_weights('data/mlp')

# 训练10轮，每轮60张图
model.fit(train_images, train_labels, batch_size=60, epochs=10)

# 存储
model.save_weights('data/mlp')

# verbose输出日志等级，0=不开启日志
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test acc:', test_acc)


errorCount = 0
predictions = model.predict(test_images)
for i in range(len(predictions)):
    if np.argmax(predictions[i]) != test_labels[i]:
        errorCount += 1
print((len(test_images) - errorCount) / len(test_images))
