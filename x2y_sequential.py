import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# MLP
model = keras.Sequential([
    # 输入层
    layers.Flatten(input_shape=(1, )),
    # 输出层
    layers.Dense(1, activation=tf.nn.softmax)
])

# 查看模型
model.summary()

# optimizer优化器=adam
# loss损失函数=sparse_categorical_crossentropy交叉熵损失函数
# metrics评估标准=sparse_categorical_accuracy稀疏分类准确率函数
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# 训练10轮，每轮64张图
steplen = 10 * 64
xGroup = np.random.rand(steplen, 1)
yGroup = xGroup * 0.2
print(xGroup)
print(yGroup)
model.fit(xGroup, yGroup, batch_size=64, epochs=10)

# verbose输出日志等级，0=不开启日志
xTestGroup = np.random.rand(steplen, 1)
yTestGroup = xGroup * 0.2
test_loss, test_acc = model.evaluate(xTestGroup, yTestGroup, verbose=2)
print('Test acc:', test_acc)


predictions = model.predict(xTestGroup)
print('x值:', xTestGroup[0])
print('预测值:', predictions[0])
print('真实值:', yTestGroup[0])

