import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_labels = keras.utils.to_categorical(train_labels, 10)   #将整形数组转化为二元类型矩阵
test_labels = keras.utils.to_categorical(test_labels, 10)


# cnn
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    # 输入层
    layers.Flatten(),
    # units该层的神经元数; activation激活函数
    layers.Dense(units=256, activation=tf.nn.relu),
    layers.Dropout(0.25),
    # 输出层有10个，分别为0-9的数字，因为是多分类任务，我们选择softmax作为激活函数
    layers.Dense(units=10, activation=tf.nn.softmax)
])

# 查看模型
model.summary()


# optimizer优化器=adam
# loss损失函数=categorical_crossentropy交叉熵损失函数
# metrics评估标准=sparse_categorical_accuracy稀疏分类准确率函数
model.compile(optimizer = keras.optimizers.Adadelta(1),
              loss = keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])

# 训练10轮，每轮60张图
model.fit(train_images, train_labels, batch_size=60, epochs=10, validation_data=(test_images, test_labels))

# 测试模型
# verbose输出日志等级，0=不开启日志
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test acc:', test_acc)

