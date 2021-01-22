from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 输出结果
# img1 = train_images[1]
# for row in img1:
#     for col in row:
#         if col == 0:
#             print(0, end="")
#         else:
#             print(1, end="")
#     print("")
# print(train_labels[1])
# exit()

train_images, test_images = train_images / 255.0, test_images / 255.0


# MLP
model = keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# model = keras.Sequential()
# # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# # 第2层卷积，卷积核大小为3*3，64个
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# # 第3层卷积，卷积核大小为3*3，64个
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.fit(train_images, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test acc:', test_acc)


predictions = model.predict(test_images)
print('预测值:', np.argmax(predictions[0]))
print('真实值:', test_labels[0])
