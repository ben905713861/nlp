import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.compat.v1.disable_eager_execution()

# 建立图
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
# 784 * 10 逻辑回归神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
label = tf.compat.v1.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(label * tf.compat.v1.log(y + 1e-10))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss)


# 建立session
sess = tf.compat.v1.Session()
# 初始化所有计算值
sess.run(tf.compat.v1.global_variables_initializer())

# 获取训练/测试用的手写图片与答案
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 训练2000次
for i in range(2000):
    train_images_size = len(train_images)
    imgs = []
    label_answers = []
    # 每次随机64张图
    for j in range(64):
        # 每次从图集中随机出一张
        randomIndex = np.random.randint(train_images_size)
        # 训练用的图片 二维转1维
        img = train_images[randomIndex].ravel()
        # one-hot编码
        img[img > 10] = 1
        imgs.append(img)
        
        # 答案 one-hot编码
        label_answer = train_labels[randomIndex]
        label_answer = np.eye(10)[label_answer]
        label_answers.append(label_answer)
    
    feed = {
        x: np.array(imgs),
        label: np.array(label_answers),
    }
    
    # 训练
    (_train_step, _loss) = sess.run([train_step, loss], feed_dict=feed)
    
    if (i + 1) % 10 == 0:
        print('第%5d步，当前loss：%.2f' % (i + 1, _loss))

print("===============")

# 测试结果
errorCount = 0
for i in range(10000):
    # 测试用的图片 二维转1维
    img = test_images[i].ravel()
    # one-hot编码
    img[img > 10] = 1
    img = np.array([img])
    
    feed = {
        x: img,
    }
    
    _y = sess.run(y, feed_dict=feed)
    test_result = np.argmax(_y, 1)[0]
    
    if test_result != test_labels[i]:
        errorCount += 1
        test_image = test_images[i]
        test_image[test_image > 0] = 1
#         print(test_image)
#         print(test_result)
#         print(test_labels[i])

print("errorCount : %d" % errorCount)
