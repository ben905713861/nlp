import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.compat.v1.disable_eager_execution()
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=320)

# 建立图
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
x1 = x

# 784 * 256 逻辑回归神经网络
W1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.5))
# W1 = tf.Variable(tf.compat.v1.random_normal([784, 256], stddev=0.5), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([256]))
# b1 = tf.Variable(0, dtype=tf.float32)
y1 = tf.nn.relu(tf.matmul(x1, W1) + b1)

# 256 * 10 逻辑回归神经网络
W2 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.5))
# W2 = tf.Variable(tf.compat.v1.random_normal([256, 10], stddev=0.5), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([10]))
# b2 = tf.Variable(0, dtype=tf.float32)
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

y = y2
label = tf.compat.v1.placeholder(tf.float32, [None, 10])
# loss = -tf.reduce_sum(label * tf.compat.v1.log(y + 1e-10))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = label))
# train_step = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss)
train_step = tf.compat.v1.train.AdamOptimizer().minimize(loss)
# train_step = tf.compat.v1.train.AdagradOptimizer(0.3).minimize(loss)

train_result = tf.equal(y, label)
train_succ_rate = tf.reduce_mean(tf.cast(train_result, "float"))

# 建立session
sess = tf.compat.v1.Session()
# 初始化所有计算值
sess.run(tf.compat.v1.global_variables_initializer())

# 获取训练/测试用的手写图片与答案
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 测试结果
def check():
    succCount = 0
    for i in range(10000):
        # 测试用的图片 二维转1维
        img = test_images[i].ravel()
        img = np.array([img])
        feed = {
            x: img,
        }
        _y = sess.run(y, feed_dict=feed)
        test_result = np.argmax(_y, 1)[0]
        if test_result == test_labels[i]:
            succCount += 1
    print("succ : %.4f" % (succCount/10000.0))

# 训练10轮
for h in range(20):
    for i in range(1000):
        imgs = []
        label_answers = []
        # 每轮60张图
        for j in range(60):
            # 每次从图集中拿出一张
            randomIndex = i * 60 + j
            # 训练用的图片 二维转1维
            img = train_images[randomIndex].ravel()
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
        (_train_step, _loss, _x, _y, _b2) = sess.run([train_step, loss, x, y, b2], feed_dict=feed)
#         print(_b2)
#         print(_x)
#         print(_y)
#         print(_loss)
#         exit()
    print('第%5d轮，当前loss：%.6f' % (h + 1, _loss))
    check()
    print("===============")
check()

