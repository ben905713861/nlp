import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


# 创建图
x = tf.compat.v1.placeholder(tf.float32)
W = tf.compat.v1.Variable([1.], dtype=tf.float32)
b = tf.compat.v1.Variable([0.], dtype=tf.float32)
# 预测值 y = W * x + b
y = W * x + b
# 真实值
y_ = tf.compat.v1.placeholder(tf.float32)
# 方差 cost = [(y1 - _y1)^2 + ... + (yn - _yn)^2] /n
cost = tf.reduce_mean(tf.pow(y_ - y, 2))
# 梯度下降模型
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.0001).minimize(cost)


# 建立session
sess = tf.compat.v1.Session()
# 初始化所有计算值
sess.run(tf.compat.v1.global_variables_initializer())

# 训练1000次
for i in range(1000):
    # 创建steplen个随机点，大致符合y = 2 * x  + (+/-)0.1
    steplen = 2
    xGroup = np.random.randint(0, 100, steplen)
    yGroup = xGroup * 2 + (np.random.rand(steplen)-0.5)/10
    feed = {
        x: xGroup,
        y_: yGroup,
    }
    print(xGroup)
    print(yGroup)
    # 训练
    (_train_step, _W, _b, _cost) = sess.run([train_step, W, b, cost], feed_dict=feed)
    
    print("w: %f" % _W)
    print("b: %f" % _b)
    print("cost: %f" % _cost)

