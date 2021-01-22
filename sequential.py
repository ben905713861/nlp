import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import numpy as np


class Network:

    def __init__(self):
        # 学习速率，一般在 0.00001 - 0.5 之间
        self.learning_rate = 0.001

        # 输入张量 28 * 28 = 784个像素的图片一维向量
        # 参数2 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 784]表示列是784，行不定）
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, [None, 784])

        # 标签值，即图像对应的结果，如果对应数字是8，则对应label是 [0,0,0,0,0,0,0,0,1,0]
        # 这种方式称为 one-hot编码
        # 标签是一个长度为10的一维向量，值最大的下标即图片上写的数字
        self.label = tf.compat.v1.placeholder(tf.float32, [None, 10])

        # 权重，初始化全 0
        self.w = tf.Variable(tf.zeros([784, 10]))
        # 偏置 bias， 初始化全 0
        self.b = tf.Variable(tf.zeros([10]))
        # 输出 y = softmax(X * w + b)
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)

        # 损失，即交叉熵，最常用的计算标签(label)与输出(y)之间差别的方法
        self.loss = -tf.reduce_sum(self.label * tf.compat.v1.log(self.y + 1e-10))

        # 反向传播，采用梯度下降的方法(GradientDescentOptimizer)。调整w与b，使得损失(loss)最小
        # loss越小，那么计算出来的y值与 标签(label)值越接近，准确率越高
        self.train = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # 以下代码验证正确率时使用
        # argmax 返回最大值的下标，最大值的下标即答案
        # 例如 [0,0,0,0.9,0,0.1,0,0,0,0] 代表数字3
        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))

        # predict -> [true, true, true, false, false, true]
        # reduce_mean即求predict的平均数 即 正确个数 / 总数，即正确率
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))


class Train:
    def __init__(self):
        self.net = Network()

        # 初始化 session
        # Network() 只是构造了一张计算图，计算需要放到会话(session)中
        self.sess = tf.compat.v1.Session()
        # 初始化变量
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # 读取训练和测试数据，这是tensorflow库自带的，不存在训练集会自动下载
        # 项目目录下已经下载好，删掉后，重新运行代码会自动下载
        # data_set/train-images-idx3-ubyte.gz
        # data_set/train-labels-idx1-ubyte.gz
        # data_set/t10k-images-idx3-ubyte.gz
        # data_set/t10k-labels-idx1-ubyte.gz
#         self.data = input_data.read_data_sets('../data_set', one_hot=True)

    def train(self):
        # batch_size 是指每次迭代训练，传入训练的图片张数。
        # 数据集小，可以使用全数据集，数据大的情况下，
        # 为了提高训练速度，用随机抽取的n张图片来训练，效果与全数据集相近
        # https://www.zhihu.com/question/32673260
        batch_size = 64

        # 总的训练次数
        train_step = 2000

        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        
        # 开始训练
        for i in range(train_step):
            # 从数据集中获取 输入和标签(也就是答案) 独热编码
            img1 = train_images[i]
            x = []
            for row in img1:
                for col in row:
                    if col == 0:
                        x.append(0)
                    else:
                        x.append(1)
            x = np.array([x])
            
            label = np.zeros([1, 10])
            label[0, train_labels[i] - 1] = 1
            
            # 每次计算train，更新整个网络
            # loss只是为了看到损失的大小，方便打印
            _, loss = self.sess.run([self.net.train, self.net.loss],
                                    feed_dict={self.net.x: x, self.net.label: label})

            # 打印 loss，训练过程中将会看到，loss有变小的趋势
            # 代表随着训练的进行，网络识别图像的能力提高
            # 但是由于网络规模较小，后期没有明显下降，而是有明显波动
            if (i + 1) % 10 == 0:
                print('第%5d步，当前loss：%.2f' % (i + 1, loss))

    def calculate_accuracy(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        imgs = []
        label_answers = []
        for i in range(200):
            # 测试用的图片 二维转1维
            img = test_images[i].ravel()
            # one-hot编码
            img[img > 0] = 1
            imgs.append(img)
            
            # 答案 one-hot编码
            label_answer = test_labels[i]
            label_answer = np.eye(10)[label_answer]
            label_answers.append(label_answer)
            
        imgs = np.array(imgs)
        labels = np.array(label_answers)
        
        # 注意：与训练不同的是，并没有计算 self.net.train
        # 只计算了accuracy这个张量，所以不会更新网络
        # 最终准确率约为0.91
        accuracy = self.sess.run(self.net.accuracy, feed_dict={self.net.x: imgs, self.net.label: labels})
        print("准确率: %.2f " % accuracy)


if __name__ == "__main__":
    app = Train()
    app.train()
    app.calculate_accuracy()
    
    