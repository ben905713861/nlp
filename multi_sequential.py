import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from time import time

tf.compat.v1.disable_eager_execution()
# 获取训练/测试用的手写图片与答案
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

#全连接层函数

def fcn_layer(
    inputs,    #输入数据
    input_dim, #输入层神经元数量
    output_dim,#输出层神经元数量
    activation =None): #激活函数
    
    W = tf.Variable(tf.random.truncated_normal([input_dim,output_dim],stddev = 0.1))
        #以截断正态分布的随机初始化W
    b = tf.Variable(tf.zeros([output_dim]))
        #以0初始化b
    XWb = tf.matmul(inputs,W)+b # Y=WX+B
    
    if(activation==None): #默认不使用激活函数
        outputs =XWb
    else:
        outputs = activation(XWb) #代入参数选择的激活函数
    return outputs #返回
#各层神经元数量设置
H1_NN = 256
H2_NN = 64
H3_NN = 32

#构建输入层
x = tf.compat.v1.placeholder(tf.float32,[None,784],name='X')
y = tf.compat.v1.placeholder(tf.float32,[None,10],name='Y')
#构建隐藏层
h1 = fcn_layer(x,784,H1_NN,tf.nn.relu)
h2 = fcn_layer(h1,H1_NN,H2_NN,tf.nn.relu)
h3 = fcn_layer(h2,H2_NN,H3_NN,tf.nn.relu)
#构建输出层
forward = fcn_layer(h3,H3_NN,10,None)
pred = tf.nn.softmax(forward)#输出层分类应用使用softmax当作激活函数
#损失函数使用交叉熵
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = forward,labels = y))
#设置训练参数
train_epochs = 50
batch_size = 50
total_batch = int(len(train_images)/batch_size) #随机抽取样本
learning_rate = 0.01
display_step = 1
#优化器
opimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_function)
#定义准确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#保存模型
save_step = 5 #储存模型力度
import os
ckpt_dir = '.ckpt_dir/'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
#开始训练
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver() #声明完所有变量以后，调用tf.train.Saver开始记录
startTime = time()
sess.run(init)
for epochs in range(train_epochs):
    for batch in range(total_batch):
        imgs = []
        label_answers = []
        # 每次随机64张图
        for j in range(batch_size):
            # 每次从图集中随机出一张
            randomIndex = np.random.randint(60000)
            # 训练用的图片 二维转1维
            img = train_images[randomIndex].ravel()
            imgs.append(img)
            
            # 答案 one-hot编码
            label_answer = train_labels[randomIndex]
            label_answer = np.eye(10)[label_answer]
            label_answers.append(label_answer)
        
        xs, ys = imgs, label_answers
#         xs,ys = mnist.train.next_batch(batch_size)#读取批次数据
        sess.run(opimizer,feed_dict={x:xs,y:ys})#执行批次数据训练
    
    #total_batch个批次训练完成后，使用验证数据计算误差与准确率
    loss,acc =  sess.run([loss_function,accuracy],
                        feed_dict={
                            x:test_images,
                            y:test_labels})
    #输出训练情况
    if(epochs+1) % display_step == 0:
        epochs += 1 
        print("Train Epoch:",epochs,
               "Loss=",loss,"Accuracy=",acc)
    if(epochs+1) % save_step == 0:
        saver.save(sess, os.path.join(ckpt_dir,"mnist_h256_model_{:06d}.ckpt".format(epochs+1)))
        print("mnist_h256_model_{:06d}.ckpt saved".format(epochs+1))
duration = time()-startTime
print("Trian Finshed takes:","{:.2f}".format(duration))#显示预测耗时
#评估模型
accu_test =  sess.run(accuracy,feed_dict={x:test_images,y:test_labels})
print("model accuracy:",accu_test)
#恢复模型,创建会话

saver = tf.compat.v1.train.Saver()

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(ckpt_dir)#选择模型保存路径
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess ,ckpt.model_checkpoint_path)#从已保存模型中读取参数
    print("Restore model from"+ckpt.model_checkpoint_path)
