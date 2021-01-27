import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
y = tf.nn.softmax([0., 1.])
# print(sess.run(y))
# exit()
y = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = [0., 1.], labels = [0., 1.]))
res = sess.run(y)

print(res)