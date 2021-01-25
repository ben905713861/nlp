import numpy as np
import tensorflow as tf

# 2 *2 *3 -> 2*6
data = np.array([
    [
        [1,2,3],
        [3,4,5]
    ],
    [
        [10,20,30],
        [30,40,50]
    ],
])
res = data.reshape(2, -1)
print(res)
res2 = tf.cast(res/255.0, tf.float32)
print(res2)
