from keras import losses
import numpy as np

y_pred = np.asarray([1, 1], dtype=np.float32)
y_true = np.asarray([1, 3], dtype=np.float32)

result = losses.mean_squared_error(y_true, y_pred)
print(result)

import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(result))
