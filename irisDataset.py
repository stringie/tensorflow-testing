#%%
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import urllib.request
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#function for the creation of bias variable, given matrix shape
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 4])
y_ = tf.placeholder(tf.int32, shape=[None])

target = tf.one_hot(y_, 3)

z = slim.fully_connected(x, 6)
z = slim.fully_connected(z, 10)
z = slim.fully_connected(z, 20)
z = slim.fully_connected(z, 16)
z = slim.fully_connected(z, 8)
y_out = slim.fully_connected(z, 3)

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y_out))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(target, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        train_step.run(feed_dict={x: training_set.data, y_: training_set.target})
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: training_set.data, y_: training_set.target})
            print('step %d -> accuracy: %g' % (i, train_accuracy))

    print(sess.run(accuracy, feed_dict={x:test_set.data, y_: test_set.target}))
