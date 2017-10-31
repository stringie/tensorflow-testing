#%%
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import urllib.request
import numpy as np

#LOAD DATA 
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

#data and labels
x = tf.placeholder(tf.float32, [None, 4])
y_ = tf.placeholder(tf.int32, shape=[None])

#probability for dropout function
keep_prob = tf.placeholder(tf.float32)

#one hot encoded input and 3 fully connected hidden layers 
#with dropout for stabiliy
target = tf.one_hot(y_, 3)
z = slim.fully_connected(target, 8)
z = slim.dropout(z, keep_prob)
z = slim.fully_connected(z, 16)
z = slim.dropout(z, keep_prob)
z = slim.fully_connected(z, 7)
y_out = slim.fully_connected(z, 3)

#loss function
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y_out))

#fitting
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#result assesment
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        train_step.run(feed_dict={x: training_set.data, y_: training_set.target, keep_prob: 0.5})
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: training_set.data, y_: training_set.target, keep_prob: 1.0})
            print('step %d -> accuracy: %g' % (i, train_accuracy))

    print(sess.run(accuracy, feed_dict={x:test_set.data, y_: test_set.target, keep_prob: 1.0}))