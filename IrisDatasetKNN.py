#%%
import tensorflow as tf
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

#%%
x_train = tf.placeholder(tf.float32, [None, 4])
x_test = tf.placeholder(tf.float32, [None, 4])
y_train = tf.placeholder(tf.int32, [None])
y_test = tf.placeholder(tf.int64, [None])

y_target = tf.one_hot(y_train, 3)

#lower number of neighbors produces slightly worse results
k = 11

#L1 (taxicab metric)
#negative -> tf.nn.top_k gets the highest values
#reduce_sum -> sums up the error in the distance in this case L1
#abs -> no need to explain
#subtract -> uses broadcasting to subtract each of the features for all 30 flowers from each feature of all 120 flowers
#expand_dims -> expands dim of test by 1 so as to use broadcasting, since the two tensors ([120, 4], [30, 4]) are incompatible
distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(x_train, tf.expand_dims(x_test, 1))), axis=2))

#retrieves the top k values and their indeces
top_k_values, top_k_indeces = tf.nn.top_k(distance, k)
#creates a tensor from the correct labels with the predicted indeces
top_k_label = tf.gather(y_target, top_k_indeces)
#gets the majority class of the neighbors for each test flower
majority_of_neighbors = tf.reduce_sum(top_k_label, 1)
#gets the predicted label for each test flower
label_of_majority = tf.argmax(majority_of_neighbors, axis=1)

#assesment
correct_prediction = tf.equal(label_of_majority, y_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(distance, feed_dict={x_train: training_set.data, x_test: test_set.data}))
    print(sess.run(accuracy, feed_dict={x_train: training_set.data, x_test: test_set.data, y_train: training_set.target, y_test: test_set.target}))