#%%
import tensorflow as tf

#What we will be using here is labeled data of hand-drawn numbers from 0-10
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#function for the creation of weight variables, given matrix shape
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#function for the creation of bias variable, given matrix shape
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#creates a 2d convolutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#creates a 2x2 pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#placeholders for the input and labeled correct output
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#first layer variables
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

#execution of first hidden convolutional/pooling layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second layer variables
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#execution of second hidden convolutional/pooling layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#variables for first fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#fully connected neural network layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout execution for the neural network (dropout of neurons leads to the network 
# teaching itself stability and efficiency)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#final fully connected layer variables for the output
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#output
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#The loss function for the algorithm (We are using Multinomial Logistic Classification,
# hence cross entropy from Information Theory)
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#The training optimizer, minimizing the loss function
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Retrieving our performance i.e. how well we did relative to the correct results
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

#Showing percentage of correct evaluations
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Session and variable initialization
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))