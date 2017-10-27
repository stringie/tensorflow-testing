#%%
import tensorflow as tf

#What we will be using here is labeled data of hand-drawn numbers from 0-10
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#placeholder for the images
x = tf.placeholder(tf.float32, [None, 784])
#Weights
W = tf.Variable(tf.zeros([784, 10]))
#biases
b = tf.Variable(tf.zeros([10]))

#Output and Correct output
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

#The loss function for the algorithm (We are using Multinomial Logistic Classification,
# hence cross entropy from Information Theory)
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#The training optimizer, in this case: gradient descent for minimizing the loss function
#via going in the direction of steepest descent, although with current settings, 
#FtrlOptimizer produces slightly better results at learning_rate=0.4
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Session and variable initialization
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Retrieving our performance
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#Showing percentage of correct evaluations
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Print accuracy
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100, "%"