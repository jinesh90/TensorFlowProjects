"""
To train, and test, the implemented models, we will be using one of the most famous datasets called MNIST of handwritten
digits. The MNIST dataset is a training set of 60,000 examples and a test set of 10,000 examples. An example of the
data, as it is stored in the files of the examples, is shown in the preceding figure.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops


# Load data for MNIST into "data" folder, create one if not present.
dataPath = "data/"
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

mnist = input_data.read_data_sets(dataPath, one_hot=True)

# check data dimension
# print(mnist.train.images.shape) # (55000, 784)
# print(mnist.train.labels.shape) # (55000, 10)
# print(mnist.test.images.shape)  # (10000, 784)
# print(mnist.test.labels.shape)  # (10000, 10)

# for i in range(10):
#     image_0 = mnist.train.images[i]
#     image_0 = np.resize(image_0, (28, 28))
#     label_0 = input.train.labels[i]
#     print(label_0) # it will display [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
#
#     plt.imshow(image_0, cmap='Greys_r')
#     plt.show()

# set global variables
logs_path = 'log_sigmoid/' # logging path
batch_size = 10 # batch size while performing training
learning_rate = 0.005 # Learning rate
training_epochs = 10 # training epoch
display_epoch = 1


# Use Softmax classifier

L = 200 # number of neurons in layer 1
M = 100 # number of neurons in layer 2
N = 60 # number of neurons in layer 3
O = 30 # number of neurons in layer 4


# input , since every image is 28 * 28 pixel so each image is presented by 784 object array.
X = tf.placeholder(tf.float32, [None, 784], name='InputData')  # image shape 28*28=784

#XX = tf.reshape(X, [-1, 784]) # reshape input

# identify 0-9 labels so output dimension is 10.
Y_ = tf.placeholder(tf.float32, [None, 10], name='LabelData') # 0-9 digits => 10 classes

# layer 1
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1)) # Initialize random weights for the hidden layer 1
B1 = tf.Variable(tf.zeros([L])) # Bias vector for layer 1

# feed from layer one
Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)

# layer 2
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1)) # Initialize random weights for the hidden layer 2
B2 = tf.Variable(tf.ones([M])) # Bias vector for layer 2

# feed from layer two
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)

# layer 3
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1)) # Initialize random weights for the hidden layer 3
B3 = tf.Variable(tf.ones([N])) # Bias vector for layer 3

Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3) # Output from layer 3

# layer 4
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1)) # Initialize random weights for the hidden layer 4
B4 = tf.Variable(tf.ones([O])) # Bias vector for layer 4

Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4) # Output from layer 4

# last layer
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1)) # Initialize random weights for the hidden layer 5
B5 = tf.Variable(tf.ones([10])) # Bias vector for layer 5


Ylogits = tf.matmul(Y4, W5) + B5 # computing the logits
Y = tf.nn.softmax(Ylogits)# output from layer 5

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_) # final outcome using softmax cross entropy
cost_op = tf.reduce_mean(cross_entropy)*100

# Optimization op (backprop)
#train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

train_op  = tf.train.GradientDescentOptimizer(0.005).minimize(cost_op)
# Initialize the variables (i.e. assign their default value)
init_op = tf.global_variables_initializer()

# Construct model and encapsulating all ops into scopes, making Tensorboard's Graph visualization more convenient



# Create a summary to monitor cost tensor
tf.summary.scalar("cost", cost_op)

# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)

# Merge all summaries into a single op
summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    # Run the initializer
    sess.run(init_op)

    avg_cost = 0.

    # op to write logs to Tensorboard
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples / batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c, summary = sess.run([train_op, cost_op, summary_op], feed_dict={X: batch_x, Y_: batch_y})
            writer.add_summary(summary, epoch * batch_count + i)

            # Compute average loss
            avg_cost += c / batch_count

            # Display logs per epoch step
            if (epoch + 1) % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Epoch: ", epoch)
        print("Optimization Finished!")

        print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
    #train_op.run(feed_dict={image_0 = mnist.train.images[i])