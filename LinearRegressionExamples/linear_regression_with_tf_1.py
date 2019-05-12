import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

# Create 1000 points following a function y = 0.1 * x + 0.4z  where W = 0.1 and Bias B= 0.4
num_points = 1000
vectors_set = []
for i in range(num_points):
    W = 0.1 # W
    b = 0.4 # b
    x1 = np.random.normal(0.0, 1.0)#in: mean, standard deviation
    nd = np.random.normal(0.0, 0.05)#in:mean,standard deviation
    y1 = W * x1 + b
    # Add some impurity with normal distribution -i.e. nd
    y1 = y1 + nd
    # Append them and create a combined vector set:
    vectors_set.append([x1, y1])

# Separate the data point across axises:
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# Plot and show the data points in a 2D space
plot.plot(x_data, y_data, 'bo', label='Original data')
plot.legend()
plot.show()

with tf.name_scope("LinearRegression") as scope:
    W = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b
with tf.name_scope("LossFunction") as scope:
    # loss function
    loss = tf.reduce_mean(tf.square(y - y_data))
    loss_summary = tf.summary.scalar("loss", loss)
    w_ = tf.summary.histogram("W", W)
    b_ = tf.summary.histogram("b", b)

merged_op = tf.summary.merge_all()
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
writer_tensorboard = tf.summary.FileWriter('logs/', tf.get_default_graph())
sess = tf.Session()
sess.run(init)
for i in range(16):
   sess.run(train)
   print(i, sess.run(W), sess.run(b), sess.run(loss))
   plot.plot(x_data, y_data, 'ro', label='Original data')
   plot.plot(x_data, sess.run(W)*x_data + sess.run(b))
   plot.xlabel('X')
   plot.xlim(-2, 2)
   plot.ylim(0.1, 0.6)
   plot.ylabel('Y')
   plot.legend()
   plot.show()

