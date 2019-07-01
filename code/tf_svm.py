# Linear Kernel:
# K(x1, x2) = t(x1) * x2
#
# Gaussian Kernel (RBF):
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
from generate_dateSet import generate_Dataset

ops.reset_default_graph()

learning_rate = 3e-4
max_iterators = 20000
batch_size = 200
x = tf.placeholder(tf.float32, [None, 1364])
y = tf.placeholder(tf.float32, [None, 2])
dataset_size = 0
length = 0
input_layer = 0

d = generate_Dataset("data/firefox.txt")
ben_data, neg_data = d.getData()
length = d.getMaxGadget()

l1 = len(ben_data)
l2 = len(neg_data)
x = ben_data + neg_data
# x = self.Normalization(x)  # 归一化
ben_data = x[:l1]
neg_data = x[l1:]

# 转二维
ben_data = np.array(ben_data).reshape([int(len(ben_data) / (length - 10)), length - 10])
neg_data = np.array(neg_data).reshape([int(len(neg_data) / (length - 10)), length - 10])
ben_data = ben_data[:-1]
neg_data = neg_data[:-1]
dataset_size = len(ben_data) + len(neg_data)

x = []
for i in range(len(ben_data)):
    x.append(ben_data[i])
for i in range(len(neg_data)):
    x.append(neg_data[i])

y = []
for i in range(len(ben_data)):
    y.append([1.0])
for i in range(len(neg_data)):
    y.append([0.0])

index = [i for i in range(dataset_size)]  # 打乱数据集
# np.random.seed(10000)
np.random.shuffle(index)
index = np.array(index)
images = np.array(x)[index]
labels = np.array(y)[index]
print("len labels :", len(labels), "index :", labels)

images = np.array(images)
labels = np.array(labels).reshape([len(labels), 1])
print("len x:", len(images[:1]))
print("x:", images[1:2])
print("x shape:", images.shape)
print("y shape:", labels.shape)
print("length of dataset: %d" % (len(images)))
print("length of dataset: %d" % (len(labels)))
input_layer = images.shape[1]
print("input_shape:", input_layer)
# self.x ,self.y = x , y

x = images
y = labels
# Generate non-lnear data
train_x = x[:int(4 * len(x) / 5)]
train_y = y[:int(4 * len(y) / 5)]
# test_x = x[int(4*len(x)/5):]#int(6*len(x)/7)
test_x = x[:]
# test_y = y[int(4 * len(y) / 5):]#int(6*len(y)/7)
test_y = y[:]

# Declare batch size
batch_size = 350

# Initialize placeholders
x_data = tf.placeholder(shape=[None, input_layer], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, input_layer], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

# Apply kernel
# Linear Kernel
# my_kernel = tf.matmul(x_data, tf.transpose(x_data))

# Gaussian (RBF) kernel

#my_kernel=tf.matmul（x_data，tf.transpose（x_data）。
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Compute SVM Model
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
loss = tf.negative(tf.subtract(first_term, second_term))

# Create Prediction Kernel
# Linear prediction kernel
# my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                      tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.002)
train_step = my_opt.minimize(loss)

loss_vec = []
batch_accuracy = []
# Initialize variables
init = tf.global_variables_initializer()
# Create graph


x = x_data
y = y_target
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    batch_size = batch_size
    # dataset_size = self.dataset_size
    for step in range(1, max_iterators + 1):
        start = (step * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        batch_x, batch_y = train_x[start:end], train_y[start:end]
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        if step <= 10 or step % 1000 == 0 or step == max_iterators:
            acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, prediction_grid: test_x})
            print('Step %s, Accuracy %s' % (step, acc))
    fp = sess.run(false_positive, feed_dict={x: test_x, y: test_y})
    fn = sess.run(false_negative, feed_dict={x: test_x, y: test_y})
    rc = sess.run(recall, feed_dict={x: test_x, y: test_y})
    pc = sess.run(precision, feed_dict={x: test_x, y: test_y})
    # au = sess.run(auc, feed_dict={self.x: test_x, self.y: test_y})
    ac = sess.run(accu, feed_dict={x: test_x, y: test_y})
    print("false_positives : %d / %d percentage = %f " % (fp, len(test_y), fp / len(test_y) * 100))
    print("false_negatives:%d / %d percentage= %f " % (fn, len(test_y), fn / len(test_y) * 100))
    print("recall :", rc)
    print("precision :", pc)
    # print("auc :", au)
    print("accuracy :", ac)