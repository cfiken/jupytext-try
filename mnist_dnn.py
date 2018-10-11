# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
#   varInspector:
#     cols:
#       lenName: 16
#       lenType: 16
#       lenVar: 40
#     kernels_config:
#       python:
#         delete_cmd_postfix: ''
#         delete_cmd_prefix: 'del '
#         library: var_list.py
#         varRefreshCmd: print(var_dic_list())
#       r:
#         delete_cmd_postfix: ') '
#         delete_cmd_prefix: rm(
#         library: var_list.r
#         varRefreshCmd: 'cat(var_dic_list()) '
#     types_to_exclude:
#     - module
#     - function
#     - builtin_function_or_method
#     - instance
#     - _Feature
#     window_display: false
# ---

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.978905Z", "end_time": "2018-10-11T11:58:26.092908Z"}}
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.980Z", "end_time": "2018-10-11T11:58:26.097137Z"}}
mnist = input_data.read_data_sets("data/", one_hot=True)
test_images = mnist.test.images
test_labels = mnist.test.labels

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.981Z", "end_time": "2018-10-11T11:58:26.098223Z"}}
# constant

num_inputs = 784  # 28*28
num_units = 256
num_outputs = 10
num_layers = 1
batch_size = 64
learning_rate = 0.001
num_epocs = 5000
step_to_print = 10

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.983Z", "end_time": "2018-10-11T11:58:26.099142Z"}}
myfloat = tf.float32


# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.983Z", "end_time": "2018-10-11T11:58:26.099806Z"}}
x = tf.placeholder(dtype=myfloat, shape=[None, num_inputs])
y = tf.placeholder(dtype=myfloat, shape=[None, num_outputs])

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.985Z", "end_time": "2018-10-11T11:58:26.100574Z"}}
# 一層
w_1 = tf.Variable(tf.truncated_normal([num_inputs, num_units], stddev=0.1), dtype=myfloat, name='w_1')
b_1 = tf.Variable(tf.zeros([num_units]), dtype=myfloat, name='b_1')
h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.986Z", "end_time": "2018-10-11T11:58:26.101283Z"}}
w_2 = tf.Variable(tf.truncated_normal([num_units, num_units], stddev=0.1), dtype=myfloat, name='w_2')
b_2 = tf.Variable(tf.zeros([num_units]), dtype=myfloat, name='b_2')
h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)
w_3 = tf.Variable(tf.truncated_normal([num_units, num_outputs], stddev=0.1), dtype=myfloat, name='w_3')
b_3 = tf.Variable(tf.zeros([num_outputs]), dtype=myfloat, name='b_3')
out = tf.nn.softmax(tf.matmul(h_2, w_3) + b_3)

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.987Z", "end_time": "2018-10-11T11:58:26.102886Z"}}
train_loss = tf.reduce_sum(tf.square(y - out))
optimizer = tf.train.AdamOptimizer(learning_rate)
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)

training = optimizer.apply_gradients(zip(gradients, params))

grad_norm = [tf.norm(g) for g in gradients]
grad_norm_sum = tf.reduce_sum(grad_norm)

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.988Z", "end_time": "2018-10-11T11:58:26.103519Z"}}
correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, myfloat))

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.989Z", "end_time": "2018-10-11T11:58:26.104338Z"}}
init = tf.global_variables_initializer()
with tf.name_scope('summary'):
    summary_loss = tf.summary.scalar('loss', train_loss)
    summary_grad = tf.summary.scalar('gradients', grad_norm_sum)
    summary_acc = tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.990Z", "end_time": "2018-10-11T11:58:26.105500Z"}}
from datetime import datetime
now = datetime.now()
logdir = 'logs/mnist_dnn/'

# + {"ExecuteTime": {"start_time": "2018-10-11T11:58:25.992Z", "end_time": "2018-10-11T11:58:26.106818Z"}}
logdir = logdir + now.strftime("%Y%m%d-%H%M%S") + "/"
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(init)
    for e in range(num_epocs):
        xdata, ydata = mnist.train.next_batch(batch_size)
        l, _, smr = sess.run([train_loss, training, merged_summary], feed_dict={
            x: xdata,
            y: ydata
        })
        if e % step_to_print == 0:
            acc = sess.run(accuracy, feed_dict={
                x: test_images,
                y: test_labels
            })
            print('step :', e, ', loss :', l, ', accuracy: ', acc)
        writer.add_summary(smr, e)
    writer.close()            
    
# -


