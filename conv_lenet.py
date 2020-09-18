# -*- coding: utf-8 -*-
"""
 author: lvsongke@oneniceapp.com
 data:2019/09/11
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist', one_hot=True)

batch_size = 128
LR = 0.01
lr_decay = 0.99  ##学习衰减率
max_steps = 1001


def hidden_layer(input_tensor, regularizer, avg_class, resuse):
    ##创建第一个卷积层，得到特征图大小为32 28*28
    with tf.variable_scope('C1-conv', reuse=resuse):
        conv1_weights = tf.get_variable('weight', [5, 5, 1, 32],
                                        ##将权重初始化为截断的正态分布，标准差为0.1
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        ##get_variable  设置参数
        conv1_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
        ##卷积运算
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        ##relu激活函数
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    ##创建第一个池化层， 池化后结果为32 14x14
    with tf.name_scope("S2-max_pool"):
        ##最大池化层
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ##创建第二个卷积层，得到特征图大小为 64 14x14。注意第一个池化层之后得到了32个特征图，所以这里设输入的深度为32，我们在这一层选择的卷积核数量为64，所以输出的深度是64，也就是说有64个特征图
    with tf.variable_scope('C3-conv', reuse=resuse):
        ##设置权重参数、并将权重参数初始化为截断的正态分布，标准差为0.1
        conv2_weights = tf.get_variable('weight', [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        ##初始化偏置，为0
        conv2_biases = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))

        ##卷积运算
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        ##relu激活
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    ##创建第二个池化层, 结果64 7x7
    with tf.name_scope('S4-max_pool'):
        ##池化层
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ##使用get_shape函数可以得到这一层的维度信息。由于每一层网络的输入输出都是一个batch矩阵，所以通过get_shape()函数得到的维度信息会包含这个batch中。
        ##数据的个数信息。shape[1]是长度方向，reshape函数原型为reshape(tensor,shape,name)
        shape = pool2.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        reshaped = tf.reshape(pool2, [shape[0], nodes])

    ##  创建第一个全连接
    with tf.variable_scope("layer5-full1", reuse=resuse):
        Full_connection1_weights = tf.get_variable("weight", [nodes, 512],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        ##对全连接的权重加入正则化
        tf.add_to_collection('losses', regularizer(Full_connection1_weights))
        Full_connection1_biases = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.1))

        if avg_class == None:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, Full_connection1_weights) + Full_connection1_biases)
        else:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, avg_class.average(Full_connection1_weights)) + avg_class.average(
                Full_connection1_biases))

    ##创建第二个全连接
    with tf.variable_scope('layer6-full2', reuse=resuse):
        Full_connection1_weights = tf.get_variable('weight', [512, 10],
                                                   initializer=tf.truncated_normal_initializer(0.1))
        # 对全连接的权重加入正则化
        tf.add_to_collection('losses', regularizer(Full_connection1_weights))
        Full_connection1_biases = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))

        if avg_class == None:
            result = tf.matmul(Full_1, Full_connection1_weights) + Full_connection1_biases
        else:
            result = tf.matmul(Full_1, avg_class.average(Full_connection1_weights)) + avg_class.average(
                Full_connection1_biases)

        return result


###placeholder 用于对数据进行保存----只是一个占位符，在运行session时才被初始化
x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
regularizer = tf.contrib.layers.l2_regularizer(0.0001)

##注意reuse参数取之为False

y = hidden_layer(x, regularizer, avg_class=None, resuse=False)

##Variable创建一个变量
trainning_step = tf.Variable(0, trainable=False)

##加权移动平均
variable_averages = tf.train.ExponentialMovingAverage(0.99, trainning_step)

##tf.trainable_variables()获取计算图中未将trainable标记未false的参数
variable_averages_op = variable_averages.apply(tf.trainable_variables())

##reuse参数取值为true
average_y = hidden_layer(x, regularizer, variable_averages, resuse=True)

##定义损失函数为交叉熵
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)

##计算损失值
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

##定义自适应学习率
learning_rate = tf.train.exponential_decay(LR, trainning_step, mnist.train.num_examples / batch_size, lr_decay,
                                           staircase=True)

##定义优化器-梯度下降
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=trainning_step)

##control_dependencies 控制计算流图的先后顺序。先执行完train_step后在执行variable_averages_op
with tf.control_dependencies([train_step, variable_averages_op]):
    train_op = tf.no_op(name='train')

##tf.equal 判断两个计算图是否相等
crorent_predicition = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))

##tf.cast()转换格式
accuracy = tf.reduce_mean(tf.cast(crorent_predicition, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_steps):
        if i % 1000 == 0:
            x_val, y_val = mnist.validation.next_batch(batch_size)

            reshaped_x2 = np.reshape(x_val, (batch_size, 28, 28, 1))

            validate_feed = {x: reshaped_x2, y_: y_val}

            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)

            print('after {} trainning step (s), validation accuracy using average model is {:.2f}'.format(i,
                                                                                                          validate_accuracy * 100))

        x_train, y_train = mnist.train.next_batch(batch_size)
        ##用于训练的数据需要进行reshape处理
        reshaped_xs = np.reshape(x_train, (batch_size, 28, 28, 1))
        sess.run(train_op, feed_dict={x: reshaped_xs, y_: y_train})

    saver = tf.train.Saver()

    saver.save(sess, 'mnist_model')
