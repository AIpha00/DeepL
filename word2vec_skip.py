# -*- coding: utf-8 -*-
"""
 author: lvsongke@oneniceapp.com
 data:2019/09/11
"""
import tensorflow as tf
import numpy as np
import math
import vocabulary

max_steps = 100000
batch_size = 128 ##训练样本的大小
embedding_size = 128  ##嵌入向量的尺寸
skip_distance = 1  ## 训练词的相邻的几个数
num_of_samples = 2  ##对每个单词生成多少样本

vocabulary_size = 50000

valid_examples = np.random.choice(100, 16, replace=False)

num_sampled = 64 ##训练时用来做负样本的噪声单词的数量  负采样的大小


with tf.Graph().as_default():
    ##train_inputs和train_labels是训练数据及其label的placeholder
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    ##embedding 是所有50000高频单词的词向量，向量的维度是128，数值是由random_uniform()
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    ##embedding_lookup函数用于选取一个张量里索引对应的元素
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    ##nce_loss 将多分类转换为二分类，softmax，交叉熵！
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))

    ###nce_biases偏执
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    ##计算词向量embed在训练数据上的loss
    nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size)

    loss = tf.reduce_mean(nce_loss)

    ##创建优化器
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

    normalized_embeddings = embeddings / norm

    valid_inputs = tf.constant(valid_examples, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_inputs)

    ##计算相似度
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


    ##训练开始
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        ##总损失和平均损失
        total_loss= 0
        average_loss = 0

        for step in range(max_steps + 1):
            ##生成训练数据
            batch_inputs, batch_labels = vocabulary.generate_batch(batch_size, num_of_samples, skip_distance)

            ##训练
            loss_val, _ = sess.run([loss, optimizer], feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})

            ##total_loss 用于计算总损失，在每一轮迭代后都会与loss_val相加
            total_loss += loss_val

            ##打印出1000次迭代的平均损失

            ##每进行1000轮迭代就输出平均损失的值，并将average_loss和total_loss重新归零
            if step > 0 and step % 1000 == 0:
                average_loss = total_loss / 1000
                print('Average loss at {} step is:{}'.format(step, average_loss))
                average_loss = 0
                total_loss = 0
            if step > 0 and step % 5000 == 0:
                ##执行计算相似性的操作
                similar = similarity.eval()

                ##外层循环16次
                for i in range(16):
                    ##没执行一次最外层的循环，都会得到一个验证单词对应的nearest
                    ##reverse_dictionary可以得到确切的单词
                    nearest = (-similar[i, :]).argsort()[1: 8 + 1]

                    #定义需要打印的字符串，期中valid_word是通过reverse_dictionary得到的验证单词
                    valid_word = vocabulary.reverse_dictionary[valid_examples[i]]

                    print_nearest_words = "Nearest to {}:".format(valid_word)

                    for j in range(8):
                        ##获取8个最近的词
                        close_word = vocabulary.reverse_dictionary[nearest[j]]
                        print_nearest_words = "{} {},".format(print_nearest_words, close_word)

                    print(print_nearest_words)
        final_embeddings = normalized_embeddings.eval()