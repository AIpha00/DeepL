# -*- coding: utf-8 -*-
"""
 author: lvsongke@oneniceapp.com
 data:2019/09/11
"""
import numpy as np

##定义相关的参数
x = [0.8, 0.1]

##隐藏层的参数
init_state = [0.3, 0.6]
W_h = np.asarray([[0.2, 0.4], [0.7, 0.3]])
W_x = np.asarray([[0.8, 0.1]])
b_h = np.asarray([[0.2, 0.1]])
W_y = np.asarray([[0.5], [0.5]])
b_y = 0.1

# 执行两轮循环，模拟前向传播过程
for i in range(len(x)):
    # 计算h，隐藏层的参数
    before_activation = np.dot(init_state, W_h) + x[i] * W_x + b_h

    ##计算激活值
    state = np.tanh(before_activation)

    ##将此时的状态作为下一时刻的初始状态
    init_state = state

    ##计算输出值
    y_out = np.dot(state, W_y) + b_y

    ##打印隐藏值和输出值
    print('隐藏值：', state)
    print('输出值:', y_out)
