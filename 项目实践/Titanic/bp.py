# !/usr/bin/env python
# -*- coding:UTF-8 -*-

__author__ = 'Lizhenghao'
__version__ = '0.0.1'

import math
import time
import random
import numpy as np
from matplotlib import pyplot as plt

"""
实验名称 – 基于BP神经网络的分类算法实现
目的：
通过实现神经网络的BP算法，理解BP算法的基本原理和基本的程序实现方法。
带类标的数据：titanic.dat（详见相应的文件），该文件为文本文件，以@开头的行为注释行，剩下每一行为一个带类标的训练样例，由空格隔开的属性值构成，最后一个属性值为类标，取值为1.0或者-1.0，分别表示两个不同的类别。
要求：
（1）	构造一个含有一个隐藏层（隐藏层单元的个数为4，用Sigmoid单元），输出层单元数为1（也是用Sigmoid单元）的二层神经网络，用于对titanic数据进行分类；
（2）	对给定的数据集随机划分，70%作为训练数据，30%作为测试数据；
（3）	分别使用梯度下降和随机梯度下降进行训练；
（4）	分别统计梯度下降和随机梯度下降算法在测试集上类别预测的准确率（预测类别正确的测试样例的个数/测试样例的个数）。
程序设计语言
   C、C++、Java或Python

"""


# 神经网络类
class NeuralNetwork(object):
    def __init__(self, structure: list, func: list, d_func: list):
        # 判断传入参数的合法性
        if (not len(structure) == len(func)) or (not len(func) == len(d_func)):
            raise AttributeError("参数structure, func与d_func的长度必须相等")
        # 随机初始化
        # 初始化权重与偏置
        self.structure = structure
        self.layers = len(structure)
        self.weight = self.get_random_weight_with_structure()
        self.bias = self.get_random_bias_with_structure()
        self.active_func = func
        self.d_active_func = d_func
        pass

    # 根据神经网络结构获取随机权重
    def get_random_weight_with_structure(self):
        ans = [np.array([0])]
        for i in range(self.layers - 1):
            # 生成[-1, +1]的随机矩阵
            temp_array = np.random.random((self.structure[i+1], self.structure[i])) * 2 - 1
            # temp_array = np.full((self.structure[i+1], self.structure[i]), 0.1)
            ans.append(temp_array)
        return np.array(ans)

    # 根据神经网络结果获取随机偏置值
    def get_random_bias_with_structure(self):
        ans = [np.array([0])]
        for i in self.structure[1:]:
            temp_array = np.random.random((i, 1)) * 2 - 1
            # temp_array = np.full((i, 1), 0.1)
            ans.append(temp_array)
        return np.array(ans)

    # 前向传播
    def feedforward(self, a: list):
        """Return the output of the network for an input vector a"""
        # for b, w in zip(self.biases, self.weights):
        #     a = self.sigmoid(np.dot(w, a) + b)
        # return a
        a_array = []
        z_array = []
        a = np.array(a).reshape((len(a), 1))
        a_array.append(a)
        z_array.append(a)
        for i in range(1, self.layers):
            w = self.weight[i]
            b = self.bias[i]
            z = np.dot(w, a) + b
            z_array.append(z)
            a = self.active_func[i](z)
            a_array.append(a)
        return (np.array(a_array), np.array(z_array))

    # 反向传播
    def feedbackward(self, x, a_array, z_array, aim_output, study_rate=0.1):
        target = a_array.copy()
        for i in range(len(target)):
            target[i] = np.zeros(shape=target[i].shape)
        target[-1] = np.array(aim_output)
        # 用于存储dc/da(l)的矩阵
        dc_da_array = a_array.copy()
        for i in range(len(target)):
            dc_da_array[i] = np.zeros(shape=dc_da_array[i].shape)
        #
        diff_weight = self.weight.copy()
        # 权重偏差
        for i in range(len(diff_weight)):
            diff_weight[i] = np.zeros(shape=diff_weight[i].shape)
        diff_bias = self.bias.copy()
        # 偏置误差
        for i in range(len(diff_bias)):
            diff_bias[i] = np.zeros(shape=diff_bias[i].shape)
        # 损失
        ans_c = 0
        # 逐层便利神经网络
        for layer in range(self.layers - 1, 0, -1):
            # print(layer)
            # print("正在反向传播第 %d 层" % layer)
            for i in range(len(a_array[layer])):
                # print("第 %d 层的第 %d 个神经元" % (layer, i))
                for j in range(len(a_array[layer - 1])):
                    # print("*********上一层的第 %d 个神经元" % j)
                    # 代价函数
                    # c = (target[layer][i] - a_array[layer][i]) ** 2
                    dz_dw = a_array[layer - 1][j]
                    dz_db = 1
                    dz_dal_1 = self.weight[layer][i][j]
                    da_dz = self.d_active_func[layer](z_array[layer][i])
                    if layer == self.layers - 1:
                        dc_da = 2 * (a_array[layer][i] - target[layer][i])
                        dc_da_array[layer][i] = dc_da
                        cost = (target[layer][i] - a_array[layer][i]) ** 2
                        ans_c += cost
                    else:
                        dc_da = dc_da_array[layer][i]
                    dw = dz_dw * da_dz * dc_da
                    db = dz_db * da_dz * dc_da
                    da = dz_dal_1 * da_dz * dc_da
                    # 更新目标输出矩阵
                    target[layer-1][j] = target[layer-1][j] + da[0]
                    # print(dc_da_array[layer][i], dc_da_array[layer-1][j], dc_da[0])
                    dc_da_array[layer-1][j] = dc_da_array[layer-1][j] + dc_da[0]
                    diff_weight[layer][i][j] = dw
                    diff_bias[layer][i] = db
                pass
            pass
        diff_bias = diff_bias * study_rate
        diff_weight = diff_weight * study_rate
        # 更新权重与偏置
        self.weight = self.weight - diff_weight
        self.bias = self.bias - diff_bias
        return (diff_bias, diff_weight, ans_c)


if __name__ == '__main__':
    print("Hello World!")
    sigmoid = lambda x: 1/(1 + math.e**(-x))
    d_sigmoid = lambda x: sigmoid(x)*(1 - sigmoid(x))

    ann = NeuralNetwork([4, 3, 3, 1], [None, sigmoid, sigmoid, sigmoid], [None, d_sigmoid, d_sigmoid, d_sigmoid])

    x = []
    y = []
    plt.title("Matplotlib demo")
    plt.xlabel("times")
    plt.ylabel("cost")

    for i in range(1000):
        if i % 4:
            ans = ann.feedforward([1, 2, 3, 4])
            temp = ann.feedbackward([1, 2, 3, 4], ans[0], ans[1], [[1]], 0.1)
        else:
            ans = ann.feedforward([4, 3, 2, 1])
            temp = ann.feedbackward([4, 3, 2, 1], ans[0], ans[1], [[0]], 0.1)
        cost = temp[-1]
        x.append(i)
        y.append(cost)
        print(ans[0][-1], i % 4)
    plt.plot(x, y, "ob")
    plt.show()
    pass
