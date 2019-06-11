# !/usr/bin/env python
# -*- coding:UTF-8 -*-

__author__ = 'Lizhenghao'
__version__ = '0.0.1'

import random
import math
import time
import os


# 机器学习基类,所有的具体机器学习算法都要集成自此类
class LZHMachineLearningBase(object):
    def __init__(self, data: tuple):
        self.data = data
        self.test = None
        self.train = None

    # 划分数据集
    def split_random(self, p):
        try:
            train = []
            test = []
            for i in self.data:
                if random.random() < p:
                    train.append(i)
                else:
                    test.append(i)
            self.train = train
            self.test = test
            return True
        except:
            return False


# 贝叶斯分类器
# Naive Bayes: P(A|B) = P(B|A) * (P(A)/P(B))
class Bayes(LZHMachineLearningBase):
    def __init__(self, data: tuple, p=0.7):
        LZHMachineLearningBase.__init__(self, data)
        self.split_random(p)

    # 获取data的所有属性值,该方法为不适合的方法,将会在后续的版本中进行移除
    def _get_attribute(self, data):
        temp = []
        for i in data:
            temp.append(i.x)
        return list(zip(*temp))

    # 即,获取P(Ci)的值
    def get_bayes_label(self, data):
        label = []
        for i in data:
            label.append(i.y)
        label = list(set(label))
        counter = [0] * len(label)
        for i in range(len(label)):
            for j in data:
                if j.y == label[i]:
                    counter[i] = counter[i] + 1
        total_length = len(data)
        ans = dict.fromkeys(label, 0)
        for i in range(len(label)):
            ans[label[i]] = float(counter[i]) / total_length
        return ans

    # 通过一组向量预测其所属标号,获取其软分类矩阵
    def forecast(self, v, data):
        label = self.get_bayes_label(data)
        ans_dic = dict.fromkeys(sorted(label.keys()), 0)
        temp_lst = []
        for i in label.keys():
            temp_p = 1
            for j in range(v.length):
                label_counter = 0
                temp_counter = 0
                for k in data:
                    if k.y == i:
                        label_counter = label_counter + 1
                        if k.x[j] == v[j]:
                            temp_counter = temp_counter + 1
                temp_p = temp_p * (float(temp_counter) / label_counter)
            temp_lst.append(temp_p)
            ans_dic[i] = temp_p
        for i in label.keys():
            ans_dic[i] = ans_dic[i] * label[i]
        return ans_dic

    # 通过一个向量预测其最有可能属于的类
    def test_label(self, v, data):
        ans = self.forecast(v, data)
        return max(ans, key=ans.get)

    def check(self, show=False):
        num_of_right = 0
        for i in self.test:
            t = self.test_label(i.x, self.train)
            if show:
                print(i.x.data, end='\t')
                print(t, end=' \t')
                temp = self.forecast(i.x, self.train)
                # temp[-1] = temp[-1] * 2
                # temp[1] = temp[1] * 2
                print(temp, end='\t\n')
            if t == i.y:
                num_of_right = num_of_right + 1
        return float(num_of_right) / len(self.test)



if __name__ == '__main__':
    print("This is machine learning package written by Lizhenghao")
