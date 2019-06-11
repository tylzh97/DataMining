# !/usr/bin/env python
# -*- coding:UTF-8 -*-

__author__ = 'Lizhenghao'
__version__ = '0.0.1'


import random
import math
import time
import os


"""实验要求
实验名称 – 基于KNN的分类算法实现
目的：
通过实现KNN算法，理解KNN算法的基本原理和基本的程序实现方法。
带类标的数据：titanic.dat（详见相应的文件），该文件为文本文件，以@开头的行为注释行，剩下每一行为一个带类标的训练样例，由空格隔开的属性值构成，最后一个属性值为类标，取值为1.0或者-1.0，分别表示两个不同的类别。
要求：
（1）	把各个属性的值进行规范化，要求采用Range方法（即把每个属性的值映射到[0, 1]，不包括类标属性）；
（2）	对给定的数据集随机划分，70%作为训练数据，30%作为测试数据；
（3）	实现KNN算法，给出测试数据集中每个测试样例的预测类标（用欧氏距离计算实例之间的距离，K参数可调）
（4）	分别统计K=1、3、5、7、9情况下算法在测试集上类别预测的准确率（预测类别正确的测试样例的个数/测试样例的个数）。
程序设计语言
   C、C++、Java或Python
"""

# 属性-类标结构体
class Point(object):
    """
    为一个高维点,封装了距离计算等方法
    args:
        x:  输入属性值,要求为tuple类型
        y:  属性所对应的类标
    """
    def __init__(self, x: tuple, y):
        self.x = tuple(x)
        self.y = y
    
    @staticmethod
    def get_distance(x1, x2, method="E"):
        """
        静态方法,获取两点的距离,可选属性,默认为欧几里得距离
        args:
            x1: 第一个Point
            x2: 第二个Point
            method: 使用的距离算法,可缺省,默认为欧几里得距离,要求为字符串类型
                    目前支持的距离算法:
                    "E": 欧几里得距离
                    "M": 曼哈顿距离
        """ 
        # 判断两个参数长度是否相同
        if not len(x1) == len(x2):
            raise AttributeError("Point x Length error!")

        # 闭包函数,欧几里得距离算子
        def Euclid(x1: tuple, x2: tuple):
            ans = 0
            for i in range(len(x1)):
                ans = ans + (x1[i]-x2[i])**2
            ans = math.sqrt(ans)
            return ans
        
        # 闭包函数,曼哈顿算子
        def Manhattan(x1: tuple, x2: tuple):
            ans = 0
            for i in range(len(x1)):
                ans = ans + abs(x1[i]-x2[i])
            return ans
        
        da = {
            'E': Euclid,
            'M': Manhattan,
        }
        func = None
        try:
            func = da[method]
        except:
            raise AttributeError("Wrong distance method!")
        return func(x1, x2)


# 泰坦尼克数据集
class Titanic(object):
    """
    用于初始化以及处理Titanic数据集
    args:
        data_path:  data文件的存放路径,支持相对路径
        separtor:   data文件中使用的分隔符,可缺省,默认为','
        commenting: data文件中的代码注释符,可缺省,默认为'@'
        goal_colum: data文件中,待遇测属性所在的行数,遵循python风格,默认为最后一行"-1"
        data_range: data文件中,需要映射的数据范围
    return:
        None
    """
    def __init__(self, data_path, separator=',', commenting='@', goal_colum=-1, data_range=(0, 1)):
        self.goal_colum = goal_colum
        self.range = data_range
        # 判断规则文件是否存在,不存在则抛出文件不存在异常
        if not os.path.isfile(data_path):
            raise FileExistsError("Screen file(%s) does't exists!" % self.sfpath)
        # 打开数据文件
        data_file = open(data_path, 'r')
        # 将所有数据读取到内存中,该方法非常消耗内存,在数据集过大时可能会奔溃
        datas = data_file.readlines()
        # 关闭数据文件
        data_file.close()
        # 声明数据变量
        data = []
        # 分离注释与数据
        for i in datas:
            if not i[0]==commenting:
                # 此时数据为字符串类型,在后续需要使用时需要转换为整形
                # 在此处转换代码不够优雅,将会在后续的初始化方案中完成转换
                data.append(i.replace('\n', '').split(separator))
        # 将数据设置为私有变量
        self.data = data
        self._floatlize_data()
        self._range_all()
        self._pointlize_data()
    
    # 将self.data数据float化
    def _floatlize_data(self):
        data = self.data
        ans = []
        for i in data:
            temp = []
            for j in i:
                temp.append(float(j))
            ans.append(temp)
        self.data = ans
        return self.data
    
    # 获取各行数据
    # 实际操作为取矩阵转置
    def _get_columns(self):
        return list(zip(*self.data))
    
    # 为数据的Range方法,强制数据规范到指定区间内
    def _range_data(self, data, data_range=(0, 1)):
        """
        args:
            data:       单行数据,为需要规范的单行数据,要求数据可索引、迭代(array/list/tuple)
            data_range: 数据映射范围,为一个长度为2的tuple,左侧为下限右侧为上限
        return:
            返回规范化后的单行数据,Numpy的array类型
        """
        _max = float(max(data))
        _min = float(min(data))
        # 线性映射范围
        # 两点式方程(_min, data_range[0]), (_max, data_range[1])
        y = lambda x: (data_range[1]-data_range[0])*(x-_min)/(_max-_min)+data_range[0]
        ans = []
        for i in data:
            ans.append(y(float(i)))
        return ans
    
    # Range所有列的数据
    def _range_all(self, data_range=(0, 1)):
        colums = self._get_columns()
        goal = colums[self.goal_colum]
        new_data = []
        for i in colums:
            # 判断当前是否为目标行
            temp = None
            if not i == goal:
                temp = self._range_data(i)
            new_data.append(temp) if temp else new_data.append(goal)
        self.data = new_data
        self.data = self._get_columns()
        return self.data
    
    # 将所有的数据转换为自定义的point类型
    def _pointlize_data(self):
        out = []
        for i in self.data:
            # 剥离属性值
            x = i[:-1] if self.goal_colum==-1 else i[:self.goal_colum]+i[self.goal_colum+1:]
            y = i[self.goal_colum]
            p = Point(x, y)
            out.append(p)
        self.data = out
        return self.data
    
    


# KNN最近邻算法
class KNN(object):
    def __init__(self, data, k, precentage_of_training_data=0.7, method='E'):
        self.data = data
        self.k = k
        self.method = method
        self._split(precentage_of_training_data)
    
    # 划分数据集,采用随机划分
    def _split(self, p=0.7):
        """
        功能:
            划分数据集
        args:
            precentage_of_training_data: 训练集百分比比,可缺省,默认为0.7.该参数必须为[0,1]的小数,否则会报错
        return:
            (训练集, 测试集):   返回一个元祖,其中[0]为训练集, [1]为测试集
        """
        if p<0 or p>1:
            raise AttributeError("Parameter 'precentage_of_training_data' must be between 0 and 1")
        # 训练集
        train = []
        # 测试集
        test = []
        # 对数据集采取随机划分
        for i in self.data:
            if random.random() < p:
                train.append(i)
            else:
                test.append(i)
        self.train = train
        self.test = test
        return (self.train, self,test)
    
    # 根据输入的x预测类标y
    def check(self, x):
        # 获取目标点离训练集其余点的距离字典
        dis = self._get_all_distance(x)
        # 获取距离最小的k个对象
        items = self._get_kmin_point(x, dis)
        # 获取所有的类标号
        label = self._get_all_class_label()
        # 声明类标号字典并初始化
        label_dic = {}
        for i in label:
            label_dic[i] = 0
        # 记录距离最小的k个对象中每种标号出现的次数
        for i in items:
            label_dic[i.y] = label_dic[i.y] + 1
        # 获取出现次数最大的类标
        class_label = max(label_dic, key=label_dic.__getitem__)
        return class_label
    
    # 获取某个点到其余点的距离
    def _get_all_distance(self, x):
        """
        arg:
            x:      tuple类型,为一个点的坐标,要求与其余点拥有一样的长度
        return:
            dis:    训练集中每一个样本对这个点的距离,为字典类型,通过训练集point对象索引索引
        """
        dis = {}
        for i in self.train:
            d = Point.get_distance(x, i.x)
            dis[i] = d
        return dis
    
    # 获取距离目标点x最小的k个point
    def _get_kmin_point(self, x, dis):
        ans = []
        for i in sorted(dis, key=dis.__getitem__):
            if len(ans) < self.k:
                ans.append(i)
            else:
                break
        return ans
    
    # 获取训练集中所有可能的类标号
    def _get_all_class_label(self):
        label = []
        for i in self.data:
            label.append(i.y)
        return sorted(list(set(label)))
    
    # 获取测试集准确性
    def check_test(self):
        total = len(self.test)
        correct = 0
        for i in self.test:
            if self.check(i.x) == i.y:
                correct = correct + 1
        return float(correct)/total


# 程序入口
if __name__ == '__main__':
    print('Survival speculation of Titanic!')
    t = Titanic('titanic.dat')
    accuracy_dic = {}
    for i in range(1, 10, 2):
        begin_time = time.time()
        print("K=%d, Started!"%(i))
        knn = KNN(t.data, i)
        accuracy = knn.check_test()
        accuracy_dic[i] = accuracy
        end_time = time.time()
        print("K=%d, Caculating using %lfs, Accuracy: %lf%%"%(i, end_time-begin_time, 100*accuracy), end='\n\n')
    pass

"""Answer
K=1, Caculating using 2.214107s, Accuracy: 57.207207%

K=3, Caculating using 2.134266s, Accuracy: 57.055215%

K=5, Caculating using 2.181503s, Accuracy: 78.273381%

K=7, Caculating using 2.066176s, Accuracy: 78.816199%

K=9, Caculating using 2.081948s, Accuracy: 76.887519%
"""