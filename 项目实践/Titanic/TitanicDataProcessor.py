# !/usr/bin/env python
# -*- coding:UTF-8 -*-

__author__ = 'Lizhenghao'
__package__ = 'TitanicDataProcessor'
__version__ = '0.0.1'

import numpy as np
import random
import math
import os


# 一个高维向量,为一个高维的向量,封装了一些方法.用于表示Record的属性值
class Vector(object):
    def __init__(self, data: tuple):
        if not (isinstance(data, tuple) or isinstance(data, list)):
            raise TypeError("Type of attribute 'data' should be tuple-like.")
        # protect类型的list变量,确保外界不可访问
        self._data = list(data)
        # private变量,确保数据不可被修改
        self.__length = len(self._data)

    def __str__(self):
        return str(self.data)

    @property
    def data(self):
        # 每次返回一个新的构造元祖,确保原本对象不会被修改
        return tuple(self._data)

    @property
    def length(self):
        return self.__length

    # 获取两点的距离
    @staticmethod
    def get_distance(p1, p2, method='E'):
        # 判断两个参数长度是否相同
        if not p1.length == p2.length:
            raise AttributeError("Points which need to computing should have the same dimension.")

        # 闭包函数,欧几里得距离算子
        def euclid(x1: tuple, x2: tuple):
            ans = 0
            for i in range(len(x1)):
                ans = ans + (x1[i] - x2[i]) ** 2
            ans = math.sqrt(ans)
            return ans

        # 闭包函数,曼哈顿算子
        def manhattan(x1: tuple, x2: tuple):
            ans = 0
            for i in range(len(x1)):
                ans = ans + abs(x1[i] - x2[i])
            return ans
        # 距离算法字典 Distance algorithm
        da = {
            'E': euclid,
            'M': manhattan,
        }
        try:
            func = da[method]
        except KeyError as e:
            raise AttributeError("Wrong distance method!\n" + str(e))
        return func(p1.data, p2.data)

    # 重写[]方法
    def __getitem__(self, item):
        return self._data[item]

    # 判断两个向量是否能运算
    @staticmethod
    def _computable(p1, p2):
        ans = True
        if not p1.length == p2.length:
            ans = False
        acceptable_types = (int, float)
        for i in range(p1.length):
            if not (type(p1[i]) in acceptable_types and type(p2[i]) in acceptable_types):
                ans = False
        return ans

    # 重写+号方法
    def __add__(self, other):
        try:
            if not __class__._computable(self, other):
                raise ValueError("These two vectors cannot addable.")
        except:
            raise AttributeError("Vector type could only add Vector object.")
        data = []
        for i in range(self.length):
            data.append((self[i] + other[i]))
        return __class__(data=tuple(data))


# 属性-类标的一条记录类,包含了属性与类标
class Record(object):
    def __init__(self, attributes: Vector, label):
        """
        :param attributes: 记录的属性值,为一个Vector变量
        :param label: 记录的类标,为Object的子类
        """
        self._attributes = attributes
        self._label = label

    def __str__(self):
        return str("{0} --> {1}".format(self.x, self.y))

    # get方法,使用简短名称x获取属性值
    @property
    def x(self):
        return self._attributes

    # get方法,使用简短名称y获取类标值
    @property
    def y(self):
        return self._label

    @property
    def length(self):
        return self._attributes.length


# 泰坦尼克数据集
class Titanic(object):
    """
    用于初始化以及处理Titanic数据集
    args:
        data_path:  data文件的存放路径,支持相对路径
        separtor:   data文件中使用的分隔符,可缺省,默认为','
        commenting: data文件中的代码注释符,可缺省,默认为'@'
        goal_column: data文件中,待遇测属性所在的行数,遵循python风格,默认为最后一行"-1"
        data_range: data文件中,需要映射的数据范围
    return:
        None
    """

    def __init__(self, data_path, separator=',', commenting='@', goal_column=-1, data_range=(0, 1)):
        self.goal_column = goal_column
        self.range = data_range
        self.data_path = data_path
        # 判断规则文件是否存在,不存在则抛出文件不存在异常
        if not os.path.isfile(self.data_path):
            raise FileExistsError("Screen file(%s) does't exists!" % self.data_path)
        # 打开数据文件
        data_file = open(data_path, 'r')
        # 将所有数据读取到内存中,该方法非常消耗内存,在数据集过大时可能会奔溃
        data_source = data_file.readlines()
        # 关闭数据文件
        data_file.close()
        # 声明数据变量
        data = []
        # 分离注释与数据
        for i in data_source:
            # 判断当前行是否为注释
            if not i[0].replace(' ', '').replace('\t', '').startswith(commenting):
                temp = i.replace('\n', '').split(separator)
                temp_line = []
                # 将文本数据转化为float类型
                for j in temp:
                    temp_line.append(float(j))
                data.append(temp_line)
        # 将数据设置为私有变量
        self.data = data
        # self.range_all()
        self._record_to_data()
        self.len_of_vector = self.data[0].length

    # 获取各行数据
    # 实际操作为取矩阵转置
    def _get_columns(self):
        temp_data = []
        for i in self.data:
            if isinstance(i, Record):
                temp_data.append(i.x)
            else:
                temp_data.append(i)
        return list(zip(*temp_data))

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
        y = lambda x: (data_range[1] - data_range[0]) * (x - _min) / (_max - _min) + data_range[0]
        ans = []
        for i in data:
            ans.append(y(float(i)))
        return ans

    # Range所有列的数据
    def range_all(self, data_range=(0, 1)):
        colums = self._get_columns()
        goal = colums[self.goal_column]
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
    def _record_to_data(self):
        out = []
        for i in self.data:
            # 剥离属性值
            x = i[:-1] if self.goal_column == -1 else i[:self.goal_column] + i[self.goal_column + 1:]
            y = i[self.goal_column]
            attribute = Vector(x)
            label = y
            p = Record(attribute, label)
            out.append(p)
        self.data = out
        return self.data

    # 获取信息熵
    @staticmethod
    def _information_entropy(dataset: tuple):
        """
        需要计算的数据集,存储对象为Record对象
        :param dataset: 一个仅包含Record对象的可索引对象
        :return: 信息熵
        """
        # 获取数据集的类标数组
        aim = [temp.y for temp in dataset]
        # 获取记录长度
        a_length = len(aim)
        # 用于记录每个类标出项次数的字典
        typo = {}
        for i in aim:
            if i in typo.keys():
                typo[i] = typo[i] + 1
            else:
                typo[i] = 0
        # 记录每个属性值出现概率的数组
        p = {}
        for i in typo.keys():
            p[i] = float(typo[i])/a_length
        entropy = 0
        p_list = list(p.values())
        for i in range(len(p_list)):
            temp = 0 - (p_list[i] * math.log(p_list[i], 2))
            entropy = entropy + temp
        return entropy

    # 获取信息增益,此方法功能不纯净,包含有数据二值化的功能,将会在后续的版本中将功能独立分离
    @staticmethod
    def gain(dataset, column, threshold):
        col_dic = {
            0: [],
            1: [],
        }
        for i in dataset:
            if i.x[column] <= threshold:
                col_dic[0].append(i)
            else:
                col_dic[1].append(i)
        ea = 0
        for i in col_dic.values():
            ea = ea + abs(len(i)/len(dataset)) * __class__._information_entropy(i)
        gain = __class__._information_entropy(dataset) - ea
        return gain

    # 获取每个属性的所有可能标号值
    def _get_items(self):
        # 用于存放结果的一个变量,存放的是每个属性值的所有可能取值集合
        ans = []
        # 获取矩阵转置
        attributes = self._get_columns()
        for i in attributes:
            temp = set(i)
            ans.append(temp)
            # 释放临时变量
            del temp
        # 释放矩阵转置变量
        del attributes
        return ans

    # 获取所有可能的阈值分割点,为每一个属性的可能取值±一个步长
    def _get_split_point(self):
        # 获取每一个属性所有可能的取值
        items = self._get_items()
        ans = []
        for i in items:
            _min = min(i)
            _max = max(i)
            tl = sorted(list(i))
            step = _max - _min
            for k in range(1, len(tl)):
                step = step if step > (tl[k] - tl[k-1]) else (tl[k] - tl[k-1])
            step = step / 10
            temp = []
            for j in i:
                if j == _min:
                    temp.append(j + step)
                elif j == _max:
                    temp.append(j - step)
                else:
                    temp.append(j - step)
                    temp.append(j + step)
            ans.append(sorted(temp))
        return ans

    # 针对每一个属性,都获取一个最好的阈值
    # 能够获取最佳阈值分割点,但是并不会修改数据
    def _get_best_threshold(self):
        ans = []
        for i in range(self.len_of_vector):
            temp = self._get_split_point()
            temp_gain = []
            for j in temp[i]:
                temp_gain.append(self.gain(self.data, i, j))
            _max_gain = max(temp_gain)
            ans.append(temp[i][temp_gain.index(_max_gain)])
        return ans

    # 根据属性阈值对数据属性进行二值化
    # 该操作会改变self.data的值,需要谨慎调用该方法
    def binaryzation_with_threshold(self):
        # 根据信息增益,获取阈值
        threshold = self._get_best_threshold()
        attribute_T = self._get_columns()
        temp_a = []
        for i in range(len(attribute_T)):
            temp = []
            for j in attribute_T[i]:
                wait_to_append = 0 if j < threshold[i] else 1
                temp.append(wait_to_append)
            temp_a.append(temp)
        # 转置属性,重新回到
        temp_a = list(zip(*temp_a))
        temp_data = []
        for i in range(len(temp_a)):
            temp_data.append(Record(Vector(temp_a[i]), self.data[i].y))
        del temp_a
        del self.data
        self.data = temp_data
        del temp_data
        return self.data

    # 获取记录的所有属性值
    def get_attribute(self):
        temp = []
        for i in self.data:
            temp.append(i.x)
        return list(zip(*temp))

    # 将数据numpy化
    def numpylize(self):
        temp_ans = []
        for i in self.data:
            temp_0 = np.array(i.x.data).reshape((len(i.x.data), 1))
            out = 1 if i.y == 1 else 0
            temp_1 = np.array([[out]])
            temp = (temp_0, temp_1)
            temp_ans.append(temp)
        self.data = temp_ans
        return temp_ans

    def split(self, p):
        train = []
        test = []
        for i in self.data:
            if random.random() < p:
                train.append(i)
            else:
                test.append(i)
        return (train, test)
