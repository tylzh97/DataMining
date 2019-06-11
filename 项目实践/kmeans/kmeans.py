# !/usr/bin/env python
# -*- coding:UTF-8 -*-

__author__ = 'Lizhenghao'
__version__ = '0.0.1'

import random
import math
import time
import os


# 属性标号类,用于记录多个属性值
# 可以理解为高维的点,其中仅包含坐标值,不包含类标值
class Point(object):
    def __init__(self, *data):
        self.data = tuple(data)
        self.length = len(self.data)
        pass

    def __str__(self):
        return str(self.data)

    @staticmethod
    def get_distance(p1, p2, method='E'):
        """
        ! 该方法为不安全方法,若参数错误则会raise错误,在使用时需要try-catch
        获取两个Point的距离,可选距离函数method,返回float类型
        :param p1: Point 1;
        :param p2: Point 2;
        :param method: The distance method you want to use;
        :return: float type, The distance between these two Point;
        """
        # 判断参数是否为Point类型,若不是则报参数异常错误
        if not (isinstance(p1, Point) and isinstance(p2, Point)):
            raise AttributeError("Two attribute should be Point style!")
        # 判断二者长度是否相等,若不是则报长度异常错误
        if not p1.length == p2.length:
            raise AttributeError("Two attribute should have same length!")

        # 闭包函数,欧几里得距离算子
        def Euclid(x1: tuple, x2: tuple):
            ans = 0
            for i in range(len(x1)):
                ans = ans + (x1[i] - x2[i]) ** 2
            ans = math.sqrt(ans)
            return ans

        # 闭包函数,曼哈顿算子
        def Manhattan(x1: tuple, x2: tuple):
            ans = 0
            for i in range(len(x1)):
                ans = ans + abs(x1[i] - x2[i])
            return ans

        # 距离算法枚举
        # Distance algorithm
        da = {
            'E': Euclid,
            'M': Manhattan,
        }
        func = None
        try:
            func = da[method]
        except:
            raise AttributeError("Wrong distance method!")
        return float(func(p1.data, p2.data))

    @staticmethod
    def get_center(*points):
        """
        ! 该方法为不安全方法,若参数错误则会raise错误,在使用时需要try-catch
        传入有限个点,计算有限个点的中心点.
        要求为Point类型,且每个点长度相同
        :param points: Several Point type variable which has same length.
        :return: Points type center point.
        """
        # 检查参数类型是否正确
        for i in points:
            if not isinstance(i, Point):
                raise AttributeError("Attribute type should be Point type.")
        point_length = points[0].length
        # 检查点长度是否一致
        for i in points:
            if not i.length == point_length:
                raise AttributeError("Length of Points should have same length.")
        # 初始化中心点坐标
        ans = [0] * point_length
        for i in points:
            for j in range(i.length):
                ans[j] = ans[j] + i.data[j]
        for i in range(len(ans)):
            ans[i] = ans[i] / len(points)
        return Point(*ans)


# 数据集通用处理方法,包含了数据集划分等常用功能
class Dataset(object):
    def __init__(self, *dataset, scale: float = 1):
        if not (scale >= 0 and scale <= 1):
            raise AttributeError("Attribute 'scale' should between 0 and 1.")
        self.scale = scale
        self.dataset = dataset
        if not self._split():
            raise RuntimeError("Error when split dataset.")
        self.length = len(dataset)

    def _split(self):
        """
        用于划分数据集,默认方法为随机划分
        该方法为protect方法,应该在初始化时执行
        :return: bool type, True oe False
        """
        try:
            train = []
            test = []
            for i in self.dataset:
                if random.random() < self.scale:
                    train.append(i)
                else:
                    test.append(i)
            self.train = train
            self.test = test
            return True
        except:
            return False


# K-Means算法
class Kmeans(object):
    def __init__(self, data: Dataset, *args, k: int = 0, begin_points: tuple = None):
        self.data = data.dataset
        self.k = k
        self.centers = begin_points
        self.clust = None
        if not self._choose_centers():
            raise RuntimeError("Fail to initialize objects. Wrong when choose centers.")

    # 根据传入参数调整起始点,其中begin_points的优先级高于k
    # 若仅传入K,则随机选取k个点作为起始点,否则则使用begin_points中的点
    def _choose_centers(self):
        try:
            if self.centers:
                self.k = len(self.centers)
                return True
            self.centers = random.sample(self.data, self.k)
            return True
        except:
            return False

    # 开始进行聚类操作
    def start(self, loop=100):
        last_center = self.centers
        i = None
        for i in range(loop):
            clust = self.cluster()
            self.clust = clust
            self.centers = tuple(clust.keys())
            if last_center == self.centers:
                break
            last_center = self.centers
        return i if i < loop else 0

    # 进行一次聚类操作,该操作会返回最新的中心点以及对应簇的元素元祖
    def cluster(self):
        # 闭包函数,转置一个二维矩阵
        def T(lst):
            return list(zip(*lst))
        distance = []
        # 获取每个点到中心点的距离
        for i in self.centers:
            temp = []
            for e in self.data:
                temp.append(Point.get_distance(i, e))
            distance.append(temp)
        # 获取距离的转置
        distance_T = T(distance)
        # 初始化簇
        clust = []
        for i in range(len(distance)):
            clust.append([])
        # 获取隶属于每个簇的元素
        for i in range(len(self.data)):
            min_index = distance_T[i].index(min(distance_T[i]))
            clust[min_index].append(self.data[i])
        # 获取新的中心点
        new_point = []
        for i in clust:
            new_point.append(Point.get_center(*i))
        # 新的中心点坐标,转换为Point类
        new_point = Point(*new_point)
        clust_dic = {}
        for i in range(len(clust)):
            clust_dic[new_point.data[i]] = clust[i]
        return clust_dic


# 程序入口
if __name__ == '__main__':
    print('Hello World!')
    data = [
        Point(3, 4),
        Point(3, 6),
        Point(7, 3),
        Point(4, 7),
        Point(3, 8),
        Point(8, 5),
        Point(4, 5),
        Point(4, 1),
        Point(7, 4),
        Point(5, 5),
    ]
    data = Dataset(*data)
    km = Kmeans(data, begin_points=(data.dataset[1], data.dataset[7]))
    # km.cluster()
    km.start(loop=10)
    pass
