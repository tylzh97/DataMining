import numpy as np
import pandas as pd


def retrieve_data(path):
    dt = []
    with open(path, 'r')as f:
        f_l = f.readlines()[8:]
        for i in f_l:
            ll = i.strip('\n').split(',')
            l0 = list(map(lambda x: float(x), ll))
            dt.append(l0)
    return np.array(dt)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def diff_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Network(object):

    def __init__(self):
        self.sizes = [3, 4, 1]
        self.biases = [2 * np.random.random((1, y)) - 1 for y in self.sizes[1:]]
        self.weights = [2 * np.random.random((x, y)) - 1 for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.lrate = 0.1  # 将学习率设置为0.1
        print(self.biases)
        print(self.weights)

    def prepare_data(self, data):
        boundary = int(len(data) * 0.7)
        np.random.shuffle(data)
        dataset = data[:, :3]
        labelset = data[:, -1]
        # 将数据集进行随机选取前70%作为训练数据
        self.trainset = dataset[:boundary]
        self.trainlabel = labelset[:boundary]
        # 将数据集进行随机选取后30%作为测试数据
        self.testset = dataset[boundary:]
        self.testlabel = labelset[boundary:]

    # 前向传播过程
    def feed_forward(self, a):
        a_lst = []
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(a, w) + b)  # 计算每一层的输出作为下一层的输入
            a_lst.append(a)
        return a_lst

    # 反向传播过程
    def backprop(self, data=None):
        self.prepare_data(data)
        aver_cost = 0
        for x, y in zip(self.trainset, self.trainlabel):
            lst = self.feed_forward(x)
            a = lst[-1]
            hidden_o = lst[0]
            diff_z = hidden_o * (1 - hidden_o)
            sigma1_3 = -(y - a) * a * (1 - a)
            matrixh_o = self.weights[-1].T
            delta_wh = np.vstack(i * diff_z * matrixh_o * sigma1_3 for i in x)
            delta_wo = sigma1_3 * hidden_o.T
            self.weights = [self.weights[0] - self.lrate * delta_wh, self.weights[1] - self.lrate * delta_wo]
            delta_bh = diff_z * matrixh_o * sigma1_3
            delta_bo = sigma1_3
            self.biases = [self.biases[0] - self.lrate * delta_bh, self.biases[1] - self.lrate * delta_bo]
            # print(self.biases)
            # print(y, a) if y == 1 and a[0][0]>0.5 else None
            aver_cost += 1 / 2 * np.square(y - a)
        return self.weights, self.biases

    def calculate_accuracy(self):
        sum = 0
        for x, y in zip(self.testset, self.testlabel):
            # print(self.feed_forward(x)[-1][0])
            if self.feed_forward(x)[-1] >= 0.5 and y == 1:
                sum += 1
            elif self.feed_forward(x)[-1] < 0.5 and y == -1:
                sum += 1
        accuracy = sum / len(self.testlabel)
        print("测试集的正确率达到了" + str(100 * accuracy)[:5] + "%")


if __name__ == "__main__":
    path = 'titanic.dat'
    data = retrieve_data(path)
    network = Network()
    for i in range(10):
        network.backprop(data)
    network.calculate_accuracy()
