"""
kNN算法实例
"""
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def create_data_set():
    """
    产生数据集
    :return:
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    kNN基础分类器
    :param in_x: 要分类的数据
    :param data_set: 已有数据集
    :param labels: 标签集
    :param k: k个临近元素用于投票
    :return: 排序后的投票list
    """
    data_set_size = data_set.shape[0]  # 数据集的大小,shape返回一个tuple,第一个元素是数据的数目,第二个是一个数据中的维度数据。
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set  # tile函数将输入的向量拼成数据集那么大的相同格式的矩阵,然后和数据集做减法。
    sq_diff_mat = diff_mat ** 2  # 计算的差进行平方,其中每一个元素进行平方运算,不是矩阵相乘。得到的还是矩阵。
    # sum函数的axis参数用于指定消除哪个或者哪些维度,也可以为tuple,默认是None即全相加。这里为1表示第二个维度相加在一起,即每条数据的和。
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5  # 计算的和进行开方,计算出欧氏距离。
    sorted_dist_indicies = argsort(distances)  # 进行排序,选出临近的。argsort函数用于输出排序后的序号。
    class_count = {}  # 用于记录投票结果
    for i in range(k):
        # 选出最近的k个进行投票
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1  # 有则加一,默认为0.
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # 投票结果排序
    return sorted_class_count


def show_figure():
    """
    展示图像
    :return:
    """
    group, label = create_data_set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(group[:, 0], group[:, 1])
    plt.show()


def auto_norm(data_set):
    """
    数据归一化
    :param data_set:数据集
    :return:
    """
    min_vals = data_set.min(0)  # 计算每列最小值
    max_vals = data_set.max(0)  # 计算每列最大值
    ranges = max_vals - min_vals  # 计算可能的取值范围
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))  # 使用原始数据减去最小值构成的矩阵得到差值
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # 使用差值除以取值范围得到归一化数据
    return norm_data_set, ranges, min_vals


if __name__ == '__main__':
    group, label = create_data_set()  # 取数据

    # numpy里的array用方括号index的时候,后一个数是维度,前一个slice形式的是这个维度上的slice
    # print(group)
    # print(group[:, 0])
    # print(group[1:, 0])
    # print(group[0:, 0])
    # print(group[:1, 0])
    # print(group[:0, 0])
    # print(group[1:2, 0])

    # print(classify0([0, 0], group, label, 3))  # 测试基础kNN

    print(auto_norm(group))  # 测试数据归一化函数
