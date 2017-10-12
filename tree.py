"""
决策树
"""
from math import log
import operator
import pickle


def create_data_set():
    """
    构建数据集
    :return:
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_ent(data_set):
    """
    计算数据集的香农熵
    :param data_set:数据集
    :return:
    """
    num_entries = len(data_set)  # 计算数据集中的实体总数
    label_counts = {}  # 用于记录所有标签数量的字典
    for feat_vec in data_set:
        # 这个循环用于统计数据集中所有类别的标签出现的频率
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0  # 初始化香农熵为0
    for key in label_counts:
        # 累加得到整体数据集的香农熵
        prob = float(label_counts[key]) / num_entries  # 计算标签的概率
        shannon_ent -= prob * log(prob, 2)  # 将-p(x)*log_2^{p(x)}累加起来
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    划分数据集的方法
    :param data_set:数据集
    :param axis: 划分的特征
    :param value:特征的值
    :return:新的数据集
    """
    ret_data_set = []  # 返回的list
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 如果遇到符合条件的向量,就将用来判断的维度剔除后加入返回的list中
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择划分后信息熵最好的特征
    :param data_set: 数据集
    :return: 特征
    """
    num_feature = len(data_set[0]) - 1  # 计算特征的数量
    base_entropy = calc_shannon_ent(data_set)  # 计算原始的信息熵
    # 初始化最好的信息增益和特征
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        # 循环所有的特征
        feat_list = [example[i] for example in data_set]  # 得到该特征的所有的值
        unique_vals = set(feat_list)  # 转化为set形式用于去重
        new_entropy = 0.0  # 这个特征的信息熵
        for value in unique_vals:
            # 循环这个特征可能的所有值
            sub_data_set = split_data_set(data_set, i, value)  # 根据这个值划分数据集
            prob = len(sub_data_set) / float(len(data_set))  # 计算这个子数据集占整体的频率
            new_entropy += prob * calc_shannon_ent(sub_data_set)  # 计算子数据集信息熵并累加起来得到整体信息熵
        info_gain = base_entropy - new_entropy  # 与原始信息熵做差得到这种划分的信息增益
        # 如果这个信息增益好于之前的,就将之存起来
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    投票决定叶子节点分类
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    递归创建决策树
    :param data_set: 数据集
    :param labels: 标签
    :return:
    """
    class_list = [example[-1] for example in data_set]  # 获取最后一列,即向量的类别
    if class_list.count(class_list[0]) == len(class_list):
        # 如果类别的list中只有一种,那么停止划分
        return class_list[0]
    if len(data_set[0]) == 1:
        # 所有特征都已经被消耗完了,还没分出来,只能投票返回出现最多的类别
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)  # 选择当前信息熵最小的划分数据集方式
    best_feat_label = labels[best_feat]  # 最好的特征的label
    my_tree = {best_feat_label: {}}  # 初始化由此开始的树
    del labels[best_feat]  # 从标签表中删除这个最好的特征
    feat_values = [example[best_feat] for example in data_set]  # 获取这个特征所有的特征值
    unique_vals = set(feat_values)  # 特征值去重
    for value in unique_vals:
        # 循环所有的特征值
        sub_labels = labels[:]  # 复制一下类标签用于传入递归下一级。避免list传引用导致的问题。
        # 递归构建决策树。按照当前最好特征划分子数据集。
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    决策树的分类器,利用已有的决策树来进行分类
    :param input_tree:生成的决策树
    :param feat_labels:特征标签的列表
    :param test_vec:测试向量
    :return:
    """
    first_str = list(input_tree.keys())[0]  # 取根节点标签
    second_dict = input_tree[first_str]  # 获取树信息
    feat_index = feat_labels.index(first_str)  # 特征标签的index
    class_label = None  # 用于返回的信息
    for key in second_dict.keys():
        # 遍历可能的子节点
        if test_vec[feat_index] == key:
            # 如果命中了特征值
            if type(second_dict[key]) == dict:
                # 如果子节点是一个树,就继续递归分类
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                # 如果是叶子节点,返回的分类就是这个节点的值
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    """
    序列化决策树存储到本地
    :param input_tree:
    :param filename:
    :return:
    """
    fw = open(filename, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):
    """
    读取序列化的决策树
    :param filename:
    :return:
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    fish_data_set, label = create_data_set()
    # 测试计算香农熵的函数
    # print(calc_shannon_ent(fish_data_set))
    # 测试划分数据集函数
    # print(split_data_set(fish_data_set, 1, 1))
    # 测试最好的数据集划分函数
    # print(choose_best_feature_to_split(fish_data_set))
    # 测试划分决策树函数
    # print(create_tree(fish_data_set, label))
    # 测试分类器函数
    # fish_tree = create_tree(fish_data_set, label[:])  # 生成树的时候会修改label这个list,所以要复制一份传进去
    # print(classify(fish_tree, label[:], [1, 1]))
    # 测试决策树的序列化
    # fish_tree = create_tree(fish_data_set, label[:])
    # store_tree(fish_tree, 'pickle_tree.info')
    print(grab_tree('pickle_tree.info'))
