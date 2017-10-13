"""
贝叶斯算法
"""
from numpy import *


def load_data_set():
    """
    产生数据集
    :return:
    """
    postin_list = [["my", "dog", "has", "flea", "problems", "help", "please"],
                    ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
                    ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
                    ["stop", "posting", "stupid", "worthless", "garbage"],
                    ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
                    ["quit", "buying", "worthless", "dog", "food", "stupid"]]
    class_vec = [0, 1, 0, 1, 0, 1]
    return postin_list, class_vec


def create_vocab_list(data_set):
    """
    返回一个包含所有可能出现的词的list
    :param data_set:
    :return:
    """
    vocab_set = set([])
    for document in data_set:
        # 遍历所有的留言,找出所有可能出现的词
        vocab_set = vocab_set | set(document)  # 用于求集合的并集
    return list(vocab_set)


def set_of_word_2_vec(vocab_list, input_set):
    """
    将输入文档转换为向量
    :param vocab_list: 词汇表
    :param input_set: 输入的文档
    :return: 文档向量
    """
    return_vec = [0] * len(vocab_list)  # 创建一个和词汇表等长的文档向量
    for word in input_set:
        # 遍历输入的文档
        if word in vocab_list:
            # 如果这个词在词汇表中,就把文档向量对应的词置成1
            return_vec[vocab_list.index(word)] = 1
        else:
            # 这个词不在词汇表中,一般不可能。
            print("the word: {} is not in my vocabulary!".format(word))
    return return_vec


def bag_of_words_2_vec_mn(vocab_list, input_set):
    """
    将输入文档转换为词袋模型向量
    :param vocab_list: 词汇表
    :param input_set: 输入的文档
    :return:
    """
    return_vec = [0] * len(vocab_list)  # 创建一个和词汇表等长的文档向量
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1  # 词袋模型将响亮对应值直接加1
    return return_vec


def train_nb0(train_matrix, train_category):
    """
    朴素贝叶斯训练器函数
    :param train_matrix: 用于训练的文档的矩阵
    :param train_category: 训练的文档的分类信息
    :return:
    """
    num_train_docs = len(train_matrix)  # 被训练文档的数量
    num_words = len(train_matrix[0])  # 词汇表的长度
    p_abusive = sum(train_category) / float(num_train_docs)  # 侮辱性文档在出现的频率
    # 初始化词出现次数统计向量和总词数统计变量
    # p_0_num = zeros(num_words)
    # p_1_num = zeros(num_words)
    # 改进初始化成全是1的矩阵,避免出现0导致频率乘积为0.
    p_0_num = ones(num_words)
    p_1_num = ones(num_words)
    p_0_denom = 2.0
    p_1_denom = 2.0
    for i in range(num_train_docs):
        # 循环所有的被训练文档
        if train_category[i] == 1:
            # 如果这个文档的分类是1,说明是侮辱性文档
            p_1_num += train_matrix[i]  # 将文档向量累加起来
            p_1_denom += sum(train_matrix[i])  # 将侮辱文档的词汇数量累加起来
        else:
            # 如果不是侮辱性文档,也累加起来
            p_0_num += train_matrix[i]
            p_0_denom += sum(train_matrix[i])
    # 用累加起来的每个词的数量减去累加起来的总体出现的词的数量,用于得出这个词的频率
    # p_1_vect = p_1_num / p_1_denom  # 计算每个词在侮辱性文档中的频率向量
    # p_0_vect = p_0_num / p_0_denom  # 计算词在非侮辱性文档中的频率向量
    # 改进,对乘积取自然对数,避免下溢出。即乘数太小导致结果为0
    p_1_vect = log(p_1_num / p_1_denom)
    p_0_vect = log(p_0_num / p_0_denom)
    return p_0_vect, p_1_vect, p_abusive


def classify_nb(vec_2_classify, p_0_vec, p_1_vec, p_class1):
    """
    朴素贝叶斯的分类器函数
    :param vec_2_classify: 要被分类的向量
    :param p_0_vec: 词分类为0的概率向量
    :param p_1_vec: 词分类为1的概率向量
    :param p_class1: 向量分类为1的频率
    :return:
    """
    # 计算要分类的向量和概率自然对数向量的乘积,再求和,相当于得到了所有概率乘积的对数,最后加上总的发生的频率的对数,得到可以用来比较的概率的对数。
    p1 = sum(vec_2_classify * p_1_vec) + log(p_class1)
    p0 = sum(vec_2_classify * p_0_vec) + log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # 测试数据集和文档向量相关函数
    # posting_list, class_vec = load_data_set()
    # vocab_list = create_vocab_list(posting_list)
    # print(vocab_list)
    # print(set_of_word_2_vec(vocab_list, posting_list[0]))
    # 测试朴素贝叶斯训练器
    postin_list, class_vec = load_data_set()
    vocab_list = create_vocab_list(postin_list)
    train_mat = []
    for postin_doc in postin_list:
        train_mat.append(set_of_word_2_vec(vocab_list, postin_doc))
    p0v, p1v, p_ab = train_nb0(train_mat, class_vec)
    # print(p0v)
    # print(p1v)
    # print(p_ab)
    # 测试朴素贝叶斯分类器
    test_entry = ["love", "my", "dalmation"]
    this_doc = array(set_of_word_2_vec(vocab_list, test_entry))
    print(classify_nb(this_doc, p0v, p1v, p_ab))
    test_entry2 = ["stupid", "garbage"]
    this_doc = array(set_of_word_2_vec(vocab_list, test_entry2))
    print(classify_nb(this_doc, p0v, p1v, p_ab))

