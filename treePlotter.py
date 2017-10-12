"""
绘制决策树
"""
import matplotlib.pyplot as plt


# 定义文本框和箭头的样式
decision_node = {"boxstyle": "sawtooth", "fc": "0.8"}
leaf_node = {"boxstyle": "round4", "fc": "0.8"}
arrow_args = {"arrowstyle": "<-"}


# def create_plot():
#     """
#     绘制图像
#     :return:
#     """
#     fig = plt.figure(1, facecolor="white")  # 创建了一个绘图区
#     fig.clf()  # 清空
#     create_plot.ax1 = plt.subplot(111, frameon=False)  # 定义一个全局的绘图区。这是一个全局变量,在下面的函数中也用到了。
#     # 绘制了两个节点
#     plot_node("decision node", (0.5, 0.1), (0.1, 0.5), decision_node)
#     plot_node("leaf node", (0.8, 0.1), (0.3, 0.8), leaf_node)
#     plt.show()


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制一个节点,利用了Matplotlib提供的文本注解功能绘制节点
    :param node_txt:注解信息,这里用于节点标签
    :param center_pt:
    :param parent_pt:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(node_txt,
                             xy=parent_pt,
                             xycoords="axes fraction",
                             xytext=center_pt,
                             textcoords="axes fraction",
                             va="center",
                             ha="center",
                             bbox=node_type,
                             arrowprops=arrow_args)


def get_num_leaf(my_tree):
    """
    获取树的叶子数量
    :param my_tree: 树的dict
    :return:
    """
    num_leafs = 0
    first_str = list(my_tree.keys())[0]  # 获取字典的第一个字段的key,即第一次划分数据集的类别标签。从这里可以开始遍历整个树。
    second_dict = my_tree[first_str]  # 第一个字段里面的dict即后面的整个树
    for key in second_dict.keys():
        # 遍历整个树,如果value不是dict说明是叶子节点
        if type(second_dict[key]) == dict:
            num_leafs += get_num_leaf(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    """
    获取树的深度
    :param my_tree: 树的dict
    :return:
    """
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        # 遍历整个树,如果是不是dict说明不是叶子节点,进行累加和递归。
        if type(second_dict[key]).__name__ == "dict":
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    """
    生成测试用的树
    :param i: 返回第几棵树
    :return:
    """
    list_of_trees = [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                     {"no surfacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return list_of_trees[i]


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    在两个点的中间绘制一个文字
    :param cntr_pt:子节点
    :param parent_pt:父节点
    :param txt_string:要绘制的文字
    :return:
    """
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    """
    绘制一个树
    :param my_tree:树的数据
    :param parent_pt:
    :param node_txt:
    :return:
    """
    num_leafs = get_num_leaf(my_tree)  # 获取子节点数目
    depth = get_tree_depth(my_tree)  # 深度
    first_str = list(my_tree.keys())[0]  # 根的label
    # 当前绘制节点的位置。
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)  # 绘制当前节点和父节点中间的标签信息
    plot_node(first_str, cntr_pt, parent_pt, decision_node)  # 绘制当前节点
    second_dict = my_tree[first_str]  # 这个节点的子tree信息
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d  # 更新全局的位置信息
    for key in second_dict.keys():
        # 循环绘制所有的子节点
        if type(second_dict[key]) == dict:
            # 如果节点是dict,直接递归绘制一个树
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            # 如果不是dict,说明是叶子节点,需要绘制叶子节点
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w  # 更新位置信息
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)  # 绘制叶子节点
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))  # 绘制本节点和父节点之间的标签信息
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d  # 更新位置信息


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = {"xticks": [], "yticks": []}
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leaf(in_tree))  # 叶子节点数目,即树的宽度
    plot_tree.total_d = float(get_tree_depth(in_tree))  # 树的深度,即高度
    plot_tree.x_off = -0.5 / plot_tree.total_w  # 当前绘制节点的位置x偏移,每次绘制会更新
    plot_tree.y_off = 1.0  # 当前绘制节点的位置y偏移,每次绘制会更新
    plot_tree(in_tree, (0.5, 1.0), "")  # 开始绘制这个树
    plt.show()


if __name__ == '__main__':
    # 基础函数测试
    # create_plot()
    # 测试树深度和叶节点函数
    test_tree = retrieve_tree(0)
    # print(get_num_leaf(test_tree))
    # print(get_tree_depth(test_tree))
    # 测试画出决策树
    create_plot(test_tree)
