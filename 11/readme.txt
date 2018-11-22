运行方法：
    python Tree.py
可添加的选项有：
    --best_parameter：若值为1，进行自动调参
    --post_pruning：若值为1，进行后剪枝
    --print_tree：若值为1，生成树的json文件
    --ignore：若值为1，忽略第7、12、13个特征
运行示例：
    python Tree.py --ignore 1 --print_tree 1
说明：
    1、后剪枝耗时1-2小时，且由于验证集的选取是随机的，有时并不能提升正确率
    2、自动调参耗时约1小时
    3、经过多次调参，得到的最优参数为max_continous_son = 3, max_leaf_length = 35