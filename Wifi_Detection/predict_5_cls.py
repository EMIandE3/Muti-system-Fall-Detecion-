# -*- coding:utf-8 -*-
import argparse

import numpy as np
import tensorflow as tf
from cross_vali_input_data_train import csv_import, DataSet

window_size = 500  # 为啥这里和数据预处理的里面不同了呀

# Parameters
batch_size = 1

# Network Parameters
n_input = 90  # WiFi activity data input (img shape: 90*window_size)
n_steps = window_size  # timesteps


def load_graph(frozen_graph_filename):  # 装载图文件，只用装载一次就行了
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
    #                     help="Frozen model file to import")
    # args = parser.parse_args()
    # 加载已经将参数固化后的图
    frozen_model_filename = "./my_model/frozen_model.pb"  # 这里是固化后的文件名，这里要改变
    graph = load_graph(frozen_model_filename)  # 这个应该在预测前就装载好模型，只进行装载一次就行

    f = open('lable_test_5.txt',"w")
    x = graph.get_tensor_by_name('prefix/Placeholder:0')  # 节点定义也是只用定义一次就行了
    y = graph.get_tensor_by_name('prefix/ArgMax:0')

    # x_fall = csv_import()
    # xx = x_fall  # 将其连成一个大矩阵
    x_test= csv_import()
    # xx = np.r_[x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk]
    x_in = DataSet(x_test)  # 放到这里面，分批读取
    x_num = x_test.shape[0]

    # We can list operations
    # op.values() gives you a list of tensors it produces
    # op.name gives you the name
    # 输入,输出结点也是operation,所以,我们可以得到operation的名字
    # for op in graph.get_operations()[0:100]:
    #     print(op.name, op.values())
    #     # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    # 操作有:prefix/Placeholder/inputs_placeholder
    # 操作有:prefix/Accuracy/predictions
    # 为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    # 注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    # 我先注释掉下面，看看这个有哪些操作

    with tf.Session(graph=graph) as sess:  # 打开这个图文件也只用打开一次就行
        for i in range(x_num):
            batch_x = x_in.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))  # 输入的文件就是500*90的数据
            y_out = sess.run(y, feed_dict={
                x: batch_x  # < 45
            })
            print(y_out)  # [[ 0.]] Yay!  # 在这里面判断y_out就行了如果出现连续的1,那就认为是一次摔倒事件
            f.write(str(y_out)+'\n')

        '''
        batch_x = x_in.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        y_out = sess.run(y, feed_dict={
            x: batch_x  # < 45
        })
        print(y_out)  # [[ 0.]] Yay!
        '''
    print("finish")
    f.close()
