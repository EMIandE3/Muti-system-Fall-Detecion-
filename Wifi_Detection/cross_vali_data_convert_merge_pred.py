import numpy as np, numpy
import csv
import glob
import os
'''#########################这个文件用于数据的预处理操作###########################'''

window_size = 1000  # 窗的大小，这里为啥也有窗？
threshold = 60  #
slide_size = 200  # less than window_size!!! 这是窗滑动的距离
n_class = 5


def dataimport(path1):
    xx = np.empty([0, window_size, 90], float)  # 返回一个新的数组，其元素不进行初始化，3维的numpy数组

    ###Input data###
    print(1)
    # data import from csv
    # 批量抓取同文件名类型的文件，文件名相似就可以抓取，然后还用sort进行排序了
    # 我将sorted改成sort看看能不能行
    input_csv_files = sorted(glob.glob(path1))  # 对这个路径下的文件排序，我们可以用它批量抓取某种格式、或者以某个字符打头的文件名
    print(input_csv_files)
    # 这里排序是为了数据输入文件和标签文件能对应起来
    for f in input_csv_files:  # 遍历这些文件
        print("input_file_name=", f)
        data = [[float(elm) for elm in v] for v in csv.reader(open(f, "r"))]  # 读出数据，将它们强制转成浮点数
        tmp1 = np.array(data)  # 变换成numpy格式，数据还是那么多
        x2 = np.empty([0, window_size, 90], float)  # 创建一个空的等会用

        # data import by slide window  通过滑窗来进行数据导入处理
        k = 0   # len(tmp1)是返回这个文件数据的行数
        while k <= (len(tmp1) + 1 - 2 * window_size):  # 在进行数据的滑窗处理
            # np.dstack在列上面进行拼接，把一列的数据搞成一个向量，即window_size行做成一个向量
            x = np.dstack(np.array(tmp1[k:k + window_size, 0:90]).T)  # 1:91是输入，现在我的数据就是0:90
            x2 = np.concatenate((x2, x), axis=0)  # 再将数据连在一起，后面就是所有的数据了，拼成一行多个向量
            k += slide_size   # 窗滑动的步长是slide_size，上传还是不能偷懒的，还是会重叠

        xx = np.concatenate((xx, x2), axis=0)  # 完成数组的拼接，也是拼成一行
    xx = xx.reshape(len(xx), -1)  # 然后在处理一下，将数据变成len(xx)这么多行

    # ##Annotation data###  # 标签的处理
    # data import from csv
    # 同类文件的抓取并排好序


    print(xx.shape)

    return xx


'''   '############### ### Main ####  这里是函数运行的地方##########################'''
if not os.path.exists("input_files/"):  # 如果不存在存在这个文件夹，现在我存在了,这是最终输入数据的文件夹
    os.makedirs("input_files/")  # 如果不存在就创建这个文件夹
# 'env', 'fall', 'squat', 'walk',
for i, label in enumerate(['fall']):  # 用来按照标签读取文件 'env','fall','squat','walk','sitdown','pickup'
    # i是索引，label是具体的值“bed”那些
    filepath1 = "./predit/final_test_data/" + label + "*.csv"  # 文件地址
    # 下面两个是输出文件的地址，输入和标签
    outputfilename1 = "./input_files_tst/xx_" + str(window_size) + "_" + str(threshold) + "_" + label + ".csv"

    x = dataimport(filepath1)  # 这个是对同一类的数据读取并进行预处理的
    # 处理后，再将数据保存
    with open(outputfilename1, "w") as f:  # 这是要把同一类的所有文件写到同一个文件里面呀
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(x)  # 按行写入x, writerows()将一个二维列表中的每一个列表写为一行。

    print(label + "finish!")
