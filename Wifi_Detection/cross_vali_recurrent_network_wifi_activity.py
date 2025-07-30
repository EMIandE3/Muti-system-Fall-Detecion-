from __future__ import print_function
import sklearn as sk
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib import rnn  # 我这个tensorflow版本下没有这个库了，要新创建一个环境了
from sklearn.model_selection import KFold, cross_val_score
import csv
from sklearn.utils import shuffle
import os

# Import WiFi Activity data
# csv_convert(window_size,threshold)
from cross_vali_input_data_train import csv_import, DataSet

window_size = 500  # 为啥这里和数据预处理的里面不同了，因为降低频率了
threshold = 60   # 这个和数据预处理里面是相同的

# Parameters
learning_rate = 0.0001
training_iters = 800  # 这里训练了2000代，好多要
batch_size = 200
display_step = 50

# Network Parameters
n_input = 90 # WiFi activity data input (img shape: 90*window_size)
n_steps = window_size # timesteps
n_hidden = 200 # hidden layer num of features original 200
n_classes = 6 # WiFi activity total classes  总共7类的分类,现在是6类

# Output folder
OUTPUT_FOLDER_PATTERN = "LR{0}_BATCHSIZE{1}_NHIDDEN{2}/"  # 这个自己设置好了
output_folder = OUTPUT_FOLDER_PATTERN.format(learning_rate, batch_size, n_hidden)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))  # //随机初始化权重
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):  # 定义自己的循环神经网落

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow  搞一个LSTM层
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

##### main #####
pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
cvscores = []
confusion_sum = [[0 for i in range(n_classes)] for j in range(n_classes)]  # 定义混淆矩阵

# data import  'env','fall','squat','walk','sitdown','pickup'
x_env, x_fall, x_squat, x_walk, x_sitdown, x_pickup, \
y_env, y_fall, y_squat, y_walk, y_sitdown, y_pickup = csv_import()

print(" env =",len(x_env), " fall=", len(x_fall), " squat =", len(x_pickup), " walk=", len(x_walk), " sitdown=", len(x_sitdown), " pickup=", len(x_pickup))

# data shuffle 把数据打乱
x_env, y_env = shuffle(x_env, y_env, random_state=0)
x_fall, y_fall = shuffle(x_fall, y_fall, random_state=0)
x_squat, y_squat = shuffle(x_squat, y_squat, random_state=0)
x_walk, y_walk = shuffle(x_walk, y_walk, random_state=0)
x_sitdown, y_sitdown = shuffle(x_sitdown, y_sitdown, random_state=0)
x_pickup, y_pickup = shuffle(x_pickup, y_pickup, random_state=0)


# k_fold，好像这里用了K值交叉验证，每一次的都要训练2000代，那么总共要训练20000代
kk = 10

# Launch the graph  用于运行TensorFlow操作的类.
with tf.Session() as sess:
    for i in range(kk):  # 这就是k折交叉验证循环网络

        # Initialization
        train_loss = []
        train_acc = []
        validation_loss = []
        validation_acc = []

        # Roll the data   np.roll沿着给定轴滚动数组元素。超出最后位置的元素将会滚动到第一个位置。
        x_env = np.roll(x_env, int(len(x_env) / kk), axis=0)  # 让数据进行滚动输入
        y_env = np.roll(y_env, int(len(y_env) / kk), axis=0)
        x_fall = np.roll(x_fall, int(len(x_fall) / kk), axis=0)
        y_fall = np.roll(y_fall, int(len(y_fall) / kk), axis=0)
        x_squat = np.roll(x_squat, int(len(x_squat) / kk), axis=0)
        y_squat = np.roll(y_squat, int(len(y_squat) / kk), axis=0)
        x_walk = np.roll(x_walk, int(len(x_walk) / kk), axis=0)
        y_walk = np.roll(y_walk, int(len(y_walk) / kk), axis=0)
        x_sitdown = np.roll(x_sitdown, int(len(x_sitdown) / kk), axis=0)
        y_sitdown = np.roll(y_sitdown, int(len(y_sitdown) / kk), axis=0)
        x_pickup = np.roll(x_pickup, int(len(x_pickup) / kk), axis=0)
        y_pickup = np.roll(y_pickup, int(len(y_pickup) / kk), axis=0)

        # data separation  np.r_：是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
        # 这里连接的行数为它的总行数除以kk,每次拿出前面的这么多行进行训练'env','fall','squat','walk','sitdown','pickup'
        wifi_x_train = np.r_[x_env[int(len(x_env) / kk):], x_fall[int(len(x_fall) / kk):], x_squat[int(len(x_squat) / kk):], \
                        x_walk[int(len(x_walk) / kk):], x_sitdown[int(len(x_sitdown) / kk):], x_pickup[int(len(x_pickup) / kk):]]

        wifi_y_train = np.r_[y_env[int(len(y_env) / kk):], y_fall[int(len(y_fall) / kk):], y_squat[int(len(y_squat) / kk):], \
                        y_walk[int(len(y_walk) / kk):], y_sitdown[int(len(y_sitdown) / kk):], y_pickup[int(len(y_pickup) / kk):]]

        wifi_y_train = wifi_y_train[:,1:]
                                # 'env','fall','squat','walk','sitdown','pickup'
        wifi_x_validation = np.r_[x_env[:int(len(x_env) / kk)], x_fall[:int(len(x_fall) / kk)], x_squat[:int(len(x_squat) / kk)], \
                        x_walk[:int(len(x_walk) / kk)], x_sitdown[:int(len(x_sitdown) / kk)], x_pickup[:int(len(x_pickup) / kk)]]

        wifi_y_validation = np.r_[y_env[:int(len(y_env) / kk)], y_fall[:int(len(y_fall) / kk)], y_squat[:int(len(y_squat) / kk)], \
                        y_walk[:int(len(y_walk) / kk)], y_sitdown[:int(len(y_sitdown) / kk)], y_pickup[:int(len(y_pickup) / kk)]]

        wifi_y_validation = wifi_y_validation[:,1:]

        # data set
        wifi_train = DataSet(wifi_x_train, wifi_y_train)
        wifi_validation = DataSet(wifi_x_validation, wifi_y_validation)
        print(wifi_x_train.shape, wifi_y_train.shape, wifi_x_validation.shape, wifi_y_validation.shape)
        saver = tf.train.Saver()  # 这个不是保存模型，那不是每次都会进行保存吗？只会保存10个模型
        sess.run(init)
        step = 1

        # Keep training until reach max iterations  迭代训练
        while step < training_iters:
            batch_x, batch_y = wifi_train.next_batch(batch_size)
            x_vali = wifi_validation.images[:]
            y_vali = wifi_validation.labels[:]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            x_vali = x_vali.reshape((-1, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            acc_vali = sess.run(accuracy, feed_dict={x: x_vali, y: y_vali})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            loss_vali = sess.run(cost, feed_dict={x: x_vali, y: y_vali})

            # Store the accuracy and loss
            train_acc.append(acc)
            train_loss.append(loss)
            validation_acc.append(acc_vali)
            validation_loss.append(loss_vali)

            if step % display_step == 0:
                print("Iter " + str(step) + ", Minibatch Training  Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc) + ", Minibatch Validation  Loss= " + \
                    "{:.6f}".format(loss_vali) + ", Validation Accuracy= " + \
                    "{:.5f}".format(acc_vali) )
            step += 1

        #Calculate the confusion_matrix
        cvscores.append(acc_vali * 100)
        y_p = tf.argmax(pred, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_vali, y: y_vali})
        y_true = np.argmax(y_vali,1)
        print(sk.metrics.confusion_matrix(y_true, y_pred))
        confusion = sk.metrics.confusion_matrix(y_true, y_pred)
        confusion_sum = confusion_sum + confusion

        # Save the Accuracy curve
        fig = plt.figure(2 * i - 1)
        plt.plot(train_acc)
        plt.plot(validation_acc)
        plt.xlabel("n_epoch")
        plt.ylabel("Accuracy")
        plt.legend(["train_acc","validation_acc"],loc=4)
        plt.ylim([0,1])
        plt.savefig((output_folder + "Accuracy_" + str(i) + ".png"), dpi=150)

        # Save the Loss curve
        fig = plt.figure(2 * i)
        plt.plot(train_loss)
        plt.plot(validation_loss)
        plt.xlabel("n_epoch")
        plt.ylabel("Loss")
        plt.legend(["train_loss","validation_loss"],loc=1)
        plt.ylim([0,2])
        plt.savefig((output_folder + "Loss_" + str(i) + ".png"), dpi=150)

    print("Optimization Finished!")
    print("%.1f%% (+/- %.1f%%)" % (np.mean(cvscores), np.std(cvscores)))
    saver.save(sess, output_folder + "model.ckpt")

    #Save the confusion_matrix
    np.savetxt(output_folder + "confusion_matrix.txt", confusion_sum, delimiter=",", fmt='%d')
    np.savetxt(output_folder + "accuracy.txt", (np.mean(cvscores), np.std(cvscores)), delimiter=".", fmt='%.1f')
