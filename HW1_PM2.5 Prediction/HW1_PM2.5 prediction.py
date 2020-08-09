import math
import pandas as pd
import numpy as np
import csv
# load dataset
train_data = pd.read_csv("./dataset/train.csv",
                         usecols=range(3, 27), encoding='big5')
test_data = pd.read_csv("./dataset/test.csv", usecols=range(
    2, 11), header=None, encoding='big5')

# preprocess(1)
# NR replace to '0.0'
train_data = train_data.replace(['NR'], [0.0])  # NR replace to '0.0'
test_data = test_data.replace(['NR'], [0.0])
# pd to array
train_array = np.array(train_data).astype(float)
test_array = np.array(test_data).astype(float)

# preprocess(2)
# train
x_list = []
train_y = np.zeros([5652, 1], dtype=float)
# split to each month (18*480) *12
month_train = np.empty([216, 480])  # (18*12,24*20)
month_test = np.empty([216, 180])  # (18*12,9*20)
for month in range(12):
    sample = np.empty([18, 480])
    sample2 = np.empty([18, 180])
    for day in range(20):
        sample[:, day*24:(day+1)*24] = train_array[18 *
                                                   (20*month+day):18*(20*month+day+1), :]
        sample2[:, day*9:(day+1)*9] = test_array[18 *
                                                 (20*month+day):18*(20*month+day+1), :]
    month_train[month*18:(month+1)*18, :] = sample
    month_test[month*18:(month+1)*18, :] = sample2

month_train_array = np.array(month_train)
month_test_array = np.array(month_test)
# adjust to (471*12,18*9) = (5652,162)
for m in range(12):
    for j in range(480-9):
        mat = month_train_array[m*18:(m+1)*18, j:j+9].flatten()
        x_list.append(mat)
        train_y[m*471 + j] = month_train_array[m*18+9, j+9]  # PM2.5 在第9 row

train_x = np.array(x_list)  # train data (471*12,18*9) (5652,162)

# test
test = []
for m in range(12):
    for j in range(20):
        mat = month_test_array[m*18:(m+1)*18, j:j+9].flatten()
        test.append(mat)
test = np.array(test)  # (240,162)

# Normalize
# train
mean_x = np.mean(train_x, axis=0)  # axis = 0 :橫軸
std_x = np.std(train_x, axis=0)
for i in range(len(train_x)):  # 12*471
    for j in range(len(train_x[0])):  # 18*9
        if std_x[j] != 0:
            train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]

# test
for i in range(len(test)):
    for j in range(len(test[0])):
        if std_x[j] != 0:
            test[i][j] = (test[i][j] - mean_x[j]) / std_x[j]
# add ans label
test = np.concatenate((np.ones([240, 1]), test), axis=1).astype(float)

# 切 train and validation data (8:2)
x_train_set = train_x[: math.floor(len(train_x) * 0.8), :]
y_train_set = train_y[: math.floor(len(train_y) * 0.8)]
x_validation = train_x[math.floor(len(train_x) * 0.8):, :]
y_validation = train_y[math.floor(len(train_y) * 0.8)]

# adagrad algorithm and save model
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12*471, 1]), train_x), axis=1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum((np.dot(x, w) - train_y)**2) / 471 / 12)  # RMSE
    if(t % 100 == 0):
        print(str(t) + ":" + str(loss))
    p = 2 * np.dot(x.transpose(), np.dot(x, w))
    adagrad += p**2
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - train_y)  # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)

# predit
w = np.load('weight.npy')
ans = np.dot(test, w)

# output ans file
with open('ans.csv', mode='w', newline='') as ans_file:
    ans_writer = csv.writer(ans_file)
    header = ['id', 'value']
    ans_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans[i][0]]
        ans_writer.writerow(row)
