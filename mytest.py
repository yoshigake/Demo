import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SemHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2

global layer_number
global lose


def init_data():
    train_set = pd.read_excel('D:/TRAINNCL/TRAIN2/DNN/dataset/train.xlsx')
    test_set = pd.read_excel('D:/TRAINNCL/TRAIN2/DNN/dataset/test.xlsx')
    train_set_character = np.array(train_set[["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]])
    test_set_character = np.array(test_set[["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]])
    train_set_output = np.array(train_set[["EH"]])
    test_set_output = np.array(test_set[["EH"]])

    row_train_set = train_set_character.shape[0]
    col_train_set = train_set_character.shape[1]
    row_test_set = test_set_character.shape[0]
    col_test_set = test_set_character.shape[1]

    train_character = np.zeros(col_train_set, row_train_set)
    test_character = np.zeros(col_test_set, row_test_set)

    for i in range(col_test_set):
        train_character[i, :] = (train_set_character.T[i, :] - np.min(train_set_character.T[i, :])) / (
                np.max(train_set_character.T[i, :]) - np.min(train_set_character.T[i, :]))
        test_character[i, :] = (test_set_character.T[i, :] - np.min(test_set_character.T[i, :])) / (
                np.max(test_set_character.T[i, :]) - np.min(test_set_character.T[i, :]))

    # transposed &
    train_output = np.zeros(1, row_train_set)
    test_output = np.zeros(1, row_test_set)
    train_output = (train_set_output.T[0, :] - np.min(train_set_output.T[0, :])) / (
            np.max(train_output.T[0, :]) - np.min(train_set_output.T[0, :]))
    test_output = (test_set_output.T[0, :] - np.min(test_output.T[0, :])) / (
            np.max(test_set_output.T[0, :]) - np.min(test_set_output.T[0, :]))
    return train_character, test_character, train_output, test_output


def relu(line):
    active = np.maximum(0, line)
    return active


def init_parameter(X, Y):
    dim = [20, 30, 20, 1]
    layer_number = len(dim)
    param = {}
    param["w1"] = np.random.rand(dim[0], X.shape[0]) / np.sqrt(X.shape[0])
    param["b1"] = np.random.rand(dim[0], 1) * 0.1
    for i in range(1, layer_number):
        param["w" + str(i + 1)] = np.random.rand(dim[i], dim[i - 1]) / np.sqrt(X.shape[0])
        param["b" + str(i + 1)] = np.random.rand(dim[i], 1) * 0.1
    return param


def forward_propagation(param, X):
    varia = {}
    varia["line1"] = np.dot(param["w1"], X) + param["b1"]
    varia["active1"] = relu(varia["line1"])
    for i in range(1, layer_number):
        varia["line" + str(i + 1)] = np.dot(param["w" + str(i + 1)], varia["line" + str(i)]) + param["b" + str(i + 1)]
        varia["active" + str(i + 1)] = relu(varia["line" + str(i + 1)])
    return varia


def back_propagation(param, varia, X, Y):
    num = X.shape[1]
    grad = {}
    grad["dl" + str(layer_number)] = (varia["active" + str(layer_number)] - Y) / num
    for i in range(layer_number, 1, -1):
        grad["dw" + str(i)] = np.dot(grad["dl" + str(i)], varia["A" + str(i - 1)].T)
        grad["db" + str(i)] = np.sum(grad["dl" + str(i)])
        grad["dl" + str(i - 1)] = np.dot(varia["w" + str(i)], grad["dw" + str(i)].T)
    grad["dw1"] = np.dot(grad["dl1"], X.T)
    grad["db1"] = np.sum(grad["dl1"], axis=1, keepdims=True)
    return grad


def loss_function(varia, X, Y):
    loss = np.sum((varia["A" + str(layer_number)] - Y) ** 2) / (2 * Y.shape[1])
    return loss


def grad_down(param, iteration_coefficient, grad):
    for i in range(1, layer_number + 1):
        param["w" + str(i)] = param["w" + str(i)] - iteration_coefficient * grad["dw" + str(i)]
        param["b" + str(i)] = param["b" + str(i)] - iteration_coefficient * grad["db" + str(i)]
    return param


def my_dnn_train(X, Y, iteration_times, iteration_coefficient):
    param = init_parameter(X, Y)
    for i in range(1, iteration_times + 1):
        varia = forward_propagation(param, X)
        grad = back_propagation(param, varia, X, Y)
        param = grad_down(param, iteration_coefficient, grad)
        loss = loss_function(varia, X, Y)
        if i % 100 == 0 and i != 0:
            print("第{}次迭代损失函数的值为：{}".format(i, loss))
            loss_function.append(loss)
    return param


def my_dnn_predict(X_train, Y_train, X_test, Y_test, param):
    varia_train = forward_propagation(param, X_train)
    Y_train_predict = varia_train["A" + str(layer_number)]
    varia_test = forward_propagation(param, X_test)
    Y_test_predict = varia_test["A" + str(layer_number)]
    print("训练集准确率为：{}%", format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("测试集准确率为：{}%", format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))
    return Y_train_predict, Y_test_predict


iteration_times = 8000
iteration_coefficient = 0.01

train_character, test_character, train_output, test_output = init_data()
param = init_parameter(train_character, train_output)
varia = forward_propagation(param, train_character)
grad = back_propagation(param, varia, train_character, train_output)
parameter = grad_down(param, iteration_coefficient)
parameter = my_dnn_train(train_character, train_output, iteration_times, iteration_coefficient)
Y_train_predict, Y_test_predict = my_dnn_predict(train_character, train_output, test_character, test_output)
