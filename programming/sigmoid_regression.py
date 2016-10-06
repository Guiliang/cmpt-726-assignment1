import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, u, s):
    """
    :param x: training data
    :param u: parameter 1
    :param s: parameter 2
    :return: sigmoid function
    """
    return 1 / (1 + np.exp((u - x) / s))


(countries, features, values) = a1.load_unicef_data()
targets = values[:, 1]
data = values[:, 10]  # feature 11

# split data
N_TRAIN = 100
data_train = data[0:N_TRAIN, :]
data_test = data[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# transfer to Array and reshape
x_train_reshaped = data_train.A1.reshape(data_train.size)
t_train_reshaped = t_train.A1.reshape(t_train.size)
x_test_reshaped = data_test.A1.reshape(data_test.size)
t_test_reshaped = t_test.A1.reshape(t_test.size)

# parameter
u1 = 100
s1 = 2000

u2 = 10000
s2 = 0

'''transfer to sigmoid'''
# print (type(x_train_reshaped))
# print ([u] * data_train.size)
x_train_sigmoid_1 = sigmoid(x_train_reshaped, u1, s1)
x_train_sigmoid_2 = sigmoid(x_train_reshaped, u2, s2)
x_test_sigmoid_1 = sigmoid(x_test_reshaped, u1, s1)
x_test_sigmoid_2 = sigmoid(x_test_reshaped, u2, s2)

'''calculate the parameter '''
matrix_combine = np.matrix([x_train_sigmoid_1, x_train_sigmoid_2, np.ones(x_train_sigmoid_1.size)])
matrix_pinv = np.linalg.pinv(matrix_combine)
parameters = np.matrix(t_train_reshaped) * matrix_pinv
print ("the parameter is:%s" % parameters)

'''construct the regression function'''
def sigmoid_regress(x):
    return parameters[0, 0] * sigmoid(x, u1, s1) + parameters[0, 1] * sigmoid(x, u2, s2) + parameters[0, 2]


# print (sigmoid_regress(np.array(1)))

RMS_train = np.sqrt(
    ((np.power((sigmoid_regress(x_train_reshaped) - t_train_reshaped), 2)).sum()) / t_train_reshaped.size)
print ("the training error is: %f" % RMS_train)

RMS_test = np.sqrt(
    ((np.power((sigmoid_regress(x_test_reshaped) - t_test_reshaped), 2)).sum()) / x_test_reshaped.size)
print ("the testing error is: %f" % RMS_test)

plt.plot(np.sort(x_train_reshaped), sigmoid_regress(np.sort(x_train_reshaped)), '-')
plt.ylabel('Under-5 mortality rate (U5MR) 2011')
plt.title('sigmoid regression method 1')
plt.xlabel('GNI per capita')
plt.show()
#
# print(sigmoid_regress(np.array(72145)))


# '''Method 2'''
# x_train_combine = np.transpose(np.array([x_train_sigmoid_1, x_train_sigmoid_2, np.ones(x_train_sigmoid_1.size)]))
# # Fit regression model
# results = sm.OLS(t_train_reshaped, x_train_combine).fit()
# parameters_ = results.params
#
# print parameters_
#
#
# def sigmoid_regress_(x):
#     return parameters_[0] * sigmoid(x, u1, s1) + parameters_[1] * sigmoid(x, u2, s2) + parameters[0, 2]
#
#
# RMS_train_ = np.sqrt(
#     ((np.power((sigmoid_regress_(x_train_reshaped) - t_train_reshaped), 2)).sum()) / t_train_reshaped.size)
# print ("the training error is: %f" % RMS_train_)
#
# RMS_test_ = np.sqrt(
#     ((np.power((sigmoid_regress_(x_test_reshaped) - t_test_reshaped), 2)).sum()) / x_test_reshaped.size)
# print ("the testing error is: %f" % RMS_test_)
#
# plt.plot(np.sort(x_train_reshaped), sigmoid_regress_(np.sort(x_train_reshaped)), '-')
# plt.ylabel('Under-5 mortality rate (U5MR) 2011')
# plt.title('sigmoid regression method 2')
# plt.xlabel('GNI per capita')
# plt.show()
