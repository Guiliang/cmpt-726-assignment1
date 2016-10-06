#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

RMS_train_record = []
RMS_test_record = []


def fit_function(parameter_input, x_data):
    return np.transpose(parameter_input.dot(x_data))


for feature_count in range(7, 15):
    print ("\n feature : %d" % (feature_count + 1))
    targets = values[:, 1]
    data = values[:, feature_count]
    # x = a1.normalize_data(x)
    N_TRAIN = 100
    data_train = data[0:N_TRAIN, :]
    data_test = data[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    data_train_degree_3 = (np.power(data_train, 3)).A1.reshape(data_train.size)
    data_train_degree_2 = (np.power(data_train, 2)).A1.reshape(data_train.size)
    data_train_degree_1 = (np.power(data_train, 1)).A1.reshape(data_train.size)
    matrix_combine_train = np.matrix(
        [data_train_degree_3, data_train_degree_2, data_train_degree_1, np.ones(data_train_degree_3.size)])
    print("matrix combine train shape is", matrix_combine_train.shape)

    data_test_degree_3 = (np.power(data_test, 3)).A1.reshape(data_test.size)
    data_test_degree_2 = (np.power(data_test, 2)).A1.reshape(data_test.size)
    data_test_degree_1 = (np.power(data_test, 1)).A1.reshape(data_test.size)
    matrix_combine_test = np.matrix(
        [data_test_degree_3, data_test_degree_2, data_test_degree_1, np.ones(data_test_degree_3.size)])
    print("matrix combine test shape is", matrix_combine_test.shape)

    matrix_pinv = np.linalg.pinv(matrix_combine_train)
    parameters = np.matrix(t_train.A1.reshape(t_train.size)) * matrix_pinv

    print("the first training data is: %s" % features[feature_count])

    print("the fitting parameter is: %s" % parameters)
    # difference = fit_function(parameters, matrix_combine_test) - t_test
    RMS_train = np.sqrt(
        ((np.power((fit_function(parameters, matrix_combine_train) - t_train), 2)).sum()) / t_train.size)
    # RMS_train = (0.5 * np.power((fit_function(data_train) - data_train), 2)).sum()
    RMS_train_record.append(int(RMS_train))
    print ("the training error is: %f" % RMS_train)
    RMS_test = np.sqrt(
        ((np.power((fit_function(parameters, matrix_combine_test) - t_test), 2)).sum()) / t_test.size)
    # RMS_test = (0.5 * np.power((fit_function(data_test) - t_test), 2)).sum()
    RMS_test_record.append(int(RMS_test))
    print ("the testing error is: %f" % RMS_test)

print(tuple(RMS_train_record))
print(tuple(RMS_test_record))
N = len(RMS_train_record)
ind = np.arange(8, N + 8)
# width = 1 / 1.5
plt.bar(ind, RMS_train_record, 0.35, color='r')
plt.ylabel('RMS')
plt.title('Fit with polynomials, 1 feature, no regularization, training error')
plt.xlabel('feature')
plt.show()

plt.bar(ind, RMS_test_record, 0.35, color='r')
plt.ylabel('RMS')
plt.title('Fit with polynomials, 1 feature, no regularization, testing error')
plt.xlabel('feature')
plt.show()
