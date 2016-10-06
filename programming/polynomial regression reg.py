#!/usr/bin/env python

import assignment1 as a1
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

N_TRAIN = 100
values_training_data_no = a1.normalize_data(values)
values_training_data_no = values_training_data_no[0:N_TRAIN, :]
values_training_data = values[0:N_TRAIN, :]
targets = values[0:N_TRAIN, 1]


def ploy_regress_function(parameter_input, x_data):
    return x_data.dot(parameter_input)


def mean(numbers):
    return float(sum(numbers)) / len(numbers)


reg_lambda_test_error = []
'''test lambda'''
reg_lambda = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
for reg_lambda_num in range(0, 8):

    test_error_record_cross_normalize = []

    '''cross validation'''
    for validate_num in range(0, 10):
        print('cross:', validate_num)
        cross_test_value_no = values_training_data_no[validate_num * 10:validate_num * 10 + 10, :]
        cross_test_value = values_training_data[validate_num * 10:validate_num * 10 + 10, :]
        cross_test_targets = targets[validate_num * 10:validate_num * 10 + 10, :]
        if validate_num == 0:
            cross_train_value_no = values_training_data_no[10:, :]
            cross_train_value = values_training_data[10:, :]
            cross_train_targets = targets[10:, :]
        elif validate_num == 9:
            cross_train_value_no = values_training_data_no[:90, :]
            cross_train_value = values_training_data[:90, :]
            cross_train_targets = targets[:90, :]
        else:
            cross_train_part1_no = values_training_data_no[:validate_num * 10, :]
            cross_train_part2_no = values_training_data_no[(validate_num + 1) * 10:, :]
            cross_train_value_no = np.concatenate((cross_train_part2_no, cross_train_part1_no), axis=0)
            cross_train_part1 = values_training_data[:validate_num * 10, :]
            cross_train_part2 = values_training_data[(validate_num + 1) * 10:, :]
            cross_train_value = np.concatenate((cross_train_part2, cross_train_part1), axis=0)
            cross_train_targets_part1 = targets[:validate_num * 10, :]
            cross_train_targets_part2 = targets[(validate_num + 1) * 10:, :]
            cross_train_targets = np.concatenate((cross_train_targets_part2, cross_train_targets_part1), axis=0)
        print(cross_test_value_no.shape)
        print(cross_train_value_no.shape)

        x_train_cross = cross_train_value_no[:, 7:40]
        # x_train_normalize_cross = a1.normalize_data(x_train_cross)
        t_train_cross = cross_train_targets
        x_test_cross = cross_test_value_no[:, 7:40]
        # x_test_normalize_cross = a1.normalize_data(x_test_cross)
        t_test_cross = cross_test_targets

        x_train_array_normalize_cross = np.array(x_train_cross)
        t_train_array_cross = np.array(t_train_cross)
        x_test_array_normalize_cross = np.array(x_test_cross)
        t_test_array_cross = np.array(t_test_cross)

        x_train_array_degree_2_cross = 1.0 * np.concatenate(
            (x_train_array_normalize_cross, np.power(x_train_array_normalize_cross, 2),
             np.transpose(np.matrix(np.ones(90)))), axis=1)

        x_test_array_degree_2_cross = 1.0 * np.concatenate(
            (x_test_array_normalize_cross, np.power(x_test_array_normalize_cross, 2),
             np.transpose(np.matrix(np.ones(10)))), axis=1)

        print(x_train_array_degree_2_cross.shape)
        '''solve the parameter with lambda'''
        tempt1 = np.transpose(x_train_array_degree_2_cross).dot(x_train_array_degree_2_cross)
        print('shape tempt1', tempt1.shape)
        tempt2 = inv(tempt1 + reg_lambda[reg_lambda_num] * np.identity(tempt1.shape[0]))
        print('shape tempt2', tempt2.shape)
        parameters = tempt2.dot(np.transpose(x_train_array_degree_2_cross)).dot(t_train_array_cross)

        print(parameters)

        '''construct the function'''
        testing_prediction = ploy_regress_function(parameters, x_test_array_degree_2_cross)
        print('lambda is:', reg_lambda[reg_lambda_num])
        parameters_rms = reg_lambda[reg_lambda_num] * (np.power(parameters, 2).sum())
        print('parameters_rms:', parameters_rms)
        error_fun = np.power((testing_prediction - t_test_array_cross), 2).sum()
        print('error_fun:', error_fun)
        RMS_test = np.sqrt(error_fun / 10.0)
        print ("the testing error is: %f" % RMS_test)
        test_error_record_cross_normalize.append(RMS_test)
        print ('\n')

    mean_test_error = mean(test_error_record_cross_normalize)
    reg_lambda_test_error.append(mean_test_error)
    print ("the mean testing error is: %f" % mean_test_error)

print(reg_lambda_test_error)

plt.semilogx(reg_lambda, reg_lambda_test_error)
plt.ylabel('validation set error')
plt.title('Regularized Polynomial Regression')
plt.xlabel('polynomial using lambda')
plt.show()
