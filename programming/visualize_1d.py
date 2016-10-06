#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()


def fit_function(parameter_input, x_data):
    return np.transpose(parameter_input.dot(x_data))


targets = values[:, 1]
x = values[:, 7:]
# x = a1.normalize_data(x)
for features_count in range(3, 6):
    N_TRAIN = 100
    # Select a single feature.
    x_train = x[0:N_TRAIN, features_count]
    x_train_sorted = np.sort(x[0:N_TRAIN, features_count], axis=0)
    t_train = targets[0:N_TRAIN]
    x_test = x[N_TRAIN:, features_count]
    t_test = targets[N_TRAIN:, :]
    print(x_test.size)
    print(t_test.size)

    data_train_degree_3 = (np.power(x_train, 3)).A1.reshape(x_train.size)
    data_train_degree_2 = (np.power(x_train, 2)).A1.reshape(x_train.size)
    data_train_degree_1 = (np.power(x_train, 1)).A1.reshape(x_train.size)
    matrix_combine_train = np.matrix(
        [data_train_degree_3, data_train_degree_2, data_train_degree_1, np.ones(data_train_degree_3.size)])

    data_test_degree_3 = (np.power(x_test, 3)).A1.reshape(x_test.size)
    data_test_degree_2 = (np.power(x_test, 2)).A1.reshape(x_test.size)
    data_test_degree_1 = (np.power(x_test, 1)).A1.reshape(x_test.size)
    matrix_combine_test = np.matrix(
        [data_test_degree_3, data_test_degree_2, data_test_degree_1, np.ones(data_test_degree_1.size)])

    data_train_degree_3_sort = (np.power(x_train_sorted, 3)).A1.reshape(x_train_sorted.size)
    data_train_degree_2_sort = (np.power(x_train_sorted, 2)).A1.reshape(x_train_sorted.size)
    data_train_degree_1_sort = (np.power(x_train_sorted, 1)).A1.reshape(x_train_sorted.size)
    matrix_combine_train_sort = np.matrix(
        [data_train_degree_3_sort, data_train_degree_2_sort, data_train_degree_1_sort, np.ones(data_train_degree_3.size)])

    matrix_pinv = np.linalg.pinv(matrix_combine_train)
    parameters = np.matrix(t_train.A1.reshape(t_train.size)) * matrix_pinv

    x_train_reshaped = x_train.A1.reshape(x_train.size)
    t_train_reshaped = t_train.A1.reshape(t_train.size)
    x_test_reshaped = x_test.A1.reshape(x_test.size)
    t_test_reshaped = t_test.A1.reshape(t_test.size)

    print(fit_function(parameters, matrix_combine_train).A1)

    plt.plot(sorted(x_train_reshaped), fit_function(parameters, matrix_combine_train_sort).A1, '-', x_train_reshaped,
             t_train_reshaped, '*',
             x_test_reshaped, t_test_reshaped, '.')

    plt.xlabel(features[features_count+7])
    plt.title(
        'feature %d,draw the training data points, learned polynomial, and test data points' % (features_count + 8))
    plt.ylabel('Under-5 mortality rate (U5MR) 2011')
    plt.legend(['learned polynomial', 'Training data', 'Testing data'])

    # # Plot a curve showing learned function.
    # # Use linspace to get a set of samples on which to evaluate
    # x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    #
    # # TO DO:: Put your regression estimate here in place of x_ev.
    # # Evaluate regression on the linspace samples.
    # y_ev = np.random.random_sample(x_ev.shape)
    # y_ev = 100 * np.sin(x_ev)

    # plt.plot(x_ev, y_ev, 'r.-')
    # plt.plot(x_train.A1.reshape(x_train.size), t_train.A1.reshape(t_train.size), 'g.')
    # plt.title('A visualization of a regression estimate using random outputs')
    plt.show()
