#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

"""\nfind the highest child mortality rate in 1990 """
highest_1990_child_mo = float("-inf")
highest_1990_child_mo_country_sequence = 0
country_count = -1
for data in values[:, 0]:
    country_count += 1
    if highest_1990_child_mo < data:
        highest_1990_child_mo = data
        print "the highest 1990 updated!"
        highest_1990_child_mo_country_sequence = country_count

print ("the highest_1990_child_mo_country is %s with sequence %d and country %s" % (
    highest_1990_child_mo[0, 0], highest_1990_child_mo_country_sequence,
    countries[highest_1990_child_mo_country_sequence]))

"""\nfind the highest child mortality rate in 2011 """
highest_2011_child_mo = float("-inf")
highest_2011_child_mo_country_sequence = 0
country_count = -1
for data in values[:, 1]:
    country_count += 1
    if highest_2011_child_mo < data:
        highest_2011_child_mo = data
        print "the highest 2011 updated!"
        highest_2011_child_mo_country_sequence = country_count

print ("the highest_2011_child_mo_country is %s with sequence %d and country %s" % (
    highest_2011_child_mo[0, 0], highest_2011_child_mo_country_sequence,
    countries[highest_2011_child_mo_country_sequence]))

targets = values[:, 1]
x = values[:, 7:40]
x_n = a1.normalize_data(x)
# x = a1.normalize_data(x)

N_TRAIN = 100

x_train = x[0:N_TRAIN, :]
x_train_normalize = x_n[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
x_test_normalize = x_n[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

x_train_array = np.array(x_train)
x_train_array_normalize = np.array(x_train_normalize)
t_train_array = np.array(t_train)
x_test_array = np.array(x_test)
x_test_array_normalize = np.array(x_test_normalize)
t_test_array = np.array(t_test)

# # X is the independent variable (bivariate in this case)
# X = np.array([[0.44, 0.68, 0.21], [0.99, 0.23, 0.11]])
# # vector is the dependent data
# vector = [109.85, 155.72]
# # predict is an independent variable for which we'd like to predict the value
# predict = [0.49, 0.18, 0.3]

train_error_record = []
train_error_record_normalize = []
test_error_record = []
test_error_record_normalize = []


# for degree_num in range(1, 7):
#     poly = PolynomialFeatures(degree=degree_num)
#     train_ = poly.fit_transform(x_train_array)
#     predict_ = poly.fit_transform(x_test_array)
#     clf = linear_model.LinearRegression()
#     clf.fit(train_, t_train_array)
#
#     predict_result = np.array(clf.predict(predict_))
#     predict_train = np.array(clf.predict(train_))
#     print predict_train.shape
#     print t_train_array.shape
#     # print predict_result[:, 0]
#     # RMS_train_ = (0.5 * np.power((predict_train - t_train_array), 2)).sum()
#     RMS_train_ = np.sqrt(
#         ((np.power((predict_train - t_train_array), 2)).sum()) / predict_train.size)
#     print ("the training error is: %f" % RMS_train_)
#     train_error_record.append(RMS_train_)
#
#     # RMS_test_ = (0.5 * np.power((predict_result - t_test_array), 2)).sum()
#
#     RMS_test_ = np.sqrt(
#         ((np.power((predict_result - t_test_array), 2)).sum()) / predict_result.size)
#     print ("the testing error is: %f" % RMS_test_)
#     test_error_record.append(RMS_test_)

def ploy_regress_function(parameter_input, x_data):
    return x_data * parameter_input


'''combine the training data'''
for degree_num in range(1, 7):

    print('\n##Now the degree is %d##' % degree_num)

    x_train_array_degree = x_train_array
    x_test_array_degree = x_test_array
    x_train_array_degree_n = x_train_array_normalize
    x_test_array_degree_n = x_test_array_normalize

    if degree_num != 1:
        for i in range(2, degree_num + 1):
            x_train_array_degree = np.concatenate((np.power(x_train_array, i), x_train_array_degree), axis=1)
            x_test_array_degree = np.concatenate((np.power(x_test_array, i), x_test_array_degree), axis=1)
            x_train_array_degree_n = np.concatenate((np.power(x_train_array_normalize, i), x_train_array_degree_n),
                                                    axis=1)
            x_test_array_degree_n = np.concatenate((np.power(x_test_array_normalize, i), x_test_array_degree_n), axis=1)

    x_train_array_degree = np.concatenate((x_train_array_degree, np.transpose(np.matrix(np.ones(100)))), axis=1)
    x_test_array_degree = np.concatenate((x_test_array_degree, np.transpose(np.matrix(np.ones(95)))), axis=1)
    x_train_array_degree_n = np.concatenate((x_train_array_degree_n, np.transpose(np.matrix(np.ones(100)))), axis=1)
    x_test_array_degree_n = np.concatenate((x_test_array_degree_n, np.transpose(np.matrix(np.ones(95)))), axis=1)

    print(x_train_array_degree.shape)
    print(x_test_array_degree.shape)
    print(x_train_array_degree_n.shape)
    print(x_test_array_degree_n.shape)

    '''solve the parameter'''
    matrix_pinv = np.linalg.pinv(x_train_array_degree)
    parameters = matrix_pinv * np.matrix(t_train_array)
    matrix_pinv_n = np.linalg.pinv(x_train_array_degree_n)
    parameters_n = matrix_pinv_n * np.matrix(t_train_array)
    print ('degree:', degree_num)
    print ('parameters shape is', parameters.shape)
    print ('parameters_n shape is', parameters_n.shape)
    # print ('parameters are %s', parameters)

    '''construct the function'''
    testing_prediction = ploy_regress_function(parameters, x_test_array_degree)
    training_prediction = ploy_regress_function(parameters, x_train_array_degree)
    testing_prediction_n = ploy_regress_function(parameters_n, x_test_array_degree_n)
    training_prediction_n = ploy_regress_function(parameters_n, x_train_array_degree_n)

    RMS_train = np.sqrt(
        ((np.power((training_prediction - t_train_array), 2)).sum()) / training_prediction.size)
    print ("the training error is: %f" % RMS_train)
    train_error_record.append(RMS_train)

    RMS_test = np.sqrt(
        ((np.power((testing_prediction - t_test_array), 2)).sum()) / testing_prediction.size)
    print ("the testing error is: %f" % RMS_test)
    test_error_record.append(RMS_test)

    RMS_train_n = np.sqrt(
        ((np.power((training_prediction_n - t_train_array), 2)).sum()) / training_prediction_n.size)
    print ("the training error is: %f" % RMS_train_n)
    train_error_record_normalize.append(RMS_train_n)

    RMS_test_n = np.sqrt(
        ((np.power((testing_prediction_n - t_test_array), 2)).sum()) / testing_prediction_n.size)
    print ("the testing error is: %f" % RMS_test_n)
    test_error_record_normalize.append(RMS_test_n)

# # print predict_result[:, 0]
#     # RMS_train_ = (0.5 * np.power((predict_train - t_train_array), 2)).sum()
#     RMS_train_ = np.sqrt(
#         ((np.power((predict_train - t_train_array), 2)).sum()) / predict_train.size)
#     print ("the training error is: %f" % RMS_train_)
#     train_error_record.append(RMS_train_)
#
#     # RMS_test_ = (0.5 * np.power((predict_result - t_test_array), 2)).sum()
#
#     RMS_test_ = np.sqrt(
#         ((np.power((predict_result - t_test_array), 2)).sum()) / predict_result.size)
#     print ("the testing error is: %f" % RMS_test_)
#     test_error_record.append(RMS_test_)
#
print(tuple(train_error_record))
print(tuple(test_error_record))
print(tuple(train_error_record_normalize))
print(tuple(test_error_record_normalize))

'''plot without normalization'''
plt.plot(range(1, 7), train_error_record, 'r-', range(1, 7), test_error_record, 'g-')
plt.ylabel('RMS')
plt.title('polynomial regression degree 1 to 6')
plt.xlabel('degree')
plt.legend(['RMS training error', 'RMS testing error'])
plt.show()

'''plot with normalization'''
plt.plot(range(1, 7), train_error_record_normalize, 'r-', range(1, 7), test_error_record_normalize, 'g-')
plt.ylabel('RMS')
plt.title('polynomial regression with normalization degree 1 to 6')
plt.xlabel('degree')
plt.legend(['RMS training error', 'RMS testing error'])
plt.show()
