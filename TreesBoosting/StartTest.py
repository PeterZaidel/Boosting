# Import the necessary modules and libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import california_housing

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from RegressionTree import MyTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime
from GradientBoostTree import  GradientBoostTree
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
import pickle
from sklearn.datasets import load_svmlight_file
from Blending import  Blending

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

def calc_time(fn, *args):
    dt0 = datetime.now()
    res = fn(*args)
    dt1 = datetime.now()
    return res, dt1-dt0


test_trees_pp = False
test_trees_sinus = False
test_trees_house = False
test_boosting_house = False
test_boosting_real = False

test_blending_real = True

find_blending_real_best_params = False


if test_trees_pp:
    print('-------- TREEES TESTING ----------')
    X, y = pickle.load(open('./test.p', 'rb')).values()

    sk_regr = DecisionTreeRegressor(max_depth=1, random_state=42)
    my_regr = MyTreeRegressor(max_depth=1, random_state=42)
    _, sk_regr_train_time = calc_time(sk_regr.fit, X, y)
    sk_regr_pred, sk_regr_test_time = calc_time(sk_regr.predict, X)

    _, my_regr_train_time = calc_time(my_regr.fit, X, y)
    my_regr_pred, my_regr_test_time = calc_time(my_regr.predict, X)

    my_regr_error = mean_squared_error(y, my_regr_pred)
    sk_regr_error = mean_squared_error(y, sk_regr_pred)

    print('my regression tree MSE: ', my_regr_error)
    print('sk regression tree MSE: ', sk_regr_error)
    print('sk regression tree train time: ', sk_regr_train_time)
    print('my regression train time: ', my_regr_train_time)
    print('my regression tree pred time: ', my_regr_test_time)
    print('sk regression tree pred time: ', sk_regr_test_time)

if test_trees_house:
    print('-------- TREEES TESTING ----------')
    dataset = california_housing.fetch_california_housing()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=43)

    sk_regr = DecisionTreeRegressor(max_depth=5, random_state=42)
    my_regr = MyTreeRegressor(max_depth=5, random_state=42)
    _, sk_regr_train_time = calc_time(sk_regr.fit, X_train, y_train)
    sk_regr_pred, sk_regr_test_time = calc_time(sk_regr.predict, X_test)

    _, my_regr_train_time = calc_time(my_regr.fit, X_train, y_train)
    my_regr_pred, my_regr_test_time = calc_time(my_regr.predict, X_test)

    my_regr_error = mean_squared_error(y_test, my_regr_pred)
    sk_regr_error = mean_squared_error(y_test, sk_regr_pred)

    print('my regression tree MSE: ', my_regr_error)
    print('sk regression tree MSE: ', sk_regr_error)
    print('sk regression tree train time: ', sk_regr_train_time)
    print('my regression train time: ', my_regr_train_time)
    print('my regression tree pred time: ', my_regr_test_time)
    print('sk regression tree pred time: ', sk_regr_test_time)


if test_trees_sinus:
    print('-------- TREEES TESTING ----------')

    rng = np.random.RandomState(1)
    X_train = np.sort(5 * rng.rand(80, 1), axis=0)
    y_train = np.sin(X_train).ravel()
    #y_train = y_train + 0.1 * (0.5 - rng.rand(y_train.shape[0]))

    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_test = np.sin(X_test).ravel()

    sk_regr = DecisionTreeRegressor(max_depth=100, random_state=42)
    my_regr = MyTreeRegressor(max_depth=100, random_state=42, presort=True)
    _, sk_regr_train_time =  calc_time(sk_regr.fit, X_train, y_train)
    sk_regr_pred, sk_regr_test_time = calc_time(sk_regr.predict, X_test)

    _, my_regr_train_time = calc_time( my_regr.fit, X_train, y_train)
    my_regr_pred, my_regr_test_time = calc_time(my_regr.predict, X_test)

    my_regr_error = mean_squared_error(y_test, my_regr_pred)
    sk_regr_error = mean_squared_error(y_test, sk_regr_pred)

    print('my regression tree MSE: ', my_regr_error)
    print('sk regression tree MSE: ', sk_regr_error)
    print('sk regression tree train time: ', sk_regr_train_time)
    print('my regression train time: ', my_regr_train_time)
    print('my regression tree pred time: ', my_regr_test_time)
    print('sk regression tree pred time: ', sk_regr_test_time)

    # Plot the results
    plt.figure()
    plt.scatter(X_train, y_train, s=20, edgecolor="black",
                c="darkorange", label="data")

    plt.plot(X_test, my_regr_pred, color="cornflowerblue",
             label="my regr max_depth = 5", linewidth=2)

    plt.plot(X_test, sk_regr_pred, color="yellowgreen", label="sklearn regr max_depth = 5", linewidth=2)

    # leaf_vals = np.array(sorted(my_regr.get_leaf_values()))
    #
    # plt.plot(np.linspace(0, 5, leaf_vals.shape[0]), leaf_vals  )

    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()



def test_boosting_trees(model, X_train, Y_train, X_test, Y_test, n_estimators_list):
    errors = []
    for est_num in n_estimators_list:
        model.n_estimators = est_num
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        error = mean_squared_error(Y_test, pred)
        errors.append(error)
    return errors

def test_sklearn_gbm(model, X_train, Y_train, X_test, Y_test, n_estimators_list):
    model.n_estimators = 1
    model.warm_start = True
    n_estimators_list = sorted(n_estimators_list)
    errors = []
    for est_num in n_estimators_list:
        model.n_estimators = est_num
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        error = mean_squared_error(Y_test, pred)
        errors.append(error)
    return errors

if test_boosting_house:



    dataset = california_housing.fetch_california_housing()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=43)

    my_boost = GradientBoostTree(verbose=True,
                                 base_estimator=DecisionTreeRegressor,
                                 max_depth=2,
                                 n_estimators=200,
                                 random_state=42,
                                 learning_rate=0.1,
                                 min_samples_split=2,
                                 subsample=1.0)

    sk_boost = GradientBoostingRegressor(verbose=True,
                                         criterion='mse',
                                         max_depth=2,
                                         n_estimators=200,
                                         random_state=42,
                                         learning_rate=0.1,
                                         min_samples_split=2,
                                         subsample=1.0)

    _, my_boost_train_time = calc_time(my_boost.fit, X_train, y_train)
    my_boost_pred, my_boost_test_time = calc_time(my_boost.predict, X_test, y_test)

    _, sk_boost_train_time = calc_time(sk_boost.fit, X_train, y_train)
    sk_boost_pred, sk_boost_test_time = calc_time(sk_boost.predict, X_test)

    my_boost_error =  mean_squared_error(y_test, my_boost_pred)
    sk_boost_error = mean_squared_error(y_test, sk_boost_pred)


    print('my boosting MSE: ', my_boost_error)
    print('sk boosting MSE: ', sk_boost_error)


    print('sk boosting train time: ', sk_boost_train_time)
    print('my boosting train time: ', my_boost_train_time)


    print('my boosting pred time: ', my_boost_test_time)
    print('sk boosting pred time: ', sk_boost_test_time)

    exit()
    #
    my_boost_pred, my_boost_errors,= my_boost.predict_over_estimators(X_test, y_test)
    my_boost_errors = np.array(my_boost_errors)
    #
    # my_boost_errors = test_boosting_trees(my_boost, X_train, y_train, X_test, y_test, np.arange(1, my_boost.n_estimators + 1))
    # my_boost_errors = np.array(my_boost_errors)

    print('my boost errors:', my_boost_errors)

    sk_boost_errors = test_boosting_trees(sk_boost, X_train, y_train, X_test, y_test, np.arange(1, my_boost.n_estimators + 1))
    sk_boost_errors = np.array(sk_boost_errors)

    print('sk boost errors:', sk_boost_errors)

    plt.plot(my_boost_errors[1:], color= 'green')
    plt.plot(sk_boost_errors[1:], color = 'blue')
    plt.plot(sk_boost_errors[1:] + 0.03 * sk_boost_errors[1:], color='grey', linestyle='--')
    plt.plot(sk_boost_errors[1:] - 0.03 * sk_boost_errors[1:], color = 'grey', linestyle='--')
    plt.show()


if test_boosting_real:

    def load_data(filename):
        columns = []
        rows = []
        vals = []

        target = []

        train_file = open(filename)

        for r_id, l in enumerate(train_file.readlines()):
            arr = l.split(' ')
            target.append(float(arr[0]))

            for d in arr[1:]:
                da = d.split(':')
                if len(da) != 2:
                    continue
                col_id = int(da[0])
                val = float(da[1])

                rows.append(r_id)
                columns.append(col_id)
                vals.append(val)

        train_file.close()

        X = np.zeros((max(rows) + 1, max(columns) + 1))
        for r, c, v in zip(rows, columns, vals):
            if v == 0:
                print('[{0}, {1}] = {2}'.format(r, c, v))
            X[r, c] = v

        return X, np.array(target)


    X_train, y_train = load_svmlight_file('../../Data/hw1/reg.train.txt')
    X_test, y_test = load_svmlight_file('../../Data/hw1/reg.test.txt')
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    sk_params = dict(
        verbose=True,
        criterion='mse',
        max_depth=2,
        n_estimators=400,
        random_state=42,
        learning_rate=0.1,
        min_samples_split=5,
        subsample=0.6)
    #
    # my_params=dict(
    #             verbose=False,
    #             base_estimator=DecisionTreeRegressor,
    #             max_depth = 2,
    #             n_estimators=200,
    #             random_state = 42,
    #             learning_rate=0.1,
    #             min_samples_split=3,
    #             subsample=1.0)

    #
    my_boost = GradientBoostTree(verbose=True,
                base_estimator=MyTreeRegressor,
                max_depth = 2,
                n_estimators=400,
                random_state = 3213,
                learning_rate=0.1,
                min_samples_split=5,
                subsample=0.6)

    sk_boost = GradientBoostingRegressor(**sk_params)

    print("Training My Boosting: \n")
    _, my_boost_train_time = calc_time(my_boost.fit, X_train, y_train)
    my_boost_pred, my_boost_test_time = calc_time(my_boost.predict, X_test, y_test)

    print('Training Sklearn Boosting: \n')
    _, sk_boost_train_time = calc_time(sk_boost.fit, X_train, y_train)
    sk_boost_pred, sk_boost_test_time = calc_time(sk_boost.predict, X_test)

    my_boost_error =  mean_squared_error(y_test, my_boost_pred)
    sk_boost_error = mean_squared_error(y_test, sk_boost_pred)


    print('my boosting MSE: ', my_boost_error)
    print('sk boosting MSE: ', sk_boost_error)


    print('sk boosting train time: ', sk_boost_train_time)
    print('my boosting train time: ', my_boost_train_time)


    print('my boosting pred time: ', my_boost_test_time)
    print('sk boosting pred time: ', sk_boost_test_time)

    # #
    my_boost_pred, my_boost_errors = my_boost.predict_over_estimators(X_test, y_test)
    my_boost_errors = np.array(my_boost_errors)
    #
    # my_boost_errors = test_boosting_trees(my_boost, X_train, y_train, X_test, y_test, np.arange(1, my_boost.n_estimators + 1))
    # my_boost_errors = np.array(my_boost_errors)

    print('my boost errors:', my_boost_errors)

    sk_boost = GradientBoostingRegressor(verbose=True,
        criterion='mse',
        max_depth=2,
        n_estimators=400,
        random_state=42,
        learning_rate=0.1,
        min_samples_split=5,
        subsample=0.6)
    sk_boost_errors = test_sklearn_gbm(sk_boost, X_train, y_train, X_test, y_test, np.arange(1, my_boost.n_estimators + 1))
    sk_boost_errors = np.array(sk_boost_errors)

    print('sk boost errors:', sk_boost_errors)

    plt.xlabel('n_estimators')
    plt.ylabel('MSE')
    plt.plot(my_boost_errors[1:], color= 'green', label = 'my boosting test error')
    plt.plot(sk_boost_errors[1:], color = 'blue', label = 'sklearn boosting test error')
    plt.plot(sk_boost_errors[1:] + 0.03 * sk_boost_errors[1:], color='grey', linestyle='--', label = 'sklearn error + 3%')
    plt.plot(sk_boost_errors[1:] - 0.03 * sk_boost_errors[1:], color = 'grey', linestyle='--', label = 'sklearn error - 3%')
    plt.legend()
    plt.show()


if test_blending_real:
    X_train, y_train = load_svmlight_file('../../Data/hw1/reg.train.txt')
    X_test, y_test = load_svmlight_file('../../Data/hw1/reg.test.txt')
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    sk_params = dict(
        )

    #
    my_boost = GradientBoostTree(verbose=False,
                                 base_estimator=MyTreeRegressor,
                                 max_depth=1,
                                 n_estimators=200, #5
                                 random_state=3213,
                                 learning_rate=0.1,
                                 min_samples_split=4,
                                 subsample=0.4)

    # sk_boost = GradientBoostingRegressor(
    #     verbose=False,
    #     criterion='mse',
    #     max_depth=2,
    #     n_estimators=400,
    #     random_state=42,
    #     learning_rate=0.3,
    #     min_samples_split=5,
    #     subsample=0.5)

    my_blending = Blending(GradientBoostTree(verbose=False,
                                 base_estimator=MyTreeRegressor,
                                 max_depth=1,
                                 n_estimators=90, #5
                                 random_state=3213,
                                 learning_rate=0.3,
                                 min_samples_split=5,
                                 subsample=1.0), [
        MLPRegressor(hidden_layer_sizes=(2,), max_iter=50, random_state=45),
        LinearRegression()
    ])

    my_boost.fit(X_train, y_train)

    linreg = LinearRegression()
    perceptron = MLPRegressor(hidden_layer_sizes=(2,), max_iter=50, random_state=45)

    linreg.fit(X_train, y_train)
    perceptron.fit(X_train, y_train)

    linreg_pred = linreg.predict(X_test)
    perceptron_pred = perceptron.predict(X_test)
    boosting_pred = my_boost.predict(X_test)

    linreg_error = mean_squared_error(y_test, linreg_pred)
    perceptron_error = mean_squared_error(y_test, perceptron_pred)
    boosting_error = mean_squared_error(y_test, boosting_pred)

    # sk_blending = Blending(sk_boost, [
    #     MLPRegressor(hidden_layer_sizes=(2,), max_iter=100),
    #     LinearRegression()
    # ])

    #_, sk_blending_train_time = calc_time(sk_blending.fit, X_train, y_train)
    _, my_blending_train_time = calc_time(my_blending.fit, X_train, y_train)

    #sk_pred = sk_blending.predict(X_test)
    my_pred = my_blending.predict(X_test)

    #sk_blending_error = mean_squared_error(y_test, sk_pred)
    my_blending_error = mean_squared_error(y_test, my_pred)




    print('my blending MSE on test: ', my_blending_error)
    print('linreg mse on test: ', linreg_error)
    print('perceptron mse on test: ', perceptron_error)
    print('boosting mse on test: ', boosting_error)

    #print('sk blending MSE on test: ', sk_blending_error)


    # print('sk blending train time: ', sk_blending_train_time)
    # print('my blending train time: ', my_blending_train_time)

    # my_blending = Blending( GradientBoostTree(verbose=True,
    #                              base_estimator=MyTreeRegressor,
    #                              max_depth=2,
    #                              n_estimators=400,
    #                              random_state=3213,
    #                              learning_rate=0.3,
    #                              min_samples_split=5,
    #                              subsample=0.5), [
    #     MLPRegressor(hidden_layer_sizes=(2,), max_iter=100),
    #     LinearRegression()
    # ])
    #
    # sk_blending = Blending(GradientBoostingRegressor(
    #     verbose=False,
    #     criterion='mse',
    #     max_depth=2,
    #     n_estimators=400,
    #     random_state=42,
    #     learning_rate=0.3,
    #     min_samples_split=5,
    #     subsample=0.5), [
    #     MLPRegressor(hidden_layer_sizes=(2,), max_iter=100),
    #     LinearRegression()
    # ])
    #
    # sk_errors = sk_blending.fit_predict_test(X_train, y_train, X_test, y_test, 400)
    # sk_errors = np.array(sk_errors)
    #
    #
    # my_blending.fit(X_train, y_train)
    # _, my_errors = my_blending.predict_over_estimators(X_test, y_test)
    # my_errors = np.array(my_errors)
    #
    # plt.xlabel('n_estimators')
    # plt.ylabel('MSE')
    # plt.plot(my_errors, color= 'green', label = 'my blending test error')
    # plt.plot(sk_errors, color = 'blue', label = 'sklearn blending test error')
    # plt.plot(sk_errors + 0.03 * sk_errors, color='grey', linestyle='--', label = 'sklearn error + 3%')
    # plt.plot(sk_errors - 0.03 * sk_errors, color = 'grey', linestyle='--', label = 'sklearn error - 3%')
    # plt.legend()
    # plt.show()






if find_blending_real_best_params:
    X_train, y_train = load_svmlight_file('../../Data/hw1/reg.train.txt')
    X_test, y_test = load_svmlight_file('../../Data/hw1/reg.test.txt')
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    possible_lr = list(np.linspace(0.01, 1.0, 5))
    possible_subsampling = [0.5, 0.6, 1.0]
    possible_max_depth = [1,2,4]

    from itertools import product
    from tqdm import tqdm

    results = []
    params = list(product(possible_lr, possible_subsampling, possible_max_depth))
    for i, (lr, subsample, max_depth) in enumerate(params):
        my_boost = GradientBoostTree(verbose=False,
                                     base_estimator=MyTreeRegressor,
                                     max_depth= max_depth,
                                     n_estimators=400,
                                     learning_rate= lr,
                                     min_samples_split= 20,
                                     subsample= subsample)

        my_blending = Blending(my_boost, [
            MLPRegressor(hidden_layer_sizes=(2,), max_iter=100),
            LinearRegression()
        ])

        my_blending.fit(X_train, y_train)
        my_pred, errors = my_blending.predict_over_estimators(X_test, y_test)

        n_estimators_best = np.argmin(np.array(errors)) + 1
        error_best =  min(errors)

        #my_error = mean_squared_error(y_test, my_pred)
        res = [(lr, subsample, max_depth, n_estimators_best), error_best]

        results.append(res)
        print('{0}/{1}  mse: {2}  params: {3} '.format(i, len(params), error_best , (lr, subsample, max_depth, n_estimators_best)))


    best_params = min(results, key = lambda  x: x[1])
    print('BEST: ', best_params)

    pickle.dump(results, open('my_blending_params.pkl', 'wb'))


