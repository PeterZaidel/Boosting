# Import the necessary modules and libraries

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from datetime import datetime
from XGBoostTree import XGBoostTree
from XGBoosting import XGBoosting
from XGBoosting import MSEMetric, LogLossMetric

import xgboost as xgb

def calc_time(fn, *args):
    dt0 = datetime.now()
    res = fn(*args)
    dt1 = datetime.now()
    return res, dt1-dt0

test_xgboost_housing = True
test_xgboost_real = True

if test_xgboost_real:

    data_train = np.loadtxt('../../Data/hw1/spam.train.txt')
    data_test = np.loadtxt('../../Data/hw1/spam.test.txt')
    X_train, y_train = data_train[:, 1:], data_train[:, 0]
    X_test, y_test = data_test[:, 1:], data_test[:, 0]

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    n_estimators = 400

    my_boost = XGBoosting(verbose=True,
                          max_depth=5,
                          n_estimators= n_estimators,
                          random_state=87765,
                          learning_rate=0.2,
                          subsample=0.7,
                          gamma=1.0,
                          lamb=2.0,
                          loss=LogLossMetric)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    xgb_params = {'max_depth': 5,
                  'eta': 0.2,
                  'lambda': 2.0,
                   #'alpha': 1.0,
                   'gamma': 1.0,
                   'subsample': 0.7,
                   'tree_method': 'exact',
                  #'min_child_weight': 0.0,
                  #'scale_pos_weight':1.0,# float(y_train[y_train == 1].shape[0])/float(y_train[y_train == 0].shape[0]),
                  'silent': 1,
                  'objective': 'binary:logistic',
                  'eval_metric': 'logloss'}

    num_round =  n_estimators
    evres = dict()
    model = xgb.train(xgb_params, dtrain, num_round, evals=[(dtest, 'test'), (dtrain, 'train')], evals_result=evres,
                      verbose_eval=1)
    xgb_pred = model.predict(dtest)

    xgb_errors = np.array(evres['test']['logloss'])


    print("Training My Boosting: \n")
    _, my_boost_train_time = calc_time(my_boost.fit, X_train, y_train)
    # my_boost_pred, my_boost_test_time = calc_time(my_boost.predict, X_test, y_test)

    my_boost_pred, my_boost_errors  = my_boost.predict_over_estimators(X_test, y_test)


    plt.xlabel('n_estimators')
    plt.ylabel('LogLoss')
    plt.plot(my_boost_errors, color= 'green', label = 'my  test error')
    plt.plot(xgb_errors, color = 'blue', label = 'xgboost test error')
    plt.plot(xgb_errors + 0.03 * xgb_errors, color='grey', linestyle='--', label = 'xgboost error + 3%')
    plt.plot(xgb_errors - 0.03 * xgb_errors, color = 'grey', linestyle='--', label = 'xgboost error - 3%')
    plt.legend()
    plt.show()



    #my_boost_error = log_loss(y_test, my_boost_pred)
    #xgb_boost_error = log_loss(y_test, xgb_pred)
    #
    # print("----------------------")
    #
    # print('my boosting LogLoss: ', my_boost_error)
    # print('xgboost boosting LogLoss: ', xgb_boost_error)
    #
    # print('my boosting train time: ', my_boost_train_time)
    #
    # print('my boosting pred time: ', my_boost_test_time)