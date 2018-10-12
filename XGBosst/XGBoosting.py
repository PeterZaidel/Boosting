import numpy as np
from RegressionTree import MyTreeRegressor

from XGBoostTree import XGBoostTree
from VovaXGboostTree import MyDecisionTreeRegressorGained
from sklearn.metrics import log_loss
from sklearn.ensemble  import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(x))



class ConstantLogLossModel:
    def __init__(self):
        pass
    def fit(self, X, y):
        self._mean = y.mean()

    def predict(self, X):
        sigmoid_prediction = np.repeat(self._mean, X.shape[0])
        return np.log(sigmoid_prediction) - np.log(1.0 - sigmoid_prediction)

    def predict_proba(self, X):
        return np.c_[np.repeat(1.0 - self._mean, X.shape[0]), np.repeat(self._mean, X.shape[0])]


class ConstantMSEModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self._mean = 2.0 * y.mean()

    def predict(self, X):
        return np.repeat(self._mean, X.shape[0])


class LogRegModel:
    def __init__(self):
        self._model = LogisticRegression()
        pass

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        pred = self._model.predict_proba(X)
        return pred[:, 1]


class LogLossMetric:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        sigmoid_pred = sigmoid(y_pred)
        return log_loss(y_true, sigmoid_pred)
        #return (-y_true * np.log(sigmoid_pred) - (1.0 - y_true)* np.log(1.0 - sigmoid_pred)).mean()

    def grad(self, y_true, y_pred):
        return sigmoid(y_pred) - y_true

    def gessian(self, y_true, y_pred):
        return sigmoid(y_pred) - sigmoid(y_pred)**2

    def forward_predict(self, pred):
        return sigmoid(pred)


class MSEMetric:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def grad(self, y_true, y_pred):
        return 2.0 * (y_true - y_pred)

    def gessian(self, y_true, y_pred):
        return 2.0 * np.ones(y_true.shape[0])

    def forward_predict(self, pred):
        return pred


class XGBoosting:

    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 max_depth=3,
                 lamb = 0.1,
                 gamma = 0.0,
                 scale_pos_weight = 1.0,

                 random_state=None,
                 max_features=None,
                 verbose=False,
                 base_estimator = XGBoostTree,
                 init_estimator = ConstantLogLossModel,
                 loss = MSEMetric):

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample

        self.max_depth = max_depth
        self.max_features = max_features
        self.gamma = gamma
        self.lm = lamb

        self.scale_pos_weight = scale_pos_weight


        self.random_state = random_state

        self.verbose = verbose

        self.base_estimator = base_estimator
        self.loss = loss()
        self.init_estimator = init_estimator
        pass



    def __generate_subsample_mask(self, X_shape):
        ids = np.random.choice(np.arange(X_shape[0]), size=self.subsample_size)
        mask = np.zeros(X_shape[0]).astype(bool)
        mask[ids] =True
        return mask

    def __subsample_data(self, X, y, grad):
        #return X, y
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        return X[ids[:self.subsample_size]], y[ids[:self.subsample_size]], grad[ids[:self.subsample_size]]


    def fit(self, X, y):

        self.subsample_size = int(self.subsample * X.shape[0])

        self.models = [None]*self.n_estimators


        self.models[0] = self.init_estimator()
        self.models[0].fit(X,y)

        prediction = self.models[0].predict(X)


        if self.verbose:
            print('model[{0}], train loss: {1}'.format(0, self.loss(y, prediction)))


        train_errors = []
        train_errors.append(self.loss(y, prediction))
        current_depth = self.max_depth

        for i in range(self.n_estimators):

            anti_grad = -1.0 * self.loss.grad(y, prediction)
            gessian =  self.loss.gessian(y, prediction)

            self.models[i] = self.base_estimator(
                                   max_depth= current_depth,
                                   lamb=  self.lm,
                                   gamma = self.gamma
                                   )

            subsample_mask = self.__generate_subsample_mask(X.shape)

            subsampled_X = X[subsample_mask]
            subsampled_y = y[subsample_mask]
            subsampled_prediction = prediction[subsample_mask]

            # subsampled_sigmoid_prediction = sigmoid_prediction[subsample_mask]

            subsampled_anti_grad = anti_grad[subsample_mask]
            subsampled_gessian = gessian[subsample_mask]


            self.models[i].fit(subsampled_X, subsampled_anti_grad, subsampled_gessian)

            ai = self.models[i].predict(X)
            prediction = prediction + self.learning_rate * ai

            if self.verbose:
                print('model[{0}], train loss: {1},  depth = {2}]'.format(i, self.loss(y, prediction), current_depth))

            train_errors.append(self.loss(y, prediction))

        return self

    def predict(self, X, y_true= None):
        pred = np.zeros(X.shape[0])

        for i, m in enumerate(self.models):
            lr = self.learning_rate
            if i == 0:
                lr = 1.0

            pred = pred + lr * m.predict(X)

            if self.verbose and y_true is not None:
                print('n_estimators:{0} test loss:{1}'.format(i+1, self.loss(y_true, pred)))

        return self.loss.forward_predict(pred)

    def predict_over_estimators(self, X, y_true):
        pred = np.zeros(X.shape[0])
        errors = []

        for i, m in enumerate(self.models):
            lr = self.learning_rate
            if i == 0:
                lr = 1.0
            pred = pred +  lr * m.predict(X)
            if self.verbose and y_true is not None:
                print('n_estimators:{0}  test loss:{1}'.format(i + 1, self.loss(y_true, pred)))
            errors.append(self.loss(y_true, pred))

        return self.loss.forward_predict(pred), errors
