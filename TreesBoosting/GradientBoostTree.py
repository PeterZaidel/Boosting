import numpy as np
from RegressionTree import MyTreeRegressor
from sklearn.metrics import mean_squared_error

class GradientBoostTree:

    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 min_samples_split=2,
                 max_depth=3,
                 super_max_depth = 5,
                 min_impurity_split=None,
                 random_state=None,
                 max_features=None,
                 verbose=False,
                 presort=True,
                 base_estimator = MyTreeRegressor):

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.super_max_depth = super_max_depth


        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features

        self.presort = presort


        self.random_state = random_state

        self.verbose = verbose

        self.base_estimator = base_estimator

        pass



    @staticmethod
    def __metric(y_pred, y_true):
        return ((y_pred - y_true)**2).mean()

    @staticmethod
    def __metric_antigrad( y_pred, y_true):
        return -2.0 * (y_pred - y_true)

    @staticmethod
    def __metric_step(y_pred, error):
        return (y_pred * error).sum() / ((y_pred**2).sum())


    # def __init_estimator(self):
    #     return self.base_estimator(min_samples_split=self.min_samples_split,
    #                                max_depth=self.max_depth,
    #                                max_features=self.max_features,
    #                                min_impurity_split=self.min_impurity_split
    #                                )

    def __generate_subsample_mask(self, X_shape):
        ids = np.random.choice(np.arange(X_shape[0]),size=self.subsample_size)
        mask = np.zeros(X_shape[0]).astype(bool)
        mask[ids] =True
        return mask



    def __subsample_data(self, X, y, grad):
        #return X, y
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        return X[ids[:self.subsample_size]], y[ids[:self.subsample_size]], grad[ids[:self.subsample_size]]


    def fit(self, X, y):

        if self.presort:
            self._presorted_X = np.argsort(X, axis=0)

        self.subsample_size = int(self.subsample * X.shape[0])

        self.models = [None]*self.n_estimators
        self.b = np.zeros(self.n_estimators, dtype=float)


        self.models[0] = self.base_estimator(min_samples_split=self.min_samples_split,
                                   max_depth=self.max_depth,
                                   random_state=self.random_state
                                   )
        self.models[0].fit(X,y)
        self.b[0] = 1.0

        prediction = self.models[0].predict(X)

        if self.verbose:
            print('model[{0}], loss: {1}, b[{0}] = {2}'.format(0, mean_squared_error(y, prediction), self.b[0]))


        train_errors = []
        train_errors.append(mean_squared_error(y, prediction))
        current_depth = self.max_depth

        for i in range(1, self.n_estimators):

            grad_loss = self.__metric_antigrad(prediction, y)
            self.models[i] = self.base_estimator(min_samples_split=self.min_samples_split,
                                   max_depth= current_depth,
                                   random_state=self.random_state
                                   )

            subsample_mask = self.__generate_subsample_mask(X.shape)

            subsampled_X = X[subsample_mask]
            subsampled_y = y[subsample_mask]
            subsampled_grad = grad_loss[subsample_mask]

            out_of_bag_X = X[~subsample_mask]
            out_of_bag_y = y[~subsample_mask]
            out_of_bag_grad = grad_loss[~subsample_mask]

            #subsampled_X, subsampled_y, subsampled_grad = self.__subsample_data(X, y, grad_loss)
            #self.models[i].fit(X, grad_loss)

            self.models[i].fit(subsampled_X, subsampled_grad)

            # TODO: ORIGINAL
            # ai = self.models[i].predict(X)
            # self.b[i] = self.learning_rate * self.__metric_step(ai, y - prediction)
            if self.subsample  < 1.0:
                ai = self.models[i].predict(X[subsample_mask])
                self.b[i] = self.learning_rate * self.__metric_step(ai, y[subsample_mask] - prediction[subsample_mask])
            else:
                ai = self.models[i].predict(X)
                self.b[i] = self.learning_rate * self.__metric_step(ai, y - prediction)

            ai = self.models[i].predict(X)
            prediction = prediction + self.b[i] * ai


            if self.verbose:
                print('model[{0}], loss: {1}, b[{0}] = {2} , depth = {3}]'.format(i, mean_squared_error(y, prediction), self.b[i], current_depth))

            train_errors.append(mean_squared_error(y, prediction))
            # if len(train_errors) > 4 \
            #         and  np.abs(train_errors[-1] - train_errors[-2])  < 1e-4 \
            #         and current_depth < self.max_depth:
            #     current_depth += 1



        return self

    def predict(self, X, y_true= None):
        pred = np.zeros(X.shape[0])
        pred = pred + self.models[0].predict(X)
        if self.verbose and y_true is not None:
            print('n_estimators:{0}  mse:{1}'.format(1, mean_squared_error(y_true, pred)))

        for i, m in enumerate(self.models):
            if i ==0:
                continue
            pred = pred + self.b[i] * m.predict(X)
            if self.verbose and y_true is not None:
                print('n_estimators:{0}  mse:{1}'.format(i+1, mean_squared_error(y_true, pred)))

        # pred = np.zeros(X.shape[0])
        # for i, m in enumerate(self.models):
        #     pred = pred + self.b[i] * m.predict(X)
        #     if self.verbose and y_true is not None:
        #         print('n_estimators:{0}  mse:{1}'.format(i + 1, mean_squared_error(y_true, pred)))

        return pred

    def predict_over_estimators(self, X, y_true):
        pred = np.zeros(X.shape[0])
        pred = pred + self.models[0].predict(X)

        errors = []

        if self.verbose and y_true is not None:
            print('n_estimators:{0}  mse:{1}'.format(1, mean_squared_error(y_true, pred)))

        errors.append(mean_squared_error(y_true, pred))

        for i, m in enumerate(self.models):
            if i == 0:
                continue
            pred = pred + self.b[i] * m.predict(X)
            if self.verbose and y_true is not None:
                print('n_estimators:{0}  mse:{1}'.format(i + 1, mean_squared_error(y_true, pred)))
            errors.append(((y_true - pred)**2).mean())

        return pred, errors
