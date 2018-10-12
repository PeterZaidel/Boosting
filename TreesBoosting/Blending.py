import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Blending:
    def __init__(self, metamodel=None, models=None, test_size=0.5):
        self.metamodel = metamodel
        self.models = models

        self.test_size = test_size

    def fit(self, X, y):
        X_train, X_meta, y_train, y_meta = train_test_split(X, y, test_size=self.test_size)

        for m in self.models:
            m.fit(X_train, y_train)

        y_meta_pred = np.array(list(map(lambda x: x.predict(X_meta), self.models))).T
        xx_meta = np.c_[X_meta, y_meta_pred]
        #self.metamodel.fit(y_meta_pred, y_meta)

        self.metamodel.fit(xx_meta, y_meta)


    def fit_predict_test(self, X, y, X_test, y_test, max_n_estimators):
        X_train, X_meta, y_train, y_meta = train_test_split(X, y, test_size=self.test_size)
        for m in self.models:
            m.fit(X_train, y_train)

        y_meta_pred = np.array(list(map(lambda x: x.predict(X_meta), self.models))).T

        y_test_meta = np.array(list(map(lambda x: x.predict(X_test), self.models))).T

        self.metamodel.n_estimators = 1
        self.metamodel.warm_start = True
        errors = []

        for est_num in range(1, max_n_estimators):
            self.metamodel.n_estimators = est_num
            self.metamodel.fit(y_meta_pred, y_meta)

            pred = self.metamodel.predict(y_test_meta)

            error = mean_squared_error(y_test, pred)
            errors.append(error)

        return errors



    def predict(self, X):
        y_meta = np.array(list(map(lambda x: x.predict(X), self.models))).T

        xx_meta = np.c_[X, y_meta]
        #return self.metamodel.predict(y_meta)

        return self.metamodel.predict(xx_meta)


    def predict_over_estimators(self, X, y_true):
        y_meta = np.array(list(map(lambda x: x.predict(X), self.models))).T
        return self.metamodel.predict_over_estimators(y_meta, y_true)