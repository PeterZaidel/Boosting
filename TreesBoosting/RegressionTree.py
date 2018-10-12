import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from threading import Lock

def pool_map(func, tasks, chunksize = 1 , threads_num = 20):
    pool = ThreadPool(threads_num)
    res = pool.map(func, tasks, chunksize=chunksize)
    pool.close()
    pool.join()
    return res

class MyTreeRegressor:
    NON_LEAF_TYPE = 'NON_LEAF'
    LEAF_TYPE = 'LEAF'

    def __init__(self,
                 min_samples_split=3,
                 max_depth=5,
                 min_impurity_split=0.1,
                 max_features = None,
                 random_state = None,
                 #decrease_leaf = False,
                 split_on_diff = True):

        self.tree = dict()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state
        #self.decrease_leaf = decrease_leaf
        self.split_on_diff = split_on_diff

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if max_features == 'sqrt':
            self.get_feature_ids = self.__get_feature_ids_sqrt
        elif max_features == 'log2':
            self.get_feature_ids = self.__get_feature_ids_log2
        elif max_features is None:
            self.get_feature_ids = self.__get_feature_ids_N
        else:
            raise -1


    def __get_feature_ids_sqrt(self, n_feature):

        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)

        return feature_ids[:np.sqrt(n_feature)]  # Ваш код

    def __get_feature_ids_log2(self, n_feature):
        feature_ids = np.arange(n_feature)

        np.random.shuffle(feature_ids)

        return feature_ids[:np.log2(n_feature)]  # Ваш код

    def __get_feature_ids_N(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)

        return feature_ids  # Ваш код

    def __sort_samples(self, x, y, feature_idx = None):
        sorted_idx = x.argsort(axis=0)
        return x[sorted_idx], y[sorted_idx]

    @staticmethod
    def __div_samples(x, y, feature_id, threshold):
        left_mask = x[:, feature_id] > threshold
        right_mask = ~left_mask
        return x[left_mask], x[right_mask], y[left_mask], y[right_mask]


    @staticmethod
    def __calc_impurities_array(Y):
        N = Y.shape[1]

        y_squared = Y**2

        sum_squared_L = (y_squared[:, :N - 1]).cumsum(1)
        sum_squared_R = y_squared.sum(1)[:, None] - sum_squared_L

        mean_squared_L = sum_squared_L / np.arange(1, N)
        mean_squared_R = sum_squared_R / (N - np.arange(1, N))

        sum_L = (Y[:, :N - 1]).cumsum(1)
        sum_R = Y.sum(1)[:, None] - sum_L

        mean_L = sum_L / np.arange(1, N)
        mean_R = sum_R / (N - np.arange(1, N))


        cum_std_L = mean_squared_L - mean_L**2
        cum_std_R = mean_squared_R - mean_R**2

        impurities = cum_std_L * (np.arange(1, N) / N) + cum_std_R * ((N - np.arange(1, N)) / N)

        return impurities

    def __calc_possible_border_ids(self, X, eps = 1e-27):
        diff = np.diff(X.T, axis = 1)
        idxs = np.argwhere(np.abs(diff) > eps)
        list_result_border_ids = []
        for i in range(X.shape[1]):
            pidxs = idxs[idxs[:, 0] == i][:,1]
            if pidxs.shape[0] == 0:
                pidxs = np.array([float('+inf')])
            list_result_border_ids.append(pidxs)

        return list_result_border_ids
    def __find_threshold_array(self, X, Y, feature_ids):
        #sorted_X, sorted_Y = self.__sort_samples(X[:, feature_ids],Y)
        sorted_ids = np.argsort(X[:, feature_ids], axis=0)
        sorted_X, sorted_Y = X[sorted_ids, feature_ids], Y[sorted_ids]

        possible_border_ids = self.__calc_possible_border_ids(sorted_X)


        # possible_border_ids = np.arange(self.min_samples_split, sorted_X.shape[0] - self.min_samples_split)
        if len(sorted_Y.shape) ==1:
            sorted_Y = sorted_Y[None, :]

        if len(sorted_X.shape) == 1:
            sorted_X = sorted_X[None, :]

        sorted_Y = sorted_Y.T
        impurities = self.__calc_impurities_array(sorted_Y)

        if not self.split_on_diff:
            min_impurity_index = np.argmin(impurities, axis =1)
            min_impurity = impurities[np.arange(impurities.shape[0]), min_impurity_index.ravel()]
        else:
            min_impurity_index = []
            min_impurity_index_none_ids = []

            for feature_index, feature_pbids in enumerate(possible_border_ids):
                if feature_pbids[0] == float('+inf'):
                    min_impurity_index.append(0)
                    min_impurity_index_none_ids.append(len(min_impurity_index)-1)
                    continue

                idx = np.argmin(impurities[feature_index, feature_pbids])
                min_impurity_index.append(feature_pbids[idx])

            min_impurity_index = np.array(min_impurity_index)
            min_impurity_index_none_ids = np.array(min_impurity_index_none_ids)
            min_impurity = impurities[np.arange(impurities.shape[0]), min_impurity_index.ravel()]
            if min_impurity_index_none_ids.shape[0] > 0:
                min_impurity[min_impurity_index_none_ids] = float('+inf')

        best_split_right_index = min_impurity_index + 1

        min_impurity[best_split_right_index <= self.min_samples_split] = np.float('+inf')
        min_impurity[best_split_right_index >= sorted_Y.shape[1] - self.min_samples_split] = np.float('+inf')

        best_threshold = (sorted_X[best_split_right_index-1, np.arange(sorted_X.shape[1])]
                          + sorted_X[best_split_right_index, np.arange(sorted_X.shape[1])] ) / 2.0
        #best_threshold = sorted_X[best_split_right_index, np.arange(sorted_X.shape[1])]
        return min_impurity, best_threshold



    def __set_leaf(self, x, y, node_id):

        mean_value = y.mean()

        # if self.decrease_leaf:
        #     n = float(x.shape[0])
        #     mean_value = mean_value * (n/(1.0 + n + np.sqrt(n)))

        impurity = ((y - y.mean())**2).mean()
        self.tree[node_id] = {'type': self.LEAF_TYPE,
                              'value': mean_value,
                              'obj_num': x.shape[0],
                              'impurity': impurity
                               }

    def __check_is_leaf(self, X, Y, depth):

        if X.shape[0] < 2 * self.min_samples_split + 2 or np.unique(Y).shape[0] == 1:
            return True

        # проверка, является ли вершина листовой
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        # если дисперсия в листе меньше порога
        if self.min_impurity_split is not None and \
                        np.mean((Y - Y.mean()) ** 2) < self.min_impurity_split:
            return True

        return  False


    def __fit_node(self, X, Y, node_id, depth):

        #print('fitting depth: {0} node_id: {1}'.format(depth, node_id))

        if self.__check_is_leaf(X, Y, depth):
            self.__set_leaf(X, Y, node_id)
            return

        node_impurity = ((Y - Y.mean())**2).mean()

        feature_ids = self.get_feature_ids(X.shape[1])#np.array([48])#np.arange(X.shape[1])#

        th_results = self.__find_threshold_array(X, Y, feature_ids)
        impurities = th_results[0]
        thresholds = th_results[1]

        best_impurity_index = np.argmin(impurities)
        best_impurity = impurities[best_impurity_index]

        best_treshold = thresholds[best_impurity_index]
        best_feature_id = feature_ids[best_impurity_index]

        # если не удалось разделить выборку, то узел - выходной
        if best_impurity == float('+inf'):
            self.__set_leaf(X, Y, node_id)
            return

        left_x, right_x, left_y, right_y = self.__div_samples(X, Y, best_feature_id, best_treshold)

        if left_x.shape[0] == 0 or right_x.shape[0] == 0:
            self.__set_leaf(X, Y, node_id)
            return

        self.tree[node_id] = {'type': self.NON_LEAF_TYPE,
                              'feature_id': best_feature_id,
                              'threshold': best_treshold,
                              'impurity': node_impurity,
                              'obj_num': X.shape[0]}

        self.__fit_node(left_x, left_y, 2 * node_id + 1, depth + 1)
        self.__fit_node(right_x, right_y, 2 * node_id + 2, depth + 1)

        return

    def fit(self, X, Y):
        self.__fit_node(X, Y,  0, 0)

    def __predict_value(self, x, node_id):
        node = self.tree[node_id]

        if node['type'] == self.__class__.NON_LEAF_TYPE:
            feature_id, threshold = node['feature_id'], node['threshold']
            if x[feature_id] > threshold:
                return self.__predict_value(x, 2 * node_id + 1)
            else:
                return self.__predict_value(x, 2 * node_id + 2)
        else:
            return node['value']

    def __predict_value_fast(self, X, node_id, mask):
        node = self.tree[node_id]

        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_id, threshold = node

            mask_left = (X[:, feature_id] <= threshold) & mask
            mask_right = (~mask_left) & mask

            res = self.__predict_value_fast(X, 2*node_id +1, mask_left) \
                  + self.__predict_value_fast(X, 2*node_id +2, mask_right)

            return res

            #
            # if x[feature_id] > threshold:
            #     return self.__predict_value(x, 2 * node_id + 1)
            # else:
            #     return self.__predict_value(x, 2 * node_id + 2)
        else:
            res = np.zeros(X.shape[0])
            res[mask] = node[1]
            return res

    def predict(self, X):
        return np.array([self.__predict_value(x, 0) for x in X])

    def get_leaf_values(self):
        res = []
        for node in self.tree:
            if self.tree[node]['type'] == self.LEAF_TYPE:
                res.append(self.tree[node]['value'])
        return res
