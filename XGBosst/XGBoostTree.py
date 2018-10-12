import numpy as np

# self.tree[node_id] = {'type': self.NON_LEAF_TYPE,
#                       'node_id': node_id,
#                       'feature_id': best_feature_id,
#                       'threshold': best_treshold,
#                       'impurity': best_impurity,
#                       'obj_num': X.shape[0],
#                       'value': node_value}

class TreeNode:
    NON_LEAF_TYPE = 'NON_LEAF'
    LEAF_TYPE = 'LEAF'


    node_id = None
    value = None
    gain = float('-inf')
    objects_in_node = None
    type = NON_LEAF_TYPE
    feature_id = None
    threshold = None

    def __init__(self):
        pass

class XGBoostTree:


    NONE_VALUE = float('-inf')

    def __init__(self,
                 max_depth=5,
                 max_features = None,
                 random_state = None,
                 split_on_diff = True,
                 lamb = 1.0,
                 gamma = 1.0):

        #self.tree = dict()

        max_count_nodes = (2**np.arange(max_depth+1)).sum()

        self.tree = [None]*max_count_nodes

        self.lamb = lamb
        self.gamma = gamma

        self.max_depth = max_depth
        self.random_state = random_state
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
        #np.random.shuffle(feature_ids)

        return feature_ids  # Ваш код

    def __sort_samples(self, x, y, feature_idx = None):
        sorted_idx = x.argsort(axis=0)
        return x[sorted_idx], y[sorted_idx]



    def __calc_xgboost_impurities_array(self, grad, gessian):
        N = grad.shape[1]

        G_full =  (grad).sum(1)
        S_full = (gessian).sum(1)

        full_Gain = (G_full)**2 / (S_full + self.lamb)
        sum_G_L = (grad)[:, :N-1].cumsum(1)
        sum_G_R = G_full[:, None] - sum_G_L

        sum_G_L_squared = sum_G_L**2
        sum_G_R_squared = sum_G_R**2

        sum_S_L = (gessian)[:, :N-1].cumsum(1)
        sum_S_R = S_full[:, None] - sum_S_L

        impurities = sum_G_L_squared/ (sum_S_L + self.lamb) \
                     + sum_G_R_squared/ (sum_S_R + self.lamb) \
                     - full_Gain[:, None] - self.gamma

        return impurities


    def __calc_possible_border_ids(self, X, eps = 1e-15):
        diff = np.diff(X.T, axis = 1)
        idxs = np.argwhere(np.abs(diff) > eps)
        list_result_border_ids = []
        for i in range(X.shape[1]):
            pidxs = idxs[idxs[:, 0] == i][:,1]
            if pidxs.shape[0] == 0:
                pidxs = np.array([self.NONE_VALUE])
            list_result_border_ids.append(pidxs)

        return list_result_border_ids


    def __div_samples(self, x, grad, grad2, feature_id, threshold):
        left_mask = x[:, feature_id] < threshold
        right_mask = ~left_mask
        return x[left_mask], x[right_mask], grad[left_mask], grad[right_mask], grad2[left_mask], grad2[right_mask]



    def __find_threshold_array(self, X, grad, gessian, feature_ids):
        sorted_ids = np.argsort(X[:, feature_ids], axis=0)
        sorted_X = X[sorted_ids, feature_ids]
        sorted_grad  = grad[sorted_ids].T
        sorted_gessian = gessian[sorted_ids].T

        possible_border_ids = self.__calc_possible_border_ids(sorted_X)


        # possible_border_ids = np.arange(self.min_samples_split, sorted_X.shape[0] - self.min_samples_split)
        if len(sorted_grad.shape) ==1:
            sorted_Y = sorted_grad[None, :]

        if len(sorted_X.shape) == 1:
            sorted_X = sorted_X[None, :]

        gains = self.__calc_xgboost_impurities_array(sorted_grad, sorted_gessian)

        if not self.split_on_diff:
            max_gain_index = np.argmax(gains, axis =1)
            max_gain = gains[np.arange(gains.shape[0]), max_gain_index.ravel()]
        else:
            max_gain_index = []
            max_gain_index_none_ids = []

            for feature_index, feature_pbids in enumerate(possible_border_ids):
                if feature_pbids[0] == self.NONE_VALUE:
                    max_gain_index.append(0)
                    max_gain_index_none_ids.append(len(max_gain_index)-1)
                    continue

                idx = np.argmax(gains[feature_index, feature_pbids])
                max_gain_index.append(feature_pbids[idx])

            max_gain_index = np.array(max_gain_index)
            max_gain_index_none_ids = np.array(max_gain_index_none_ids)
            try:
                max_gain = gains[np.arange(gains.shape[0]), max_gain_index.ravel()]
            except:
                print('aa')
            if max_gain_index_none_ids.shape[0] > 0:
                max_gain[max_gain_index_none_ids] = self.NONE_VALUE

        best_split_right_index = max_gain_index + 1

        # max_gain[best_split_right_index <= self.min_samples_split] = self.NONE_VALUE
        # max_gain[best_split_right_index >= sorted_grad.shape[1] - self.min_samples_split] = self.NONE_VALUE

        best_threshold = (sorted_X[best_split_right_index-1, np.arange(sorted_X.shape[1])]
                          + sorted_X[best_split_right_index, np.arange(sorted_X.shape[1])] ) / 2.0
        #best_threshold = sorted_X[best_split_right_index, np.arange(sorted_X.shape[1])]

        best_gain_index = np.argmax(max_gain)
        best_gain = max_gain[best_gain_index]

        best_treshold = best_threshold[best_gain_index]
        best_feature_id = feature_ids[best_gain_index]

        return best_gain, best_feature_id, best_treshold



    def __get_node_value(self, x, grad, gessian):
        G =  (grad).sum()
        S = (gessian).sum()

        node_value =  -1.0 * G / (S + self.lamb)
        return node_value

    def __set_leaf(self, x, grad, gessian,  node_id):

        leaf_value = self.__get_node_value(x, grad, gessian)

        node = TreeNode()
        node.node_id = node_id
        node.value = leaf_value
        node.type = TreeNode.LEAF_TYPE
        node.objects_in_node = x.shape[0]
        node.gain = 0

        self.tree[node_id] = node

    def __check_is_leaf(self, X, grad, gessian, depth):

        if X.shape[0] < 2:
            return True

        # проверка, является ли вершина листовой
        if depth >= self.max_depth:
            return True

        return  False



    def __fit_node(self, X, grad, gessian, node_id, depth):

        #print('fitting depth: {0} node_id: {1}'.format(depth, node_id))

        if self.__check_is_leaf(X, grad, gessian, depth):
            self.__set_leaf(X, grad, gessian, node_id)
            return

        feature_ids = self.get_feature_ids(X.shape[1])

        #th_results = self.split_xgboost(X, grad, gessian)
        th_results = self.__find_threshold_array(X, grad, gessian, feature_ids)
        # gains = th_results[0]
        # thresholds = th_results[1]
        #
        # best_gain_index = np.argmax(gains)
        # best_gain = gains[best_gain_index]
        #
        # best_treshold = thresholds[best_gain_index]
        # best_feature_id = feature_ids[best_gain_index]

        best_gain,best_feature_id, best_treshold = th_results

        # # если не удалось разделить выборку, то узел - выходной
        # if best_impurity == self.NONE_VALUE:
        #     self.__set_leaf(X, grad, gessian, node_id)
        #     return

        left_mask = (X[:, best_feature_id] > best_treshold)
        right_mask = ~left_mask

        left_x, left_grad = X[left_mask], grad[left_mask]
        right_x, right_grad = X[right_mask], grad[right_mask]
        left_gessian = gessian[left_mask]
        right_gessian = gessian[right_mask]

        if left_x.shape[0] == 0 or right_x.shape[0] == 0:
            self.__set_leaf(X, grad, gessian, node_id)
            return


        node = TreeNode()

        node.type = TreeNode.NON_LEAF_TYPE
        node.node_id = node_id
        node.objects_in_node = X.shape[0]

        node.value = self.__get_node_value(X, grad, gessian)
        node.gain = best_gain
        node.threshold = best_treshold
        node.feature_id = best_feature_id

        self.tree[node_id] = node

        # self.tree[node_id] = {'type': self.NON_LEAF_TYPE,
        #                       'node_id': node_id,
        #                       'feature_id': best_feature_id,
        #                       'threshold': best_treshold,
        #                       'impurity':  best_impurity,
        #                       'obj_num': X.shape[0],
        #                       'value': node_value}

        self.__fit_node(left_x, left_grad, left_gessian, 2 * node_id + 1, depth + 1)
        self.__fit_node(right_x, right_grad,  right_gessian, 2 * node_id + 2, depth + 1)

        return

    def fit(self, X, grad, gessian):
        self.__fit_node(X, grad, gessian, 0, 0)
        self.pruning(0)
        # for node_id, node in enumerate(self.tree):
        #     if node is not None and node['type'] == self.LEAF_TYPE:
        #             self.__pruning(node_id)

    def pruning(self, node_id):
        node = self.tree[node_id]

        if node.type == TreeNode.NON_LEAF_TYPE:
            result = (node.gain < 0) & self.pruning(node_id * 2 + 1) & self.pruning(node_id * 2 + 2)
            if result == 1:
                # print('pruning '+ str(node.gain))
                self.tree[node_id].type = TreeNode.LEAF_TYPE
            return result
        elif node.gain > 0:
            return 0
        else:
            return 1



    def __predict_value(self, x, node_id):
        node = self.tree[node_id]

        if node.type == TreeNode.NON_LEAF_TYPE:
            feature_id, threshold = node.feature_id, node.threshold
            if x[feature_id] > threshold:
                return self.__predict_value(x, 2 * node_id + 1)
            else:
                return self.__predict_value(x, 2 * node_id + 2)
        else:
            return node.value

    def predict(self, X):
        return np.array([self.__predict_value(x, 0) for x in X])

    def get_leaf_values(self):
        res = []
        for node in self.tree:
            if node.type == TreeNode.LEAF_TYPE:
                res.append(node.value)
        return res
