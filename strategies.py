"""
The :mod:`al.instance_strategies` implements various active learning strategies.
"""
import math
import numpy as np
from collections import defaultdict

import scipy.sparse as ss
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from numpy import linalg as LA
import scipy
import time

class RandomBootstrap(object):
    """Class - used if strategy selected is rand"""
    def __init__(self, seed):
        """Instantiate :mod:`al.instance_strategies.RandomBootstrap`

        **Parameters**
        * seed (*int*) - trial number."""

        self.randS = RandomStrategy(seed)

    def bootstrap(self, pool, y=None, k=1):
        """
        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * y - None or possible pool
        * k (*int*) - 1 or possible bootstrap size

        **Returns**
        * randS.chooseNext(pool, k=k) - choose next pool"""

        return self.randS.chooseNext(pool, k=k)

class BootstrapFromEach(object):
    """Class - used if not bootstrapped"""
    def __init__(self, seed):
        """Instantiate :mod:`al.instance_strategies.BootstrapFromEach`

        **Parameters**
        * seed (*int*) - trial number."""

        self.randS = RandomStrategy(seed)

    def bootstrap(self, pool, y, k=1):
        """
        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * y - None or possible pool
        * k (*int*) - 1 or possible bootstrap size

        **Returns**
        * chosen array of indices"""

        data = defaultdict(lambda: [])
        for i in pool:
            data[y[i]].append(i)
        chosen = []
        num_classes = len(data.keys())

        for label in data.keys():
            candidates = data[label]
            # k is bootstrap_size, num_classes is the # of labels, normalize for each
            indices = self.randS.chooseNext(candidates, k=k/num_classes)
            chosen.extend(indices)
        return chosen


class BaseStrategy(object):
    """Class - Base strategy"""
    def __init__(self, seed=0):
        """Instantiate :mod:`al.instance_strategies.BaseStrategy`

        **Parameters**
        * seed (*int*) - 0 or trial number."""

        self.randgen = np.random.RandomState(seed)

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        pass

class RandomStrategy(BaseStrategy):

    """Class - used if strategy is rand, inherits from :mod:`al.instance_strategies.BaseStrategy`"""
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        """Overide method BaseStrategy.chooseNext

        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * X - None or pool.toarray()
        * model - None
        * k (*int*) - 1 or step size
        * current_train_indices - None or array of trained indices
        * current_train_y - None or train_indices specific to y_pool

        **Returns**
        * [list_pool[i] for i in rand_indices[:k]] - array of random permutations given pool"""
        list_pool = list(pool)
        rand_indices = self.randgen.permutation(len(pool))
        return [list_pool[i] for i in rand_indices[:k]]

class UncStrategy(BaseStrategy):
    """Class - used if strategy selected is unc, inherits from :mod:`al.instance_strategies.BaseStrategy`"""
    def __init__(self, seed=0, sub_pool = None, method_uncerten='unc_1', num_chosen_from = 10, threshold = None, decision_tree_scale = False, index = None):

        """Instantiate :mod:`al.instance_strategies.UncStrategy`
        **Parameters**
        * seed (*int*) - 0 or trial number.
        * sub_pool - None or sub_pool parameter"""

        super(UncStrategy, self).__init__(seed=seed)
        self.sub_pool = sub_pool
        self.method_uncerten = method_uncerten
        self.num_choose_from = num_chosen_from
        self.threshold = threshold
        self.scale = decision_tree_scale
        self.index = index

    def chooseNext(self, pool, X=None, model=None, k=1 ,current_train_indices = None, current_train_y = None):
        """Overide method BaseStrategy.chooseNext

        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * X - None or pool.toarray()
        * model - None
        * k (*int*) - 1 or step size
        * current_train_indices - None or array of trained indices
        * current_train_y - None or train_indices specific to y_pool

        **Returns**
        * [candidates[i] for i in uis[:k]]"""
        if not self.sub_pool:
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:5000]]
        else:
            candidates = list(pool)

        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()

        if self.scale:
            # print self.index
            X_candidates = np.copy(X[candidates])
            X_candidates[:,self.index] = model.transform(X_candidates[:,self.index])
            probs = model.predict_proba(X_candidates)
        else:
            probs = model.predict_proba(X[candidates])

        uncerts = np.min(probs, axis=1)
        uis = np.argsort(uncerts)[::-1]

        if self.num_choose_from == None:
            num_choose_from_ = (uncerts > self.threshold).sum()
        else:
            num_choose_from_ = self.num_choose_from

        chosen_from = [candidates[i] for i in uis[:num_choose_from_]]

        listABC = uis[:num_choose_from_]


        if self.method_uncerten == 'unc_1':

            chosen = chosen_from[0]
            chose_result = []
            chose_result.append(chosen)


        elif self.method_uncerten == 'unc_ce':

            neg_evi, pos_evi = model.predict_evidences(X[candidates])
            min_evi = np.min([abs(neg_evi), abs(pos_evi)], axis=0)
            min_evi_choose = min_evi[uis[:num_choose_from_]]
            index_ = np.argsort(min_evi_choose)[::-1]
            index_pool = listABC[index_[0]]
            chosen = candidates[index_pool]
            chose_result = []
            chose_result.append(chosen)
            return chose_result

        elif self.method_uncerten == 'unc_ie':

            neg_evi, pos_evi = model.predict_evidences(X[candidates])
            max_evi = np.max([abs(neg_evi), abs(pos_evi)], axis=0)
            max_evi_choose = max_evi[uis[:num_choose_from_]]
            index_ = np.argsort(max_evi_choose)
            index_pool = listABC[index_[0]]
            chosen = candidates[index_pool]
            chose_result = []
            chose_result.append(chosen)
            return chose_result

        elif self.method_uncerten == 'unc_t':
            chosen = chosen_from[-1]
            chose_result = []
            chose_result.append(chosen)

        elif self.method_uncerten == 'unc_ce_multiply':

            neg_evi, pos_evi = model.predict_evidences(X[candidates])

            multi_evi = np.multiply(abs(neg_evi), abs(pos_evi))
            multi_evi_choose = multi_evi[uis[:num_choose_from_]]
            index_ = np.argsort(multi_evi_choose)[::-1]
            index_pool = listABC[index_[0]]
            chosen = candidates[index_pool]
            chose_result = []
            chose_result.append(chosen)
            return chose_result
                       
        return chose_result


class EGLStrategy(BaseStrategy):

    def __init__(self, seed=0, sub_pool = None):
        super(EGLStrategy, self).__init__(seed=seed)
        self.sub_pool = 10


    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        
        if not self.sub_pool:
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:self.sub_pool]]
            print candidates

        else:
            candidates = list(pool)

        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()

        probs = model.predict_proba(X[candidates])
        probs_positive = probs[:,1]
        EGL_list = []
        for i in range(len(candidates)):
            positive = self.positive_function(probs_positive[i], X[candidates[i]])
            negative = self.negative_function(probs_positive[i], X[candidates[i]])
            EGL_i = self.Expected_Euclidean_Norm(positive, negative, probs_positive[i])
            EGL_list.append(EGL_i)

        ranked = np.argsort(EGL_list)[::-1]

        chosen = [candidates[j] for j in ranked[:k]]

        return chosen

    def positive_function(self, probs_true, feature_vector):
        gradient_vector = []
        intercept_ = -2 * (1 - probs_true) * probs_true * (1 - probs_true)
        gradient_vector.append(intercept_)

        feature_vector = -2 * (1 - probs_true) * probs_true * (1 - probs_true) * feature_vector
        # for i in feature_vector:
        #     gradient_i = -2 * (1 - probs_true) * probs_true * (1 - probs_true) * i
        #     gradient_vector.append(gradient_i)
        feat_vector = feature_vector.tolist()
        gradient_vector.extend(feat_vector)

        return gradient_vector

    def negative_function(self, probs_true, feature_vector):
        gradient_vector = []
        intercept_ = 2 * probs_true * probs_true * (1 - probs_true)
        gradient_vector.append(intercept_)

        feature_vector = 2 * probs_true * probs_true * (1 - probs_true) * feature_vector

        # for i in feature_vector:
        #     gradient_i = 2 * probs_true * probs_true * (1 - probs_true) * i
        #     gradient_vector.append(gradient_i)

        feat_vector = feature_vector.tolist()
        gradient_vector.extend(feat_vector)

        return gradient_vector

    def Expected_Euclidean_Norm(self, positive_feature_vector, negative_feature_vector, probs_true):

        # method1
        # positive_term = 0
        # negative_term = 0
        # for i in positive_feature_vector:
        #     positive_term += math.pow(i, 2)
        #
        # for j in negative_feature_vector:
        #     negative_term += math.pow(j, 2)
        # positive_term = math.sqrt(positive_term)
        # negative_term = math.sqrt(negative_term)

        # method2
        positive_feature_vector = np.asarray(positive_feature_vector)
        negative_feature_vector = np.asarray(negative_feature_vector)
        positive_term = math.sqrt(np.dot(positive_feature_vector.T, positive_feature_vector))
        negative_term = math.sqrt(np.dot(negative_feature_vector.T, negative_feature_vector))

        # method3
        # nrm2, = scipy.linalg.get_blas_funcs(('nrm2',), )
        # positive_term = nrm2(positive_feature_vector)
        # negative_term = nrm2(negative_feature_vector)

        return float(probs_true*positive_term + (1-probs_true) * negative_term)


class EGLStrategy_2(BaseStrategy):
    def __init__(self, seed=0, sub_pool = None, decision_tree_scale = True):

        super(EGLStrategy_2, self).__init__(seed=seed)
        self.sub_pool = None
        self.scale = decision_tree_scale

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        
        if not self.sub_pool:

            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:5000]]

        else:
            candidates = list(pool)

        # print candidates

        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()

        if self.scale:
            # print "Actively Scaling"
            X_candidates = model.transform(X[candidates])
            probs = model.predict_proba(X_candidates)
        else:
            probs = model.predict_proba(X[candidates])

        probs_positive = probs[:,1]
        EGL_list = []

        for i in range(len(candidates)):
            positive = self.positive_function(probs_positive[i], X[candidates[i]])
            negative = self.negative_function(probs_positive[i], X[candidates[i]])
            EGL_i = self.Expected_Euclidean_Norm(positive, negative, probs_positive[i])
            EGL_list.append(EGL_i)


        ranked = np.argsort(EGL_list)[::-1]

        chosen = [candidates[j] for j in ranked[:k]]

        return chosen

    def positive_function(self, probs_true, feature_vector):
        gradient_vector = []
        intercept_ = (1 - probs_true) 
        gradient_vector.append(intercept_)

        feature_vector = (1 - probs_true) * feature_vector

        feat_vector = feature_vector.tolist()
        gradient_vector.extend(feat_vector)

        return gradient_vector

    def negative_function(self, probs_true, feature_vector):
        gradient_vector = []
        intercept_ = - probs_true 
        gradient_vector.append(intercept_)

        feature_vector = - probs_true * feature_vector

        feat_vector = feature_vector.tolist()
        gradient_vector.extend(feat_vector)

        return gradient_vector

    def Expected_Euclidean_Norm(self, positive_feature_vector, negative_feature_vector, probs_true):

        # method2
        positive_feature_vector = np.asarray(positive_feature_vector)
        negative_feature_vector = np.asarray(negative_feature_vector)
        positive_term = math.sqrt(np.dot(positive_feature_vector.T, positive_feature_vector))
        negative_term = math.sqrt(np.dot(negative_feature_vector.T, negative_feature_vector))

        return float(probs_true * positive_term + (1-probs_true) * negative_term)