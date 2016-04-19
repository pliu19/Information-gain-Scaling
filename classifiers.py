'''
Created on Sep 10, 2015

@author: Mustafa
'''
from abc import abstractmethod
from scipy.sparse import diags, issparse
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.utils.validation import check_array
import numpy as np
from scipy import stats

from scale import decision_tree_scale

class TransparentModel(object):
    
    @abstractmethod
    def predict_evidences(self, X):
        """Compute evidences for class 0 and class 1 and return them"""
    
    @abstractmethod
    def get_weights(self):
        """Return the weights from class 1's perspective.
        Negative weight contributes to class 0,
        positive weight contributes to class 1"""
    
    @abstractmethod
    def get_bias(self):
        """Return the class bias, if any, from class 1's perspective.
        Negative value contributes to class 0,
        positive value contributes to class 1"""

def compute_evidences_nonnegative_matrix(weights, X):   
    X = check_array(X, accept_sparse="csr")
    neg_weights = weights * (weights < 0)
    pos_weights = weights * (weights > 0)
    if issparse(X):
        neg_evi = X * neg_weights
        pos_evi = X * pos_weights
    else:
        neg_evi = np.dot(X, neg_weights)
        pos_evi = np.dot(X, pos_weights)
    
    return neg_evi, pos_evi

def compute_evidences(weights, X):
    X = check_array(X, accept_sparse="csr")
    weights_diags = diags(weights, 0)
    dm = X * weights_diags
    if issparse(dm):
        pos_evi = dm.multiply(dm > 0).sum(1).A1
        neg_evi = dm.multiply(dm < 0).sum(1).A1
    else:
        pos_evi = np.multiply(dm, dm > 0).sum(1)
        neg_evi = np.multiply(dm, dm < 0).sum(1)        
    return neg_evi, pos_evi
    
##############
# REGRESSION #
##############

class TransparentLasso(TransparentModel, Lasso):
    '''
    Transparent Lasso
    '''

    def predict_evidences(self, X):        
        X = check_array(X, accept_sparse="csr")        
        if X.min() > 0:
            return compute_evidences_nonnegative_matrix(self.coef_, X)
        else:
            return compute_evidences(self.coef_, X)
        
    def get_weights(self):
        return self.coef_
    
    def get_bias(self):
        return self.intercept_

class TransparentRidge(TransparentModel, Ridge):
    '''
    Transparent Ridge regression
    '''

    def predict_evidences(self, X):        
        X = check_array(X, accept_sparse="csr")        
        if X.min() > 0:
            return compute_evidences_nonnegative_matrix(self.coef_, X)
        else:
            return compute_evidences(self.coef_, X)
        
    def get_weights(self):
        return self.coef_
    
    def get_bias(self):
        return self.intercept_

class TransparentLinearRegression(TransparentModel, LinearRegression):
    '''
    Transparent linear regression
    '''

    def predict_evidences(self, X):        
        X = check_array(X, accept_sparse="csr")        
        if X.min() > 0:
            return compute_evidences_nonnegative_matrix(self.coef_, X)
        else:
            return compute_evidences(self.coef_, X)
        
    def get_weights(self):
        return self.coef_
    
    def get_bias(self):
        return self.intercept_

class TransparentLinearSVR(TransparentModel, LinearSVR):
    '''
    Transparent linear SVR
    '''

    def predict_evidences(self, X):        
        X = check_array(X, accept_sparse="csr")        
        if X.min() > 0:
            return compute_evidences_nonnegative_matrix(self.coef_, X)
        else:
            return compute_evidences(self.coef_, X)
        
    def get_weights(self):
        return self.coef_
    
    def get_bias(self):
        return self.intercept_
    
##################
# CLASSIFICATION #
##################


class TransparentLogisticRegression(TransparentModel, LogisticRegression):
    '''
    Transparent logistic regression
    '''
    
    def predict_evidences(self, X):
        X = check_array(X, accept_sparse="csr")        
        if X.min() > 0:
            neg_evi, pos_evi = compute_evidences_nonnegative_matrix(self.coef_[0], X)
        else:
            neg_evi, pos_evi = compute_evidences(self.coef_[0], X)

        if self.intercept_ > 0:
            pos_evi = pos_evi + self.intercept_
        else:
            neg_evi = neg_evi + self.intercept_

        return neg_evi, pos_evi
    
    def get_weights(self):
        return self.coef_[0]
    
    def get_bias(self):
        return self.intercept_[0]

    def fit_transform(self, X, y, axis = 0):
        # m is # of instances.

        X = np.asarray(X)
        y = np.asarray(y)

        m, n = np.shape(X)

        m = float(m)

        y = 1.0 * y

        feature_split_value = []

        self.count_entropy_larger = []
        self.count_entropy_smaller = []

        for i in range(n):

            current_feature = X[:,i]
            sorted = np.unique(current_feature)

            if len(sorted) == 1:
                temp_feature_max = sorted[0]
                feature_split_value.append(sorted[0])

            else:

                interval = (sorted[1:] + sorted[:-1]) / 2.

                entropy_list = []

                for j in interval:
                    index_1 = np.where(current_feature >= j)[0]
                    index_2 = np.where(current_feature < j)[0]

                    part1_y = y[index_1]
                    part2_y = y[index_2]

                    proba_part1 = [sum(part1_y)/len(part1_y), 1-sum(part1_y)/len(part1_y)]
                    proba_part2 = [sum(part2_y)/len(part2_y), 1-sum(part2_y)/len(part2_y)]

                    expected_entropy = len(part1_y)/m * stats.entropy(proba_part1) + \
                        len(part2_y)/m * stats.entropy(proba_part2)

                    entropy_list.append(expected_entropy)

                temp_feature_max = interval[np.argmin(entropy_list)]

                feature_split_value.append(temp_feature_max)

            index_1_ = np.where(current_feature >= temp_feature_max)[0]
            index_2_ = np.where(current_feature < temp_feature_max)[0]

            part1_y_ = y[index_1_]
            part2_y_ = y[index_2_]

            count_part1_ = (sum(part1_y_), len(part1_y_)-sum(part1_y_))
            count_part2_ = (sum(part2_y_), len(part2_y_)-sum(part2_y_))

            self.count_entropy_larger.append(count_part1_)
            self.count_entropy_smaller.append(count_part2_)

        feature_split_std = []

        scale_ = X - feature_split_value

        for i in range(n):
            # current = X[:, i] - feature_split_value[i]
            std = np.sqrt(np.sum(scale_[:, i]**2)/m)

            if std == 0:
                std = 1

            feature_split_std.append(std)

        self.mns = np.asarray(feature_split_value)

        self.sstd = np.asarray(feature_split_std)

        # print self.mns

        if axis and self.mns.ndim < X.ndim:

            return ((X - np.expand_dims(self.mns, axis=axis)) /
                    np.expand_dims(self.sstd, axis=axis))
        else:

            return (X - self.mns) / self.sstd

    def transform(self, X, axis=0):

        X = np.asanyarray(X)

        if axis and self.mns.ndim < X.ndim:

            return ((X - np.expand_dims(self.mns, axis=axis)) /
                    np.expand_dims(self.sstd, axis=axis))
        else:

            return (X - self.mns) / self.sstd

class TransparentLinearSVC(TransparentModel, LinearSVC):
    '''
    Transparent Linear SVC
    '''
    
    def predict_evidences(self, X):
        X = check_array(X, accept_sparse="csr")        
        if X.min() > 0:
            return compute_evidences_nonnegative_matrix(self.coef_[0], X)
        else:
            return compute_evidences(self.coef_[0], X)
    
    def get_weights(self):
        return self.coef_[0]
    
    def get_bias(self):
        return self.intercept_[0]

class TransparentMultinomialNB(TransparentModel, MultinomialNB):
    '''
    Transparent multinomial naive Bayes
    '''

    def predict_evidences(self, X):
        X = check_array(X, accept_sparse="csr")
        
        if X.min() < 0:
            raise ValueError("Multinomial naive Bayes cannot be used with negative feature values.")
        
        weights = self.feature_log_prob_[1] - self.feature_log_prob_[0]
        
        return compute_evidences_nonnegative_matrix(weights, X)
    
    def get_weights(self):
        return self.feature_log_prob_[1] - self.feature_log_prob_[0]
    
    def get_bias(self):
        return self.class_log_prior_[1] - self.class_log_prior_[0]

class TransparentBernoulliNB(TransparentModel, BernoulliNB):
    '''
    Transparent Bernoulli naive Bayes
    '''
    
    def predict_evidences(self, X):
        p_neg_evi, p_pos_evi, a_neg_evi, a_pos_evi = self.predict_presence_absence_evidences(X)        
        return p_neg_evi + a_neg_evi, p_pos_evi + a_pos_evi
        
    def predict_presence_absence_evidences(self, X):
        
        X = check_array(X, accept_sparse="csr")
        
        absence_log_prob_ = np.log(1 - np.exp(self.feature_log_prob_))
        
        presence_log_ratios = self.feature_log_prob_[1] - self.feature_log_prob_[0]
        absence_log_ratios = absence_log_prob_[1] - absence_log_prob_[0]
        
        presence_neg_log_ratios = presence_log_ratios * (presence_log_ratios<0)
        presence_pos_log_ratios = presence_log_ratios * (presence_log_ratios>0)
        if issparse(X):
            p_neg_evi = X * presence_neg_log_ratios
            p_pos_evi = X * presence_pos_log_ratios
        else:
            p_neg_evi = np.dot(X, presence_neg_log_ratios)
            p_pos_evi = np.dot(X, presence_pos_log_ratios)
        
        absence_neg_log_ratios = absence_log_ratios * (absence_log_ratios<0)
        absence_pos_log_ratios = absence_log_ratios * (absence_log_ratios>0)
        default_a_neg_evi = absence_neg_log_ratios.sum()
        default_a_pos_evi = absence_pos_log_ratios.sum()
        if issparse(X):
            a_neg_evi = -(X * absence_neg_log_ratios) + default_a_neg_evi
            a_pos_evi = -(X * absence_pos_log_ratios) + default_a_pos_evi
        else:
            a_neg_evi = -np.dot(X, absence_neg_log_ratios) + default_a_neg_evi
            a_pos_evi = -np.dot(X, absence_pos_log_ratios) + default_a_pos_evi
        
        return p_neg_evi, p_pos_evi, a_neg_evi, a_pos_evi
    
    
    def get_bias(self):
        return self.class_log_prior_[1] - self.class_log_prior_[0]
    
    
    def get_weights(self, presence=True):
        if presence:
            return self.feature_log_prob_[1] - self.feature_log_prob_[0]
        else:
            absence_log_prob_ = np.log(1 - np.exp(self.feature_log_prob_))
            return absence_log_prob_[1] - absence_log_prob_[0]

class TransparentGaussianNB(TransparentModel, GaussianNB):
    '''
    Transparent Gaussian naive Bayes
    '''

    def predict_evidences(self, X):
        X = check_array(X, accept_sparse=None) # For now, do not accept sparse
        
        evi1 = -0.5*np.log(self.sigma_[1, :]) 
        evi1 += 0.5*np.log(self.sigma_[0, :])
        evi2 = -((X - self.theta_[1, :]) ** 2)/ (2*self.sigma_[1, :])
        evi2 += ((X - self.theta_[0, :]) ** 2)/ (2*self.sigma_[0, :])
        
        evi = evi1 + evi2
        
        pos_evi = np.sum(evi*(evi > 0), 1)
        neg_evi = np.sum(evi*(evi < 0), 1)
        
        return neg_evi, pos_evi
    
    
    def get_bias(self):        
        return np.log(self.class_prior_[1]) - np.log(self.class_prior_[0])


    def fit_transform(self, X, y, axis=0):
        # m is # of instances.

        X = np.asarray(X)
        y = np.asarray(y)

        m, n = np.shape(X)

        m = float(m)

        y = 1.0 * y

        feature_split_value = []

        self.count_entropy_larger = []
        self.count_entropy_smaller = []

        for i in range(n):

            current_feature = X[:, i]
            sorted = np.unique(current_feature)

            if len(sorted) == 1:
                temp_feature_max = sorted[0]
                feature_split_value.append(sorted[0])

            else:

                interval = (sorted[1:] + sorted[:-1]) / 2.

                entropy_list = []

                for j in interval:
                    index_1 = np.where(current_feature >= j)[0]
                    index_2 = np.where(current_feature < j)[0]

                    part1_y = y[index_1]
                    part2_y = y[index_2]

                    proba_part1 = [sum(part1_y) / len(part1_y), 1 - sum(part1_y) / len(part1_y)]
                    proba_part2 = [sum(part2_y) / len(part2_y), 1 - sum(part2_y) / len(part2_y)]

                    expected_entropy = len(part1_y) / m * stats.entropy(proba_part1) + \
                                       len(part2_y) / m * stats.entropy(proba_part2)

                    entropy_list.append(expected_entropy)

                temp_feature_max = interval[np.argmin(entropy_list)]

                feature_split_value.append(temp_feature_max)

            index_1_ = np.where(current_feature >= temp_feature_max)[0]
            index_2_ = np.where(current_feature < temp_feature_max)[0]

            part1_y_ = y[index_1_]
            part2_y_ = y[index_2_]

            count_part1_ = (sum(part1_y_), len(part1_y_) - sum(part1_y_))
            count_part2_ = (sum(part2_y_), len(part2_y_) - sum(part2_y_))

            self.count_entropy_larger.append(count_part1_)
            self.count_entropy_smaller.append(count_part2_)

        feature_split_std = []

        scale_ = X - feature_split_value

        for i in range(n):
            # current = X[:, i] - feature_split_value[i]
            std = np.sqrt(np.sum(scale_[:, i] ** 2) / m)

            if std == 0:
                std = 1

            feature_split_std.append(std)

        self.mns = np.asarray(feature_split_value)

        self.sstd = np.asarray(feature_split_std)

        # print self.mns

        if axis and self.mns.ndim < X.ndim:

            return ((X - np.expand_dims(self.mns, axis=axis)) /
                    np.expand_dims(self.sstd, axis=axis))
        else:

            return (X - self.mns) / self.sstd


    def transform(self, X, axis=0):
        X = np.asanyarray(X)

        if axis and self.mns.ndim < X.ndim:

            return ((X - np.expand_dims(self.mns, axis=axis)) /
                    np.expand_dims(self.sstd, axis=axis))
        else:

            return (X - self.mns) / self.sstd