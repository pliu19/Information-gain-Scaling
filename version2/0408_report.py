import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split, ShuffleSplit
from classifiers import TransparentLogisticRegression, TransparentLinearRegression
from matplotlib import pylab as pl
from scipy.sparse import diags
from IPython import display
from scale import decision_tree_scale
from IPython.display import display, HTML
from ipy_table import *

# define the probability of False(Could be only Bias or summation)
def prob(x):
    return np.exp(-x)/(1+np.exp(-x))

dataset = "breast-w.csv"
class_index = 9
num_cols = 10
classes = ['benign', 'malignant']
read_cols = [i for i in range(num_cols) if i != class_index]
file_path = "D:\\IIT_Master\\2016 Spring\\CS597\\uci\\uci\\uci-tar\\nominal\\"+dataset

# dataset = "diabetes.csv"
# class_index = 8
# num_cols = 9
# classes= ['tested_negative', 'tested_positive']
# read_cols = [i for i in range(num_cols) if i != class_index]
# file_path = "D:\\IIT_Master\\2016 Spring\\CS597\\uci\\uci\\uci-tar\\nominal\\"+dataset

with open(file_path, 'r') as f:
    header = f.readline()
    header = np.array(header.split(','))
    feature_names = header[read_cols]
print "The features are: ", feature_names


X = np.loadtxt(file_path, dtype=float, delimiter=",", skiprows=1, \
                   usecols=read_cols)
y = np.loadtxt(file_path, dtype=int, delimiter=",", skiprows=1, \
               usecols=(class_index,), converters={class_index: lambda x: classes.index(x)})


num_inst, num_feat = np.shape(X)
print "The shape of this data set:",np.shape(X)
print "The # of positive instances in all data: ", np.sum(y)
print "The ratio of positive instances: ", np.sum(y)/float(num_inst)
print "The ratio of negative instances: ", 1-(np.sum(y)/float(len(y)))
print ""

ss = ShuffleSplit(num_inst, n_iter=1, test_size=0.33, random_state=2)

for i, j in ss:
    train_index = i
    test_index = j

print "Y_train total:", len(y[train_index])
print "Y_train positive:", np.sum(y[train_index])
print "Ratio of positive in train instances", np.sum(y[train_index])/float(len(y[train_index]))

clf_ori = TransparentLogisticRegression()
clf_ss = TransparentLogisticRegression()
clf_ig = TransparentLogisticRegression()


X_ss = scale(X)
scale_ = decision_tree_scale()

X_ig = scale_.fit_transform(X, y)

clf_ori.fit(X, y)
clf_ss.fit(X_ss, y)
clf_ig.fit(X_ig, y)

print clf_ori.coef_
print clf_ss.coef_
print clf_ig.coef_

bias_ori = clf_ori.intercept_
bias_ss = clf_ss.intercept_
bias_ig = clf_ig.intercept_

print bias_ori, bias_ss, bias_ig

print prob(bias_ori)
print prob(bias_ss)
print prob(bias_ig)

for  i in range(num_feat):
    X_current = X_ig[:,i]
    X_current = np.reshape(X_current, (-1, 1))
    clf = TransparentLogisticRegression(fit_intercept =False)
    clf.fit(X_current, y)
    print feature_names[i], clf.coef_[0][0]

# print prob(-5.66), prob(-1.18), prob(-0.24)
