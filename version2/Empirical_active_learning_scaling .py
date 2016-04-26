__author__ = 'Ping'
import glob
import numpy as np
import scipy.sparse as ss
import argparse
import os
import re
import sys
import csv
from time import time
from zipfile import ZipFile
import matplotlib as plt

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import cross_validation
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from strategies import RandomStrategy, UncStrategy, BootstrapFromEach, EGLStrategy, EGLStrategy_2
from classifiers import TransparentLogisticRegression, TransparentGaussianNB


class LearningCurve(object):
    """Class - run multiple trials or run trials one at a time"""
    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, pick_classifier, bootstrap_size,  step_size, budget, num_trials, scaling_method, non_binary):

        self.all_performances = {}

        rows = int(budget-bootstrap_size)/step_size + 1

        column = num_trials

        result = np.zeros(shape=(rows,column))

        measures = ["accuracy", "auc"]

        for measure in measures:
            self.all_performances[measure] = []

        for t in range(num_trials):

            print "trial", t+1
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, pick_classifier, bootstrap_size,  step_size, budget, t, scaling_method, non_binary)

        return self.all_performances

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, pick_classifier, bootstrap_size,  step_size, budget, t, scaling_method, non_binary):
        """Helper method for running multiple trials."""

        pool = set(range(len(y_pool)))

        trainIndices = []
        bootstrapped = False

        # Choosing strategy
        if al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':

            if scaling_method == "InformationGainScaling":
                print "This is InformationGainScaling."
                Flag = True
            else:
                Flag = False

            active_s = UncStrategy(seed=t, decision_tree_scale=Flag, index = non_binary)
            print "The uncertain method is ", active_s.method_uncerten

        elif al_strategy == 'egl':
            active_s = EGLStrategy(seed=t)
        elif al_strategy == 'egl_2':
            active_s = EGLStrategy_2(seed=t)
            if active_s.scale:
                print "Active IG scaling"

        accuracyInTrail = []
        aucInTrail = []

        while len(trainIndices) < budget and len(pool) >= step_size:

            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model_pick_classifier, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)

            # print model_pick_classifier
            model_pick_classifier = eval(pick_classifier)

            if scaling_method == 'InformationGainScaling':

                x_pool_train = np.copy(X_pool[trainIndices])
                x_pool_train[:,non_binary] = model_pick_classifier.fit_transform(x_pool_train[:,non_binary], y_pool[trainIndices])

                X_test_ = np.copy(X_test)
                X_test_[:,non_binary] = model_pick_classifier.transform(X_test[:,non_binary])

            else:
                x_pool_train = X_pool[trainIndices]
                X_test_ = X_test

            model_pick_classifier.fit(x_pool_train, y_pool[trainIndices])

            y_probas = model_pick_classifier.predict_proba(X_test_)

            y_pred = model_pick_classifier.predict(X_test_)

            temp_accuracy = accuracy_score(y_test,y_pred)

            temp_auc = metrics.roc_auc_score(y_test, y_probas[:,1])

            accuracyInTrail.append(temp_accuracy)
            aucInTrail.append(temp_auc)
            # print model_pick_classifier.coef_

        self.all_performances["accuracy"].append(accuracyInTrail)
        self.all_performances["auc"].append(aucInTrail)

        # Transparency_model_features(model_pick_classifier, True)
        # Transparency_prediction(model_pick_classifier, X_test_, y_test)
        # loss_penalty(model_pick_classifier, x_pool_train, y_pool[trainIndices])

def Transparency_model_features(model, IG_flag):

    col = len(model.coef_[0])

    if IG_flag:
        result = np.zeros((col, 6))
        result[:,0] = model.mns
        result[:,1:3] = model.count_entropy_larger
        result[:,3:5] = model.count_entropy_smaller

        # coef_intercept = list(model.intercept_)
        coef_intercept = (list(model.coef_[0]))
        result[:,5] = coef_intercept

    else:
        result = list(model.coef_[0])

    np.savetxt("Transparency_model_features_IG.csv", result, delimiter= ',', fmt='%10.10f')

def loss_penalty(model, X_train, y_train):

    loss_list_ig = []
    num = len(y_train)

    y_train[y_train==0] = -1

    for i in range(num):
        temp = X_train[i,:]
        sum_ = np.dot(model.coef_,temp) + model.intercept_
        # print sum_
        aaa = y_train[i] * sum_
        bbb = np.exp(-aaa) + 1
        loss_list_ig.append(np.log(bbb))

    #print loss_list_ig

    print "This is the log_loss of IG"

    # print loss_list_ig
    print np.sum(loss_list_ig)
    penalty = np.dot(model.coef_, model.coef_.T) + np.square(model.intercept_)
    print penalty[0][0]

def Transparency_prediction(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)
    neg_evi, pos_evi = model.predict_evidences(X_test)

    result = np.zeros((len(y_test),5))
    result[:,0] = y_test
    result[:,1] = neg_evi
    result[:,2] = pos_evi
    result[:,3:5] = y_prob

    np.savetxt("Transparency_prediction_IG.csv", result, delimiter= ',', fmt='%10.10f')

def get_classifier(classifier, argus):
    result = classifier + '(' + argus + ')'
    return result

def load_data(dataset1, dataset2=None, make_dense=False):
    """Loads the dataset(s).
    Can handle zip files.
    If the data file extension is csv, it reads a csv file.
    Then, the last column is treated as the target variable.
    Otherwise, the data files are assumed to be in svmlight/libsvm format.

    **Parameters**

    * dataset1 (*str*) - Path to the file of the first dataset.
    * dataset2 (*str or None*) - If not None, path to the file of second dataset
    * make_dense (*boolean*) - Whether to return dense matrices instead of sparse ones (Note: data from csv files will always be treated as dense)

    **Returns**

    * (X_pool, X_test, y_pool, y_test) - Pool and test files if two files are provided
    * (X, y) - The single dataset

    """
    def _get_extensions(dataset1, dataset2):
        first_extension = dataset1[dataset1.rfind('.')+1:]
        second_extension = None
        if dataset2 is not None:
            second_extension = dataset2[dataset2.rfind('.')+1:]

        return first_extension, second_extension

    # Test if these are zipped files

    fe, se = _get_extensions(dataset1, dataset2)

    if se and fe != se:
        raise ValueError("Cannot mix and match different file formats")

    iz_zip = fe == 'zip'

    # Open the files and test if these are csv
    dataset1_file = None
    dataset2_file = None
    is_csv = False

    if iz_zip:
        my_zip_dataset1 = ZipFile(dataset1)
        inside_zip_dataset1 = my_zip_dataset1.namelist()[0] # Assuming each zip contains a single file
        inside_zip_dataset2 = None
        dataset1_file = my_zip_dataset1.open(inside_zip_dataset1)
        if dataset2 is not None:
            my_zip_dataset2 = ZipFile(dataset2)
            inside_zip_dataset2 = my_zip_dataset2.namelist()[0] # Assuming each zip contains a single file
            dataset2_file = my_zip_dataset2.open(inside_zip_dataset2)
        inside_fe, inside_se = _get_extensions(inside_zip_dataset1, inside_zip_dataset2)
        if inside_se and inside_fe != inside_se:
            raise ValueError("Cannot mix and match different file formats")

        is_csv = inside_fe == 'csv'
    else:

        dataset1_file = open(dataset1, 'r')
        if dataset2 is not None:
            dataset2_file = open(dataset2, 'r')

        is_csv = fe == 'csv'

    if dataset2 is not None:
        if is_csv:
            X_pool, y_pool = load_csv(dataset1_file)
            X_test, y_test = load_csv(dataset2_file)
        else:
            X_pool, y_pool = load_svmlight_file(dataset1_file)
            _, num_feat = X_pool.shape
            X_test, y_test = load_svmlight_file(dataset2_file, n_features=num_feat)
            if make_dense:
                X_pool = X_pool.todense()
                X_test = X_test.todense()

        le = LabelEncoder()
        y_pool = le.fit_transform(y_pool)
        y_test = le.transform(y_test)

        dataset1_file.close()
        dataset2_file.close()

        return (X_pool, X_test, y_pool, y_test)

    else:

        if is_csv:
            X, y = load_csv(dataset1_file)
        else:
            X, y = load_svmlight_file(dataset1_file)
            if make_dense:
                X = X.todense()

        le = LabelEncoder()
        y = le.fit_transform(y)

        dataset1_file.close()

        return X, y

def load_csv(dataset_file):
    X = []
    y = []
    csvreader = csv.reader(dataset_file, delimiter=',')
    next(csvreader, None)# skip names
    for row in csvreader:
        X.append(row[:-1])
        y.append(row[-1])
    X = np.array(X, dtype=float)
    y = np.array(y)
    return X, y


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', default="C:\\Users\\Administrator\\Desktop\\ibn_sina.zip", help='The path to the content file.')

    parser.add_argument('-filename', default="ibn_sina_IG", help='The name of file.')

    parser.add_argument('-feature_scaling', choices=['Original','Standard_Scaling','InformationGainScaling'] ,default='InformationGainScaling')

    # parser.add_argument('-decision_tree_scaling', default=False)

    parser.add_argument('-picking_classifier', choices=['LogisticRegression','SVC', 'GaussianNB'], default='TransparentLogisticRegression',
                        help='The underlying classifier.')

    parser.add_argument("--picking_arguments", default=["C=1"],
                        help="Represents the arguments that will be passed to the classifier (default: '').")

    parser.add_argument("-nt", "--num_trials", type=int, default=5, help="Number of trials (default: 10).")

    parser.add_argument("-st", "--strategies", choices=['qbc', 'rand','unc','egl','egl_2'], nargs='*',default='unc',
                        help="Represent a list of strategies for choosing next samples (default: unc).")

    parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                        help='Sets the Boot strap (default: 10).')

    parser.add_argument("-b", '--budget', default=1000, type=int,
                        help='Sets the budget (default: 2000).')

    parser.add_argument("-sz", '--stepsize', default=1, type=int,
                        help='Sets the step size (default: 10).')

    parser.add_argument("-cv", type=int, default=5,
                        help="Number of folds for cross validation. "
                             "Works only if a single dataset is loaded (default: 10).")

    args = parser.parse_args()


    X, y = load_data(args.path)


    num_features = X.shape[1]
    non_binary = []
    binary = []

    for i in range(num_features):
        if len(np.unique(X[:, i])) != 2:
            non_binary.append(i)
        else:
            binary.append(i)

    if len(binary) > 0:
        print "binary features exist"
        X_b = X[:, binary]
        X_b[X_b == 0] = -1
        X[:, binary] = X_b

    skf = StratifiedKFold(y, n_folds=args.cv, shuffle=True, random_state=42)

    row = int((args.budget-args.bootstrap)/args.stepsize + 1)

    file_folder = './' + args.filename + '_' + args.strategies + '/'

    if not os.path.exists(file_folder):
        os.mkdir(file_folder)

    for i in args.picking_arguments:
        if len(binary) == 0:
            print "There is no binary features."
        else:
            print "It has binary features."
            print binary

        print "################################################################"
        print "The arguments of this test:"
        print "     Dataset: %s" % args.filename
        print "     Classifier: %s, picking: %s" %(args.picking_classifier, i)
        print "     Strategy: %s " % args.strategies
        print "     budget: %s    bootstrap: %s " % (args.budget, args.bootstrap)
        print "     stepsize: %s  " % args.stepsize
        print "################################################################"

        clf = get_classifier(args.picking_classifier, i)
        print clf

        cross_vali_number = 1

        for pool, test in skf:

            if args.feature_scaling == 'Standard_Scaling':
                print "This is Standard_Scaling."
                if len(non_binary) > 0:
                    X[:, non_binary] = scale(X[:, non_binary])

            learning_api = LearningCurve()
            temp_average_result = learning_api.run_trials(X[pool], y[pool], X[test], y[test], args.strategies, clf, args.bootstrap, args.stepsize, args.budget, args.num_trials, args.feature_scaling, non_binary)

            for key in temp_average_result:
                one_measure = np.array(temp_average_result[key])
                file_name = file_folder + key + '_' +args.strategies +str(cross_vali_number) + '.csv'
                np.savetxt(file_name, one_measure.T, delimiter=",",fmt='%10.10f')

            cross_vali_number += 1

        for key in temp_average_result:
            average = np.zeros((row, args.num_trials))
            for k in range(args.cv):
                file_name = file_folder + key + '_'+ args.strategies + str(k+1) + '.csv'
                temp = np.loadtxt(file_name, dtype='float32', delimiter=',')
                average = average + temp
            average_ = average / float(args.cv)
            file_average_name = file_folder + key + '_' + args.strategies + i +'_average' + '.csv'
            np.savetxt(file_average_name, average_, delimiter=",", fmt='%10.10f')

if __name__ == '__main__':
    main()





