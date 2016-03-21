import numpy as np
from scipy import stats


class decision_tree_scale():

    def __init__(self):

        pass

    def fit_transform(self, X, y, axis = 0):
        # m is # of instances.

        X = np.asarray(X)
        y = np.asarray(y)

        # if X.ndim == 1:
        #
        #     n = np.shape(X)[0]
        #     m = 1
        #
        # else:
        #
        #     m, n = np.shape(X)

        m, n = np.shape(X)

        m = float(m)

        y = 1.0 * y
        # print set(y)
        feature_split_value = []

        self.count_entropy_larger = []
        self.count_entropy_smaller = []

        for i in range(n):

            current_feature = X[:,i]
            sorted = np.unique(current_feature)

            if len(sorted) == 1:

                feature_split_value.append(sorted[0])

            else:

                interval = (sorted[1:] + sorted[:-1]) / 2.

                entropy_list = []

                for j in interval:

                    index_1 = np.where(current_feature >= j)[0]
                    index_2 = np.where(current_feature < j)[0]

                    part1_y = y[index_1]
                    part2_y = y[index_2]

                    #print len(y[index_1]), len(y[index_2]),len(y)
                    proba_part1 = [sum(part1_y)/len(part1_y), 1-sum(part1_y)/len(part1_y)]
                    proba_part2 = [sum(part2_y)/len(part2_y), 1-sum(part2_y)/len(part2_y)]

                    expected_entropy = len(part1_y)/m * stats.entropy(proba_part1) + \
                        len(part2_y)/m * stats.entropy(proba_part2)

                    entropy_list.append(expected_entropy)

                temp_feature_max = interval[np.argmin(entropy_list)]

                index_1_ = np.where(current_feature >= temp_feature_max)[0]
                index_2_ = np.where(current_feature < temp_feature_max)[0]

                part1_y_ = y[index_1_]
                part2_y_ = y[index_2_]

                count_part1_ = (sum(part1_y_), len(part1_y_)-sum(part1_y_))
                count_part2_ = (sum(part2_y_), len(part2_y_)-sum(part2_y_))

                self.count_entropy_larger.append(count_part1_)
                self.count_entropy_smaller.append(count_part2_)

                feature_split_value.append(temp_feature_max)

        feature_split_std = []

        scale_ = X - feature_split_value

        for i in range(n):
            # current = X[:, i] - feature_split_value[i]
            std = np.sqrt(np.sum(scale_[:, i]**2)/m)

            if std == 0:
                std = 1

            feature_split_std.append(std)

        # print "Splitting value: ",feature_split_value

        self.mns = np.asarray(feature_split_value)

        self.sstd = np.asarray(feature_split_std)

        # print self.mns

        if axis and self.mns.ndim < X.ndim:

            return ((X - np.expand_dims(self.mns, axis=axis)) /
                    np.expand_dims(self.sstd, axis=axis))
        else:
            # print "True"
            return (X - self.mns) / self.sstd

    def transform(self, X, axis=0):

        X = np.asanyarray(X)

        if axis and self.mns.ndim < X.ndim:

            return ((X - np.expand_dims(self.mns, axis=axis)) /
                    np.expand_dims(self.sstd, axis=axis))
        else:

            return (X - self.mns) / self.sstd


# a = np.array([[1,2,3],[4,5,6]])
# b = [1, 0]
#
# s = decision_tree_scale()
# a_ = s.fit_transform(a, b)
#
# print a_


