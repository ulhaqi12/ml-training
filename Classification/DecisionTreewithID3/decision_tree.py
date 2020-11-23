import pandas as pd
import numpy as np


class TreeNode:
    """
    class that will represent one node of decision tree
    """

    def __init__(self):
        """
        constructor to create one Node of tree
        """

        self.dividing_by = ''
        self.class_name = ''
        self.feature_value = ''
        self.children = []


class DecisionTreeClassifier:
    """
    class that will handle Tree to perform decisions
    """

    def __init__(self):
        """
        constructor to create Tree
        """

        self.root = TreeNode()

    @staticmethod
    def calculate_entropy(X, Y):
        """
        function that will calculate entropy of data based on output.
        :param X: input dataset
        :param Y: expected output
        :return:
        """

        entropy = 0
        classes = Y.unique()
        for class_ in classes:
            entropy -= len(y[y == class_])/len(y) * np.log2(len(y[y == class_])/len(y))
        return entropy

    def calculate_information_gain(self, X, Y, feature):
        """
        function that will calculate information gain when dividing data with specific feature.
        :param X: input data
        :param Y: predicted output
        :param feature: feature for which we are supposed to divide data
        :return:
        """

        feature_unique_values = X[feature].unique()
        output_classes = Y.unique()
        feature_entropy = 0
        total_entropy = self.calculate_entropy(X, Y)

        for feature_value in feature_unique_values:
            entr_val = 0
            for class_ in output_classes:
                if len(X[(X[feature] == feature_value) & (Y == class_)]) != 0:
                    prob_value = len(X[(X[feature] == feature_value) & (Y == class_)])/len(X[X[feature] == feature_value])
                    entr_val -= (prob_value * np.log2(prob_value))
            feature_entropy += (len(X[X[feature] == feature_value])/len(Y)) * entr_val
        return total_entropy - feature_entropy

    def get_best_feature_for_division(self, X, Y):
        """
        function that will return feature name having highest Information Gain.
        :param X:
        :param Y:
        :return:
        """
        features = X.columns
        information_gain = {}

        for feature in features:
            information_gain[feature] = self.calculate_information_gain(X, Y, feature)
        return max(information_gain, key=information_gain.get)

    def fit(self, X, Y):
        """
        function that will create a decision tree based on given training data
        :param X:
        :param Y:
        :return:
        """

        best_feature = self.get_best_feature_for_division(X, Y)
        print(best_feature)


if __name__ == '__main__':
    data = pd.read_csv('../tennis_data.csv')

    x = data.drop(['Decision'], axis=1)
    y = data['Decision']

    model = DecisionTreeClassifier()
    model.fit(x, y)
