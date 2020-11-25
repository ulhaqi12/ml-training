import pandas as pd
import numpy as np


class TreeNode:
    """
    class that will represent one node of decision tree
    """

    def __init__(self, dividing_by='', class_name=None, feature_value='', children=[], data=None, Y=None):
        """
        constructor to create one Node of tree
        """

        self.dividing_by = dividing_by
        self.class_name = class_name
        self.feature_value = feature_value
        self.children = children
        self.data = data
        self.output = Y


class DecisionTreeClassifier:
    """
    class that will handle Tree to perform decisions
    """

    def __init__(self):
        """
        constructor to create Decision Tree model
        """

        self.root = None

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
        :param X: input data
        :param Y: expected output
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
        :param X: input data
        :param Y: expected output
        :return:
        """

        self.root = TreeNode(children=[])
        self.root.data = X
        self.root.output = Y
        pending_nodes = [self.root]
        while len(pending_nodes) != 0:
            node = pending_nodes.pop(0)
            best_feature = self.get_best_feature_for_division(node.data, node.output)
            best_feature_values = node.data[best_feature].unique()
            if node.class_name:
                continue
            node.dividing_by = best_feature
            for feature_value in best_feature_values:
                new_node = TreeNode(children=[])
                new_node.feature_value = feature_value
                new_node.data = node.data[node.data[best_feature] == feature_value]
                new_node.output = node.output[node.data[best_feature] == feature_value]
                node.children.append(new_node)
                if len(set(new_node.output)) == 1:
                    new_node.class_name = new_node.output.unique()[0]
                    new_node.children = None
                else:
                    pending_nodes.append(new_node)

    def predict(self, X):
        """
        function that will use trained decision tree to predict output for new example
        :param X: input data
        :return:
        """

        output = None
        node = self.root
        counter = 0
        while output is None:
            if node.class_name:
                return node.class_name
            root_feature = X[node.dividing_by]
            for child in node.children:
                if child.feature_value == root_feature:
                    node = child
            counter += 1
            if counter ==3:
                break

    def print_tree(self, node):
        if node.class_name:
            return
        else:
            print(node.dividing_by)
            print([node.feature_value for node in node.children])
            for child in node.children:
                self.print_tree(child)


if __name__ == '__main__':
    data = pd.read_csv('../tennis_data.csv')

    x = data.drop(['Decision'], axis=1)
    y = data['Decision']

    model = DecisionTreeClassifier()
    model.fit(x, y)

    print(x.loc[0])
    print('Expected output', y.loc[0])
    print('Predicted output', model.predict(x.loc[0]))
