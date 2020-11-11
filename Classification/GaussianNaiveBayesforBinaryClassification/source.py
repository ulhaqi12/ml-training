from math import pi

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


class GaussianNaiveBayes:
    """
    class that will handle the model of naive bayes using gaussian distribution
    """

    def __init__(self):
        """
        constructor that will create object of Naive bayes classifier using gaussian distribution
        """

        self.priors = {}
        self.coefficients = {}

    def fit(self, x, y):
        """
        function will take data and will train model according to input data
        :param x:
        :return:
        """

        for class_name in y.unique():
            self.priors[class_name] = len(y[y == class_name])/len(y)

        for class_name in y.unique():
            self.coefficients[class_name] = {}
            features_in_class = x[y == class_name]
            for column in features_in_class.columns:
                self.coefficients[class_name][column] = (np.average(features_in_class[column]),
                                                  np.std(features_in_class[column]))

    def get_likelihood(self, x, feature_name, class_name):
        """
        function that will calculate likelihood of a feature of new data with model
        :param x: new input data
        :param feature_name: name of feature for which likelihood is to be calculated
        :param class_name: name of class
        :return:
        """

        average, std = self.coefficients[class_name][feature_name]
        return (1 / np.sqrt(2 * pi * (std ** 2))) * np.exp((-((x - average) ** 2)/(2 * (std ** 2))))

    def predict(self, x):
        """
        function that will predict output based on input x
        :param x:
        :return:
        """

        posteriors = {}

        for class_name in self.coefficients:
            p = 1
            for feature_name in self.coefficients[class_name]:
                p = p * (self.get_likelihood(x[feature_name], feature_name, class_name)) * self.priors[class_name]
            posteriors[class_name] = p

        y = []
        for i in range(len(x)):
            max_prob = -1
            max_class = ''
            for class_name in self.coefficients:
                if posteriors[class_name][i] > max_prob:
                    max_prob = posteriors[class_name][i]
                    max_class = class_name
            y.append(max_class)

        return y


def preprocessing_of_data(input_data):
    """
    function that will perform preprocessing of input data
    :param input_data: input data that need to be pre processed
    :return:
    """

    for i in range(len(input_data)):
        input_data.loc[i, 'Gender'] = 1 if input_data.loc[i, 'Gender'] == 'Male' else 0

    return input_data


def train_test_split(data, split_ratio):
    """
    function that will split input data based on split_ratio
    :param data: dataset
    :param split_ratio: ratio of division in train and test sets
    :return:
    """

    training_data = data.iloc[:int((data.shape[0]*split_ratio)), :]
    test_data = data.iloc[int(data.shape[0]*split_ratio):, :]
    train_x = training_data.drop(['Gender'], axis=1)
    train_y = training_data['Gender']

    test_x = (test_data.drop(['Gender'], axis=1)).reset_index().drop(['index'], axis=1)
    test_y = (test_data['Gender']).reset_index().drop(['index'], axis=1)

    print('Training Examples: ', train_x.shape[0])
    print('Testing Examples: ', test_x.shape[0])

    return train_x, train_y, test_x, test_y


def calculate_accuracy(outputs, y):
    count = 0
    for output, expected in zip(list(outputs), list(y)):
        if output == expected:
            count += 1
    return count/len(outputs)


if __name__ == '__main__':
    data = pd.read_csv('../weight-height.csv')
    preprocessed_data = preprocessing_of_data(data)

    train_x, train_y, test_x, test_y = train_test_split(preprocessed_data, 0.95)

    model = GaussianNaiveBayes()
    model.fit(train_x, train_y)
    output = model.predict(test_x)
    print('Test Accuracy by my model: ', calculate_accuracy(output, test_y['Gender']) * 100)

    sk_nb = GaussianNB()
    sk_nb.fit(train_x.to_numpy(), np.array(train_y, dtype=np.float64))
    output_sk = sk_nb.predict(test_x.to_numpy())
    print("Test accuracy in sklearn: ", calculate_accuracy(output_sk, test_y['Gender']) * 100)
