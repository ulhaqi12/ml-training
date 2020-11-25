import pandas as pd
import numpy as np


class Model:

    def __init__(self, input_size, bias=np.random.randn()):
        """
        Constructor to create one model object that will perform Logistic regression
        :param input_size: number of features for each input example
        :param bias: value of bias term
        """

        self.w = np.random.rand(input_size, 1)
        self.b = bias

    def predict(self, x):
        """
        function that will take input x and compute output y.
        :param x: input data
        :return y: predicted output
        """

        y = sigmoid(np.matmul(x.to_numpy(), self.w).reshape(x.shape[0],))
        return y

    def optimize(self, x, y, outputs, learning_rate=0.05):
        """
        function that will perform one optimization step.
        :param x: input data
        :param y: expected outputs
        :param outputs: predicted outputs
        :param learning_rate: learning rate that will decide how fast tis model will learn
        :return:
        """

        self.b = self.b - (learning_rate * (np.sum(outputs - y)/len(y)))
        self.w = self.w - (learning_rate * (np.matmul(x.to_numpy().T, outputs - y)/len(y))).reshape(x.shape[1], 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_cost(y, output):
    """
    function that will calculate cost/error. It will use Cross Entropy Loss as it is simple classification problem
    :param y: expected output
    :param output: predicted output
    :return cost: calculated cost or error
    """

    cost = - np.sum((y * np.log(output)) + ((1-y) * np.log(1-output)))/y.shape[0]
    return cost


def pre_processing_of_data(input_data):
    """
    function will perform pre processing of data
    :param input_data: input dataframe
    :return results: preprocessed data
    """

    input_data = normalize(input_data, columns=['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'thal'])
    return input_data


def normalize(input_data, columns):
    """
    function tha will normalize dataset
    :param input_data: dataframe of which column we are going to normalize
    :param columns: names of column for which normalization is required
    :return normalized_data: return data with columns normalized
    """

    for column in columns:
        min_value = min(input_data.loc[:, column])
        max_value = max(input_data.loc[:, column])
        for i in range(len(input_data)):
            input_data.loc[i, column] = (float(float(input_data.loc[i, column] - min_value) /
                                               float(max_value - min_value)))

    return input_data


def train_test_split(data, split_ratio):
    """
    function that will split input data based on split_ratio
    :param data:
    :param split_ratio:
    :return:
    """

    training_data = data.iloc[:int((data.shape[0]*split_ratio)), :]
    test_data = data.iloc[int(data.shape[0]*split_ratio):, :]
    train_x = training_data.drop(['target'], axis=1)
    train_y = training_data['target']

    test_x = test_data.drop(['target'], axis=1)
    test_y = test_data['target']

    print('Training Examples: ', train_x.shape[0])
    print('Testing Examples: ', test_x.shape[0])

    return train_x, train_y, test_x, test_y


def calculate_accuracy(outputs, y):
    for i in range(len(outputs)):
        if outputs[i] >= 0.5:
            outputs[i] = 1
        else:
            outputs[i] = 0

    count = 0
    for output, expected in zip(outputs, y):
        if output == expected:
            count += 1
    return count/len(outputs)


if __name__ == '__main__':
    data = pd.read_csv('../heart.csv')
    preprocessed_data = pre_processing_of_data(data)

    train_x, train_y, test_x, test_y = train_test_split(preprocessed_data, 0.95)
    model = Model(train_x.shape[1])
    epochs = 1000

    for i in range(epochs):
        outputs = model.predict(train_x)
        cost = calculate_cost(train_y, outputs)
        model.optimize(train_x, train_y, outputs)
        if i % 100 == 0:
            print('epoch ', i, '/', epochs, ': accuracy: ', calculate_accuracy(outputs, train_y))

    test_outputs = model.predict(test_x)
    print('Test Accuracy: ', calculate_accuracy(test_outputs, test_y))
