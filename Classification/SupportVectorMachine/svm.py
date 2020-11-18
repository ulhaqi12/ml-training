import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

        y = process_output(np.matmul(x.to_numpy(), self.w).reshape(x.shape[0], ) + self.b)
        return y

    def optimize(self, x, y, outputs, learning_rate=0.0001, c=0.1):
        """
        function that will perform one optimization step.
        :param x: input data
        :param y: expected outputs
        :param outputs: predicted outputs
        :param learning_rate: learning rate that will decide how fast tis model will learn
        :return:
        """

        y_ = np.where(outputs <= 0, -1, 1)
        x_np = x.to_numpy()
        for i in range(len(outputs)):
            if y_[i] == 1:
                if (np.dot(x_np[i].reshape(2,), self.w.reshape(2,)) + self.b) >= 1:
                    self.w -= learning_rate * (2 * c * self.w)
                    self.b -= learning_rate * y_[i]
                else:
                    self.w -= learning_rate * (2 * c * self.w - np.dot(x_np[i].reshape(2, 1), y_[i]))
                    self.b -= learning_rate * y_[i]
            elif y_[i] == -1:
                if (np.dot(x_np[i].reshape(2,), self.w.reshape(2,)) + self.b) <= -1:
                    self.w += learning_rate * (2 * c * self.w)
                    self.b += learning_rate * y_[i]
                else:
                    self.w += learning_rate * (2 * c * self.w - np.dot(x_np[i].reshape(2, 1), y_[i]))
                    self.b += learning_rate * y_[i]


def calculate_cost(x, outputs, y, model):
    """
    function that will calculate cost
    :param x: input
    :param y: predicted output
    :param model: ML model object having weights
    :return:
    """

    x_np = x.to_numpy()
    cost = 0
    for i in range(len(y)):
        if y[i] == 1:
            if outputs[i] != y[i]:
                cost += max(0, 1 - (np.dot(model.w.reshape(2,), x_np[i].reshape(2,))))
        elif y[i] == 0:
            if outputs[i] != y[i]:
                cost += max(0, 1 + (np.dot(model.w.reshape(2,), x_np[i].reshape(2,))))

    print(cost)
    cost = (cost / len(y)) + ((np.sum(np.square(model.w))) / len(y))
    return cost


def process_output(y):
    """
    function that will process output of model
    :param y:
    :return:
    """

    for i in range(len(y)):
        if y[i] >= 0:
            y[i] = 1
        elif y[i] <= 0:
            y[i] = 0

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


def plot(x, y, model):
    """
    function that will plot data and visualize classifier
    :param x:
    :param y:
    :param model:
    :return:
    """

    neg_examples = x[y == 0]
    pos_examples = x[y == 1]
    plt.scatter(pos_examples['Height'], pos_examples['Weight'])
    plt.scatter(neg_examples['Height'], neg_examples['Weight'], color='red')

    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = (-(model.w[0] * x_vals) - model.b)/model.w[1]
    plt.plot(x_vals, y_vals, color='green')

    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('../weight_height_small.csv')

    # data_m = data[data['Gender'] == 'Male']
    # data_f = data[data['Gender'] == 'Female']
    #
    # data_m = data_m[data_m['Height'] > 68].reset_index(drop=True)
    # data_m = data_m[data_m['Weight'] > 195].reset_index(drop=True)
    #
    # data_f = data_f[data_f['Height'] < 64].reset_index(drop=True)
    # data_f = data_f[data_f['Weight'] < 190].reset_index(drop=True)
    #
    # data = pd.concat([data_f, data_m], ignore_index=True).reset_index(drop=True)
    #
    # print(data)
    #
    # data.to_csv('new_data.csv', index=False)

    data = preprocessing_of_data(data)

    data = data.sample(frac=1, axis=0).reset_index(drop=True)
    train_x, train_y, test_x, test_y = train_test_split(data, 0.9)

    train_x = normalize(train_x, columns=['Height', 'Weight'])
    test_x = normalize(test_x, columns=['Height', 'Weight'])

    epoch = 300
    model = Model(train_x.shape[1])

    plot(train_x, train_y.to_numpy(), model)

    for i in range(epoch):
        outputs = model.predict(train_x)
        cost = calculate_cost(train_x, outputs, train_y, model)
        print(cost)
        model.optimize(train_x, train_y, outputs)
        print('Epoch', i, ': Train Acc', calculate_accuracy(outputs, train_y.to_numpy()) * 100)

    test_outputs = model.predict(test_x)
    print('Test Accuracy:', calculate_accuracy(test_outputs, test_y.to_numpy()) * 100)
    plot(train_x, train_y.to_numpy(), model)
    print(model.w, model.b)
