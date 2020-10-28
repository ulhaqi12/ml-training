import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Model:

    def __init__(self, weight=np.random.randn(), bias=np.random.randn()):
        """
        Constructor to create one model object that will perform univariate linear regression
        :param weight:
        :param bias:
        """

        self.w = weight
        self.b = bias

    def predict(self, x):
        """
        function that will take input x and compute output y.
        :param x:
        :return y:
        """

        y = self.w * x + self.b
        return y

    def optimize(self, x, y, outputs, learning_rate=0.05):
        """
        function that will perform one optimization step.
        :param x:
        :param y:
        :param outputs:
        :param learning_rate:
        :return:
        """
        self.b = self.b - (learning_rate * (np.sum(outputs - y)/len(y)))
        self.w = self.w - (learning_rate * (np.sum(np.multiply(outputs - y, x))/len(y)))


def r_squared_score(y, outputs):
    """
    function that will calculate R Squared score
    :param actual output -> y:
    :param pridicted output -> outputs:
    :return:
    """
    y_avg = np.average(y)
    var_y = np.sum(np.square(outputs - y_avg))
    mse = np.sum(np.square(outputs - y))
    r_squared = 1 - (mse / var_y)
    return r_squared


def calculate_cost(y, output):
    """
    function that will calculate cost/error. It will use Mean Square Error(MSE) as it is simple regression problem
    :param y:
    :param output:
    :return cost:
    """

    cost = np.sum(np.square(output - y))/(2 * len(y))
    return cost


def normalize_input(x):
    """
    funciton that will return normalized fetures
    :param x:
    :return:
    """
    min_value = min(x)
    max_value = max(x)
    normalized = np.zeros(x.shape)
    for i in range(len(x)):
        normalized[i] = (float(float(x[i] - min_value) / float(max_value - min_value)))
    return normalized


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red')


if __name__ == '__main__':
    data = pd.read_csv('housing.csv')

    x = data['area']
    y = data['price']

    # calculated min and max to un-normalize data later
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    # normalization
    norm_x = normalize_input(x)
    norm_y = normalize_input(y)

    plt.scatter(norm_x, norm_y)
    plt.xlabel('Size')
    plt.ylabel('Price')

    epochs = 3000
    model = Model()

    for i in range(epochs):
        outputs = model.predict(norm_x)
        cost = calculate_cost(norm_y, outputs)
        model.optimize(norm_x, norm_y, outputs)

    denormalized_outputs = outputs * (max_y - min_y) + min_y
    abline(model.w, model.b)
    # plt.show()

    print('R Square score: ', r_squared_score(norm_y, denormalized_outputs))
