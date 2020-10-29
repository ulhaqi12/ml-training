import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Model:

    def __init__(self, input_size, bias=np.random.randn()):
        """
        Constructor to create one model object that will perform Multivariate linear regression
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

        y = np.matmul(x.to_numpy(), self.w).reshape(x.shape[0],)
        return y

    def optimize(self, x, y, outputs, learning_rate=0.005):
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


def calculate_cost(y, output):
    """
    function that will calculate cost/error. It will use Mean Square Error(MSE) as it is simple regression problem
    :param y: expected output
    :param output: predicted output
    :return cost: calculated cost or error
    """

    cost = np.sum(np.square(output - y))/(2 * len(y))
    return cost


def pre_processing_of_data(input_data, column, extend_count=0):
    """
    function will perform pre processing of data
    :param input_data: input dataframe
    :return results: preprocessed data
    """

    for i in range(2, extend_count+2):
        input_data.loc[:, column+str(i)] = input_data.loc[:, column] ** i

    input_data = normalize(input_data, columns=input_data.columns)

    print(input_data.head())
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


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    limits = np.array(axes.get_xlim())
    x_vals = np.linspace(limits[0], limits[1], 100)
    y_vals = intercept + ((slope[0][0] * x_vals) + (slope[1][0] * x_vals**2) + (slope[2][0] * x_vals**3))
    plt.plot(x_vals, y_vals, color='red')


def plot_line(x, y, model):
    """
    function that will convert all data and weights in 2d using PCA and plot to get a 2d visualization.
    :param x: input data that will be converted to 1D will be on x axis
    :param y: output will be on y axis
    :param model: model object having weight and bias
    :return:
    """

    plt.scatter(x, y)
    plt.xlabel('Area')
    plt.ylabel('Price of house')
    abline(model.w, model.b)
    plt.show()


def r_squared_score(y, outputs):
    """
    function that will calculate R Squared score
    :param y: expected output
    :param outputs: predicted output
    :return:
    """

    y_avg = np.average(y)
    var_y = np.sum(np.square(outputs - y_avg))
    mse = np.sum(np.square(outputs - y))
    r_squared = 1 - (mse / var_y)
    return r_squared


if __name__ == '__main__':
    data = pd.read_csv('housing.csv')

    y = normalize(data, columns=['price'])['price']
    x = pd.DataFrame(data['area'])
    preprocessed_x = pre_processing_of_data(x, 'area', 2)

    epochs = 500
    model = Model(input_size=3)

    for i in range(epochs):
        outputs = model.predict(preprocessed_x)
        cost = calculate_cost(y, outputs)
        if i % 100 == 0:
            print(i, cost)
        model.optimize(preprocessed_x, y, outputs)

    print('R-Squared score: ', r_squared_score(y, outputs))
    print(y)
    print(preprocessed_x['area'])
    plot_line(preprocessed_x['area'], y, model)
