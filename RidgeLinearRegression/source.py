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

    def optimize(self, x, y, outputs, learning_rate=0.05):
        """
        function that will perform one optimization step.
        :param x: input data
        :param y: expected outputs
        :param outputs: predicted outputs
        :param learning_rate: learning rate that will decide how fast tis model will learn
        :return:
        """

        lambda_value = 0.001
        lambda_vector = self.w * (2 * lambda_value)
        self.b = self.b - (learning_rate * (np.sum(outputs - y)/len(y)))
        self.w = self.w - (learning_rate * ((np.matmul(x.to_numpy().T, outputs - y)/len(y)).reshape(x.shape[1], 1) + lambda_vector))


def calculate_cost(y, output, model):
    """
    function that will calculate cost/error. It will use Mean Square Error(MSE) as it is simple regression problem
    :param y: expected output
    :param output: predicted output
    :return cost: calculated cost or error
    """

    lambda_ = 0.5
    cost = np.sum(np.square(output - y))/(2 * len(y)) + (lambda_ * np.sum(np.square(model.w)))
    return cost


def one_hot_encoding(input_data, column):
    """
    function that will perform one-hot encoding on specific column
    :param input_data: pandas dataframe of input data
    :param column: column name for which on-hot encoding is required
    :return pandas dataframe having one-hot encoded column:
    """

    unique_values = pd.unique(input_data[column])
    for value in unique_values:
        for i in range(len(input_data)):
            input_data.loc[i, value] = 1 if input_data.loc[i, column] == value else 0
    results = input_data.drop([column], axis=1)
    return results


def pre_processing_of_data(input_data):
    """
    function will perform pre processing of data including on hot encoding etc
    :param input_data: input dataframe
    :return results: preprocessed data
    """

    for i in range(len(input_data)):
        input_data.loc[i, 'mainroad'] = 1 if input_data.loc[i, 'mainroad'] == 'yes' else 0
        input_data.loc[i, 'guestroom'] = 1 if input_data.loc[i, 'guestroom'] == 'yes' else 0
        input_data.loc[i, 'basement'] = 1 if input_data.loc[i, 'basement'] == 'yes' else 0
        input_data.loc[i, 'hotwaterheating'] = 1 if input_data.loc[i, 'hotwaterheating'] == 'yes' else 0
        input_data.loc[i, 'airconditioning'] = 1 if input_data.loc[i, 'airconditioning'] == 'yes' else 0
        input_data.loc[i, 'prefarea'] = 1 if input_data.loc[i, 'prefarea'] == 'yes' else 0

    input_data = one_hot_encoding(input_data, 'furnishingstatus')
    input_data = normalize(input_data, columns=['bedrooms', 'bathrooms', 'stories', 'parking', 'area'])
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
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red')


def plot_line_2d_using_pca(x, y, model):
    """
    function that will convert all data and weights in 2d using PCA and plot to get a 2d visualization.
    :param x: input data that will be converted to 1D will be on x axis
    :param y: output will be on y axis
    :param model: model object having weight and bias
    :return:
    """

    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(x)

    plt.scatter(pca_data, y)
    plt.xlabel('14 different features converted to 1D using PCA')
    plt.ylabel('Price of house')
    new_df = [x, pd.DataFrame(model.w.reshape(1, model.w.shape[0]), columns=x.columns)]
    new_df = pd.concat(new_df, ignore_index=True)

    pca_weights = pca.fit_transform(new_df)[x.shape[0]]
    abline(pca_weights[0], model.b)
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
    data = pd.read_csv('../housing_complete.csv')

    y = data['price']
    x = data.drop(['price'], axis=1)
    preprocessed_x = pre_processing_of_data(x)

    epochs = 3000
    model = Model(input_size=14)

    for i in range(epochs):
        outputs = model.predict(preprocessed_x)
        cost = calculate_cost(y, outputs, model)
        model.optimize(preprocessed_x, y, outputs)

    print('R-Squared score: ', r_squared_score(y, outputs))
    plot_line_2d_using_pca(preprocessed_x, y, model)
