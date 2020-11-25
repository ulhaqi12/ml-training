import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    class that will handle KMeans Model
    """

    def __init__(self, no_of_clusters):
        """
        constructor that will ceate KMeans Object ho handle model
        :param no_of_clusters: no of clusters
        """
        self.k = no_of_clusters
        self.centroids = None
        self.c = None

    @staticmethod
    def distance(p1, p2):
        """
        function will find euclidean distance between two points
        :param p1: point 1
        :param p2: point 2
        :return:
        """

        distance = 0
        for i in range(len(p1)):
            distance += np.square(p2[i] - p1[i])
        distance = np.sqrt(distance)
        return distance

    def initialize_centroids(self, X):
        """
        function that will initialize centroids with data points randomly
        :param X: input dataset
        :return:
        """

        self.centroids = np.zeros((self.k, X.shape[1]))
        self.c = np.zeros(X.shape[0])
        random_indexes_list = []
        for i in range(self.k):
            random_indexes_list.append(np.random.randint(0, X.shape[0]))
        random_indexes = np.array(random_indexes_list)
        cen_index = 0
        for i in random_indexes:
            self.centroids[cen_index] = X[i]
            cen_index += 1

    def assign_data_points_to_centroids(self, X):

        for i in range(len(X)):
            min_distant_centroid = 0
            min_distance = self.distance(X[i], self.centroids[0])
            for j in range(len(self.centroids)):
                current_distance = self.distance(X[i], self.centroids[j])
                if min_distance > current_distance:
                    min_distance = current_distance
                    min_distant_centroid = j
            self.c[i] = min_distant_centroid

    def calculate_new_centroids(self, X):

        for i in range(len(self.centroids)):
            data_points = X[self.c == i]
            self.centroids[i] = np.sum(data_points, axis=0) // len(data_points)

    def fit(self, X):
        """
        function that will fit model and find clusters in data.
        :param X: input dataset
        :return:
        """

        self.initialize_centroids(X)

        for i in range(20):
            print('Epoch:', i)
            self.assign_data_points_to_centroids(X)
            plot_with_centroids(X, self.centroids, self.c)
            self.calculate_new_centroids(X)


def plot(data):
    """
    funciton that will plot dataset on 2d plane
    :param data: input dataframe
    :return:
    """

    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def plot_with_centroids(data, centroids, c):
    """
    function that will plot data and centroids with different colors
    :param data: input data
    :param centroids: centroids
    :param c: information about data points (with which centroids they belong)
    :return:
    """

    for i in range(len(centroids)):
        plotting_data = data[c == i]
        plt.scatter(plotting_data[:, 0], plotting_data[:, 1])
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('../Mall_Customers.csv')
    data_np = data.to_numpy()

    model = KMeans(no_of_clusters=5)
    model.fit(data_np)
    plot_with_centroids(data_np, centroids=model.centroids, c=model.c)
