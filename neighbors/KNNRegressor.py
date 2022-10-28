import numpy as np
from collections import Counter


class KNNRegressor:
    """
    K-nearest-neighbor classifier using L1 ('manhattan') or L2 ('euclidean') loss
    """

    def __init__(self, k=1, metric=None):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
        return self

    def predict(self, X):
        '''
        Uses the KNN model to predict classes for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of float (num_samples) - predicted target
           for each sample
        '''

        dists = self.compute_distances_no_loops(X)
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.float64)
        for i in range(num_test):
            idx = np.argpartition(dists[i], self.k)
            pred[i] = np.mean(self.train_X[idx])
        return pred

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float64 to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float64)
        if self.metric == 'manhattan':
            dists = np.abs(X[:, np.newaxis] - self.train_X).sum(axis=-1)
        if self.metric == 'euclidean':
            dists = np.sqrt((np.abs(X[:, np.newaxis] - self.train_X) ** 2).sum(axis=-1))
        return dists