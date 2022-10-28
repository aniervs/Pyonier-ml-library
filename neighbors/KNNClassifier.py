import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 ('manhattan') or L2 ('euclidean') loss
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
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''

        dists = self.compute_distances_no_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

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
        # Using float32 to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        if self.metric == 'manhattan':
            dists = np.abs(X[:, np.newaxis] - self.train_X).sum(axis=-1)
        if self.metric == 'euclidean':
            dists = np.sqrt((np.abs(X[:, np.newaxis] - self.train_X) ** 2).sum(axis=-1))
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            idx = np.argpartition(dists[i], self.k)
            pred[i] = np.bool(Counter(self.train_X[idx]).most_common(n=1)[0][0])
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            idx = np.argpartition(dists[i], self.k)
            pred[i] = np.bool(Counter(self.train_X[idx]).most_common(n=1)[0][0])
        return pred
