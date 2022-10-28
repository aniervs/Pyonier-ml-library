import numpy as np
from metrics import gini, entropy
from Node import Node
from utils import one_hot_encode


class DecisionTree:
    all_criteria = {
        'gini': gini,
        'entropy': entropy
    }

    def __init__(self, n_classes=None, max_depth=None, min_samples_split=2, criterion_name='gini'):

        self.criterion = None
        assert criterion_name in self.all_criteria.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criteria.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None

    def make_split(self, feature_index, threshold, X, y):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X and y
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X and y
            Part of the provided subset where selected feature x^j >= threshold
        """

        left_indices = np.where(X[:, feature_index] < threshold)
        right_indices = np.where(X[:, feature_index] >= threshold)

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X, y):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        left_indices = np.where(X[:, feature_index] < threshold)
        right_indices = np.where(X[:, feature_index] >= threshold)

        y_left = y[left_indices]
        y_right = y[right_indices]

        return y_left, y_right

    def choose_best_split(self, X, y):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """

        n_objects, n_features = X.shape

        feature_index, threshold, optimal_quality = None, None, 0

        h_q = self.criterion(y)

        for feat_idx in range(n_features):

            values = set(X[:, feat_idx])
            for t in values:
                y_left, y_right = self.make_split_only_y(feat_idx, t, X, y)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                h_left = self.criterion(y_left)
                h_right = self.criterion(y_right)

                current_quality = h_q - y_left.shape[0] / n_objects * h_left - y_right.shape[0] / n_objects * h_right

                if optimal_quality < current_quality:
                    optimal_quality = current_quality
                    feature_index = feat_idx
                    threshold = t

        return feature_index, threshold

    def make_tree(self, X, y, depth=0):
        """
        Recursively builds the tree

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        depth: depth of the current vertex during the construction

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        feature_index, threshold = self.choose_best_split(X, y)

        if depth == self.max_depth or feature_index is None:  # X.shape[0] < self.min_samples_split:
            node = Node(None, None)
            node.proba = np.mean(y, axis=0)
            return node

        new_node = Node(feature_index, threshold, probability=depth)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X, y)

        new_node.left_child = self.make_tree(X_left, y_left, depth + 1)
        new_node.right_child = self.make_tree(X_right, y_right, depth + 1)

        return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion = self.all_criteria[self.criterion_name]
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        y_predicted = np.argmax(self.predict_probabilities(X), axis=1)

        return y_predicted

    def dfs(self, node: Node, X):
        if node.left_child is None and node.right_child is None:
            return node.probability
        if X[node.feature_index] < node.value:
            return self.dfs(node.left_child, X)
        return self.dfs(node.right_child, X)

    def predict_probabilities(self, X):
        """
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """

        y_predicted_probs = np.array([self.dfs(self.root, obj) for obj in X])

        return y_predicted_probs
