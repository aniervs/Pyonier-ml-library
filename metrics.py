import numpy as np


def accuracy(y_pred, y_true):
    n_objects = y_pred.shape[0]
    cnt = [y_pred[i] == y_true[i] for i in range(n_objects)]
    return cnt / n_objects

def precision(y_pred, y_true):
    n_objects = y_pred.shape[0]
    
    tp = len([y_pred[i] == True and y_true[i] == True for i in range(n_objects)])
    fp = len([y_pred[i] == True and y_true[i] == False for i in range(n_objects)])

    return tp / (tp + fp)

def recall(y_pred, y_true):
    n_objects = y_pred.shape[0]
    
    tp = len([y_pred[i] == True and y_true[i] == True for i in range(n_objects)])
    fn = len([y_pred[i] == False and y_true[i] == True for i in range(n_objects)])

    return tp / (tp + fn)

def f1_score(y_pred, y_true):
    precision = precision(y_pred, y_true)
    recall = recall(y_pred, y_true)

    return 2 * precision * recall / (recall + precision)

def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """

    EPS = 0.0005

    p = np.mean(y, axis = 0)

    return -np.sum(p * np.log2(p + EPS))

def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    p = np.mean(y, axis = 0)
    return -np.sum(p * p) + 1

