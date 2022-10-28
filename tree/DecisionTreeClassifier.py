from metrics import entropy, gini


class DecisionTreeClassifier:
    def __init__(self):
        raise NotImplementedError('Implement __init__() method')

    def fit(self, X, y):
        raise NotImplementedError('Implement fit() method')

    def predict_probabilities(self, X, y):
        raise NotImplementedError('Implement predict_probabilities() method')

    def predict(self, X):
        raise NotImplementedError('Implement predict() method')
