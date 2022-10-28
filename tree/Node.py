class Node:
    def __init__(self, feature_index, threshold, probability=None):
        self.feature_index = feature_index
        self.value = threshold
        self.probability = probability
        self.left_child = None
        self.right_child = None
