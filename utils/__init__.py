import numpy as np


def count_number_calls(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)  # creates a (n_objects, n_classes) matrix
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.  # sets to 1 the corresponding class for each object
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]
