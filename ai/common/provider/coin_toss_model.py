import numpy as np
from random import randint


class CoinTossModel:
    def __init__(self, class_map):
        self.class_map = class_map

    def get_predictions(self, y_true):
        num_predictions = len(y_true)
        num_classes = len(self.class_map)
        y_pred = np.zeros(shape=(num_predictions, num_classes))

        for ix in range(num_predictions):
            predictions = np.zeros(shape=(num_classes,))
            predicted_class_ix = randint(0, num_classes - 1)
            predictions[predicted_class_ix] = 1.0
            y_pred[ix:] = predictions

        return y_pred
