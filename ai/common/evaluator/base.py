from abc import abstractmethod


class Evaluator:
    def __init__(self, desired_precision=0.8):
        self._desired_precision = desired_precision

    @abstractmethod
    def evaluate(self, class_map, y_pred, y_true):
        raise NotImplementedError()
