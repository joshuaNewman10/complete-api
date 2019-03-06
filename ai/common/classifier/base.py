class Classifier:
    def __init__(self, name):
        self.name = name

    def predict(self, X):
        raise NotImplementedError
