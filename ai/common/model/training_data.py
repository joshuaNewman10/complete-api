class TrainingData:
    def __init__(self, X_train, y_train, X_test, y_test, X_validation, y_validation):
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.X_validation = X_validation
        self.y_validation = y_validation
