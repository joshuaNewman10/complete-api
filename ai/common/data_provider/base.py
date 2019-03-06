class TrainingDataProvider:
    TRAIN_SPLIT = 0.6
    TEST_SPLIT = 0.2
    VAL_SPLIT = 0.2

    def get_training_data(self):
        raise NotImplementedError()

    def deserialize_sample(self, sample):
        raise NotImplementedError()

    def _get_training_samples(self, samples):
        raise NotImplemented()

    def _get_testing_samples(self, samples):
        raise NotImplemented()

    def _get_validation_samples(self, samples):
        raise NotImplemented()
