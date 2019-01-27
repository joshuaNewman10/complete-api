class TrainingDataProvider:
    def get_training_data(self):
        raise NotImplementedError()

    def deserialize_sample(self, sample):
        raise NotImplementedError()

    def _get_training_samples(self, samples):
        training_split = int(self.TRAIN_SPLIT * self._num_samples)
        return samples[0:training_split]

    def _get_testing_samples(self, samples):
        validation_split = self.TRAIN_SPLIT + self.VAL_SPLIT
        training_split = int(self.TRAIN_SPLIT * self._num_samples)
        validation_split = int(validation_split * self._num_samples)
        return samples[training_split:validation_split]

    def _get_validation_samples(self, samples):
        validation_split = self.TRAIN_SPLIT + self.VAL_SPLIT
        validation_split = int(validation_split * self._num_samples)
        return samples[validation_split:self._num_samples]

