import keras
import numpy

from keras.datasets import mnist

from ai.common.data_provider.base import TrainingDataProvider
from ai.common.model.sample import ImageSample
from ai.common.model.training_data import TrainingData


class MNISTTrainingDataProvider(TrainingDataProvider):
    NUM_CLASSES = 10  # 0 1 2 3 4 5 6 7 8 9

    def __init__(self, height, width, num_channels, image_file_sink, transformer, store_images=False):
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.image_file_sink = image_file_sink
        self.transformer = transformer

        self._store_images = store_images

        self.training_samples = []
        self.testing_samples = []
        self.validation_samples = []

    def get_training_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        self.training_samples = self._convert_samples(X=X_train, y=y_train)
        self.testing_samples = self._convert_samples(X=X_test, y=y_test, store_images=self._store_images)
        self.validation_samples = self._convert_samples(X=X_test, y=y_test, store_images=self._store_images)

        y_train = keras.utils.to_categorical(y_train, self.NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, self.NUM_CLASSES)

        X_train = list(map(self.transformer.transform, X_train))
        X_test = list(map(self.transformer.transform, X_test))

        X_train = numpy.reshape(X_train, (
            len(X_train), self.height, self.width,
            self.num_channels))  # adapt this if using `channels_first` image data format
        X_test = numpy.reshape(X_test, (
            len(X_test), self.height, self.width,
            self.num_channels))  # adapt this if using `channels_first` image data format

        return TrainingData(X_train, y_train, X_test, y_test, X_test, y_test)

    def _convert_samples(self, X, y, store_images=False):
        samples = []

        for xi, yi in zip(X, y):
            if store_images:
                image_file_path = self.image_file_sink.sink(xi)
                source = image_file_path
            else:
                source = None

            sample = dict(
                image=xi,
                class_name=y,
                source=source,
                vector=None,
            )
            sample = self.deserialize_sample(sample)
            samples.append(sample)

        return samples

    def deserialize_sample(self, sample):
        return ImageSample(
            image=sample["image"],
            class_name=sample["class_name"],
            source=sample["source"],
            vector=None
        )
