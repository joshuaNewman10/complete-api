import json

import keras

from ai.common.model.experiment import Experiment

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class MNISTConvnetKerasImageExperiment(Experiment):
    _name = 'cnn_keras'
    _class_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}

    HEIGHT = 28
    WIDTH = 28
    NUM_CHANNELS = 1

    NUM_TARGETS = 10

    learning_rate = 0.001
    loss_function = 'categorical_crossentropy'
    optimizer = keras.optimizers.Adam(lr=learning_rate)

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.WIDTH, self.HEIGHT, self.NUM_CHANNELS)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUM_TARGETS, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return model

    def get_params(self):
        return dict(
            model_architecture=json.dumps(self.get_model().summary()),
            loss=self.loss_function,
            optimizer="Adam",
            learning_rate=self.learning_rate,
            height=self.HEIGHT,
            width=self.WIDTH,
            num_channels=self.NUM_CHANNELS,
            num_targets=self.NUM_TARGETS
        )
