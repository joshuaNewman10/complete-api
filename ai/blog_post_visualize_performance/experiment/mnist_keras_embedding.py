import json
import keras

from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from ai.common.model.experiment import Experiment


class MNISTKerasCNNEmbeddingExperiment(Experiment):
    _name = 'mnist_keras_cnn_embedding_experiment'
    _class_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}

    HEIGHT = 28
    WIDTH = 28
    NUM_CHANNELS = 1

    NUM_TARGETS = 10

    learning_rate = 0.001
    loss_function = 'categorical_crossentropy'
    optimizer = keras.optimizers.Adam(lr=learning_rate)

    def get_model(self):
        input_img = Input(name='input', shape=(self.HEIGHT, self.WIDTH, self.NUM_CHANNELS))
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        encoded = Flatten()(X)
        X = Dense(128, activation="relu")(encoded)
        y = Dense(self.NUM_TARGETS, activation="softmax")(X)

        encoder = Model(inputs=input_img, outputs=encoded)
        classifier = Model(inputs=input_img, outputs=y)
        classifier.compile(loss=self.loss_function, optimizer='adam', metrics=["accuracy"])
        return encoder, classifier

    def get_params(self):
        encoder, classifier = self.get_model()

        return dict(
            model_architecture=json.dumps(dict(
                encoder=encoder.summary(),
                classifier=classifier.summary()
            )),
            loss=json.dumps(self.loss_function),
            optimizer="Adam",
            learning_rate=self.learning_rate,
            height=self.HEIGHT,
            width=self.WIDTH,
            num_channels=self.NUM_CHANNELS,
            num_targets=self.NUM_TARGETS
        )
