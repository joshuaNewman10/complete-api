import json
import keras

from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D

from ai.common.model.experiment import Experiment


class MNISTConvnetKerasImageExperiment(Experiment):
    _name = 'cnn_keras'
    _class_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}

    HEIGHT = 28
    WIDTH = 28
    NUM_CHANNELS = 1

    NUM_TARGETS = 10

    learning_rate = 0.001
    loss_functions = dict(
        decoder="mse",
        classifier="binary_crossentropy"
    )
    optimizer = keras.optimizers.Adam(lr=learning_rate)

    def get_model(self):
        input_img = Input(name='input', shape=(self.HEIGHT, self.WIDTH, self.NUM_CHANNELS))
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

        X = Flatten()(X)
        X = Dense(128, activation='relu')(X)
        y = Dense(10, name='classifier', activation='softmax')(X)

        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(encoded)
        X = UpSampling2D(size=(2, 2))(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = UpSampling2D(size=(2, 2))(X)
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(X)
        X = UpSampling2D(size=(2, 2))(X)
        decoded = Conv2D(name='decoder', filters=1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid',
                         padding='same')(X)

        autoencoder = Model(inputs=input_img, outputs=[decoded, y])
        autoencoder.compile(loss=self.loss_functions, optimizer='adam')
        classifier = Model(input=input_img, output=y)
        print(autoencoder.summary())
        print(classifier.summary())
        return autoencoder, classifier

    def get_params(self):
        autoencoder, classifier = self.get_model()

        return dict(
            model_architecture=json.dumps(dict(
                autoencoder=autoencoder.summary(),
                classifier=classifier.summary()
            )),
            loss=json.dumps(self.loss_functions),
            optimizer="Adam",
            learning_rate=self.learning_rate,
            height=self.HEIGHT,
            width=self.WIDTH,
            num_channels=self.NUM_CHANNELS,
            num_targets=self.NUM_TARGETS
        )
