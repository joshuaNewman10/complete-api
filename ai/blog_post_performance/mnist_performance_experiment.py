import logging
import os
import time

from argparse import ArgumentParser
from keras.callbacks import TensorBoard, ReduceLROnPlateau

from ai.blog_post_performance.experiment.mnist_keras_convnet import MNISTConvnetKerasImageExperiment
from ai.common.callback.keras import GradientDebugger
from ai.common.data_provider.mnist import MNISTTrainingDataProvider
from ai.common.evaluator.multi_class import MultiClassEvaluator
from ai.image.transformer.base import ImageTransformer
from ai.common.provider.coin_toss_model import CoinTossModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


def run_experiment(experiment_directory):
    evaluator = MultiClassEvaluator()
    experiment = MNISTConvnetKerasImageExperiment(evaluator=evaluator, experiment_directory=experiment_directory)
    class_map = experiment.class_map
    experiment_name = experiment.name

    height = experiment.HEIGHT
    width = experiment.WIDTH
    num_channels = experiment.NUM_CHANNELS

    log_dir = os.path.join(experiment_directory, experiment_name)
    model = experiment.get_model()

    transformer = ImageTransformer(width=width, height=height, num_channels=num_channels)
    training_data_provider = MNISTTrainingDataProvider(height=height, width=width, num_channels=num_channels, transformer=transformer)
    training_data = training_data_provider.get_training_data()

    X_train = training_data.X_train
    y_train = training_data.y_train

    X_test = training_data.X_test
    y_test = training_data.y_test
    test_samples = training_data_provider.testing_samples

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir=log_dir),
        ReduceLROnPlateau(factor=0.01, verbose=1)
    ]

    LOGGER.info("Beginning training for experiment %s", experiment)

    history = model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1,
                        callbacks=callbacks,
                        validation_data=(training_data.X_validation, training_data.y_validation))
    history = history.history

    y_test_pred = model.predict(X_test)
    experiment.persist_test_data_predictions(
        y_test_pred=y_test_pred,
        y_test_true=y_test,
        x_test=[test_sample.source for test_sample in test_samples],
        samples=test_samples
    )

    coin_toss_model = CoinTossModel(class_map=class_map)
    coin_toss_test_pred = coin_toss_model.get_predictions(y_test)

    experiment.compute_and_persist_evaluation_data(
        y_test_pred=y_test_pred,
        y_test_true=y_test
    )

    experiment.compute_and_persist_coin_toss_evaluation_data(
        y_test_pred=coin_toss_test_pred,
        y_test_true=y_test
    )

    experiment.persist_experiment(model, history)


def main(experiment_directory):
    start = time.time()
    run_experiment(experiment_directory)
    end = time.time()

    LOGGER.info("Ran experiment in %s seconds", end - start)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_directory", default="training_runs")

    args = parser.parse_args()
    main(args.experiment_directory)
