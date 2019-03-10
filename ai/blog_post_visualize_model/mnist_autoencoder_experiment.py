import logging
import os
import time
from argparse import ArgumentParser

from keras.callbacks import TensorBoard, ReduceLROnPlateau

from ai.blog_post_visualize_performance.experiment.mnist_keras_autoencoder_convnet import MNISTConvnetKerasImageExperiment
from ai.common.callback.keras import GradientDebugger
from ai.common.data_provider.mnist import MNISTTrainingDataProvider
from ai.common.evaluator.multi_class import MultiClassEvaluator

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


def run_experiment(experiment_directory):
    evaluator = MultiClassEvaluator()
    experiment = MNISTConvnetKerasImageExperiment(evaluator=evaluator, experiment_directory=experiment_directory)
    experiment_name = experiment.name

    height = experiment.HEIGHT
    width = experiment.WIDTH
    num_channels = experiment.NUM_CHANNELS

    log_dir = os.path.join(experiment_directory, experiment_name)
    autoencoder, classifier = experiment.get_model()

    training_data_provider = MNISTTrainingDataProvider(height=height, width=width, num_channels=num_channels)
    training_data = training_data_provider.get_training_data()

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir=log_dir),
        ReduceLROnPlateau(factor=0.01, verbose=1)
    ]

    LOGGER.info("Beginning training for experiment %s", experiment)

    history = autoencoder.fit(training_data.X_train,
                              {'decoder': training_data.X_train, 'classifier': training_data.y_train},
                              epochs=2,
                              batch_size=128,
                              shuffle=True,
                              validation_data=(training_data.X_test,
                                               {'decoder': training_data.X_test, 'classifier': training_data.y_test}),
                              callbacks=callbacks)

    history = history.history

    y_test_pred = classifier.predict(training_data.X_test)

    experiment.persist_test_data_predictions(
        y_test_pred=y_test_pred,
        y_test_true=training_data.y_test,
        x_test=None
    )

    experiment.compute_and_persist_evaluation_data(
        y_test_pred=y_test_pred,
        y_test_true=training_data.y_test
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
