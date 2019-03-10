import logging
import os
import time
from argparse import ArgumentParser

from keras.callbacks import TensorBoard, ReduceLROnPlateau

from ai.blog_post_visualize_performance.experiment.mnist_keras_embedding import MNISTKerasCNNEmbeddingExperiment
from ai.blog_post_visualize_performance.provider.image_embedding_data import ImageEmbeddingDataProvider
from ai.common.callback.keras import GradientDebugger
from ai.common.data_provider.mnist import MNISTTrainingDataProvider
from ai.common.evaluator.multi_class import MultiClassEvaluator
from ai.common.sink.image_file import ImageFileSink
from ai.image.transformer.base import ImageTransformer

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


def run_experiment(experiment_directory, download_directory):
    evaluator = MultiClassEvaluator()
    experiment = MNISTKerasCNNEmbeddingExperiment(evaluator=evaluator, experiment_directory=experiment_directory)
    experiment_name = experiment.name
    experiment_directory = experiment.experiment_directory

    image_file_sink = ImageFileSink(download_directory)

    height = experiment.HEIGHT
    width = experiment.WIDTH
    num_channels = experiment.NUM_CHANNELS
    transformer = ImageTransformer(width=width, height=height, num_channels=num_channels)

    encoder, classifier = experiment.get_model()
    training_data_provider = MNISTTrainingDataProvider(height=height, width=width,
                                                       transformer=transformer, num_channels=num_channels,
                                                       image_file_sink=image_file_sink,
                                                       store_images=True)

    training_data = training_data_provider.get_training_data()

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir="/tmp/logs"),
        ReduceLROnPlateau(factor=0.01, verbose=1)
    ]

    LOGGER.info("Beginning training for experiment %s", experiment)

    X_train = training_data.X_train
    y_train = training_data.y_train

    X_validation = training_data.X_validation
    y_validation = training_data.y_validation

    X_test = training_data.X_test
    y_test = training_data.y_test

    history = classifier.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1,
                             callbacks=callbacks,
                             validation_data=(X_validation, y_validation))

    history = history.history

    y_test_pred = classifier.predict(X_test)

    experiment.compute_and_persist_evaluation_data(
        y_test_pred=y_test_pred,
        y_test_true=training_data.y_test
    )

    experiment.persist_experiment(classifier, history)

    embedding_data_provider = ImageEmbeddingDataProvider(
        experiment_directory,
        experiment_name,
        encoder,
        transformer=transformer,
    )

    test_samples = training_data_provider.testing_samples
    embedding_data_provider.store_embedding_data(test_samples[:1000])

    experiment.persist_test_data_predictions(
        y_test_pred=y_test_pred,
        y_test_true=y_test,
        x_test=[test_sample.source for test_sample in test_samples],
        samples=test_samples
    )


def main(experiment_directory, download_directory):
    start = time.time()
    run_experiment(experiment_directory, download_directory)
    end = time.time()

    LOGGER.info("Ran experiment in %s seconds", end - start)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_directory", default="training_runs")
    parser.add_argument("--download_directory", default="/tmp/images")

    args = parser.parse_args()
    main(args.experiment_directory, args.download_directory)
