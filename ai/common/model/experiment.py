import os
import logging
import time

import numpy as np

from ai.common.sink.json_file import JSONFileSink
from ai.common.sink.jsonl_file import JSONLFileSink

LOGGER = logging.getLogger(__name__)


class Experiment:
    """
    Base Experiment class
    """
    _name = "experiment"
    _class_map = None

    _class_map_file_path = None
    _coin_toss_evaluation_data_file_path = None
    _evaluation_data_file_path = None
    _model_file_path = None
    _training_metadata_file_path = None
    _test_data_predictions_file_path = None

    def __init__(self, experiment_directory, evaluator=None):
        """
        :param experiment_directory: path to experiment directory to store training data
        :param evaluator: ai.common.evaluator.Evaluator
        """
        self._evaluator = evaluator
        experiment_directory = os.path.join(experiment_directory, self._name)
        self.experiment_directory = os.path.join(experiment_directory, str(time.time()))
        self._create_experiment_directory()

        self._class_map_sink = JSONFileSink(self.class_map_file_path)
        self._training_metadata_sink = JSONFileSink(self.training_metadata_file_path)
        self._coin_toss_evaluation_data_sink = JSONFileSink(self.coin_toss_evaluation_data_file_path)
        self._evaluation_data_sink = JSONFileSink(self.evaluation_data_file_path)
        self._test_data_predictions_sink = JSONLFileSink(self.test_data_predictions_file_path)

    def __repr__(self):
        LOGGER.info("Experiment %s with params %s", self.name, self.params)

    def get_model(self):
        raise NotImplementedError()

    def get_params(self):
        return dict()

    def persist_test_data_predictions(self, y_test_pred, y_test_true, samples, x_test=None):
        if not x_test:
            x_test = [None for _ in range(len(y_test_true))]

        for yi_proba, yi_true, xi, sample, in zip(y_test_pred, y_test_true, x_test, samples):
            yi_pred_ix = np.argmax(yi_proba)
            yi_pred_name = self.inverted_class_map[yi_pred_ix]

            prediction_entry = dict(
                yi_proba=yi_proba,
                yi_pred_ix=yi_pred_ix,
                yi_pred_name=yi_pred_name,
                yi_true=yi_true,
                xi=xi,
                sample_type=type(sample).__name__
            )
            self._test_data_predictions_sink.receive(item=prediction_entry)

    def compute_and_persist_evaluation_data(self, y_test_pred, y_test_true):
        evaluation_data = self._evaluator.evaluate(class_map=self._class_map, predictions=y_test_pred,
                                                   true_labels=y_test_true)
        self._evaluation_data_sink.receive(item=evaluation_data)

    def compute_and_persist_coin_toss_evaluation_data(self, y_test_pred, y_test_true):
        evaluation_data = self._evaluator.evaluate(class_map=self._class_map, predictions=y_test_pred,
                                                   true_labels=y_test_true)
        self._coin_toss_evaluation_data_sink.receive(item=evaluation_data)

    def persist_experiment(self, model, history):
        LOGGER.info("Persisting experiment to path %s", self.experiment_directory)
        model.save(self.model_file_path)
        training_metadata = dict(history=history, params=self.params)
        self._training_metadata_sink.receive(item=training_metadata)
        self._class_map_sink.receive(item=self.class_map)

    def _create_experiment_directory(self):
        if not os.path.exists(self.experiment_directory):
            os.makedirs(self.experiment_directory)

    @property
    def name(self):
        if self._name is None:
            self._name = "_".join([self.__name__ + str(time.time())])
        return self._name

    @property
    def coin_toss_evaluation_data_file_path(self):
        if self._coin_toss_evaluation_data_file_path is None:
            coin_toss_evaluation_data_file_path = "_".join([self.name, "coin_toss_evaluation_data_path.json"])
            self._coin_toss_evaluation_data_file_path = os.path.join(self.experiment_directory,
                                                                     coin_toss_evaluation_data_file_path)
        return self._coin_toss_evaluation_data_file_path

    @property
    def evaluation_data_file_path(self):
        if self._evaluation_data_file_path is None:
            evaluation_data_file_path = "_".join([self.name, "evaluation_data_path.json"])
            self._evaluation_data_file_path = os.path.join(self.experiment_directory, evaluation_data_file_path)
        return self._evaluation_data_file_path

    @property
    def model_file_path(self):
        if self._model_file_path is None:
            model_file_path = ".".join([self.name, "h5"])
            self._model_file_path = os.path.join(self.experiment_directory, model_file_path)
        return self._model_file_path

    @property
    def test_data_predictions_file_path(self):
        if self._test_data_predictions_file_path is None:
            test_data_predictions_file_path = "_".join([self.name, "test_data_predictions.jsonl"])
            self._test_data_predictions_file_path = os.path.join(self.experiment_directory,
                                                                 test_data_predictions_file_path)
        return self._test_data_predictions_file_path

    @property
    def training_metadata_file_path(self):
        if self._training_metadata_file_path is None:
            training_metadata_file_path = "_".join([self.name, "training_metadata.json"])
            self._training_metadata_file_path = os.path.join(self.experiment_directory, training_metadata_file_path)
        return self._training_metadata_file_path

    @property
    def class_map_file_path(self):
        if self._class_map_file_path is None:
            class_map_file_path = "_".join([self.name, "class_map.json"])
            self._class_map_file_path = os.path.join(self.experiment_directory, class_map_file_path)
        return self._class_map_file_path

    @property
    def params(self):
        return self.get_params()

    @property
    def class_map(self):
        return self._class_map

    @property
    def inverted_class_map(self):
        return {ix: class_name for class_name, ix in self.class_map.items()}
