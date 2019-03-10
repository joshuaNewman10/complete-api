import unittest

from mock import Mock

from ai.common.model.experiment import Experiment


class ExperimentModelTests(unittest.TestCase):
    def setUp(self):
        self.experiment = Experiment(experiment_directory="test", evaluator=None)
        self.experiment._create_experiment_directory = Mock()

    def test_get_params(self):
        params = self.experiment.get_params()
        self.assertEqual(params, {})


if __name__ == "__main__":
    unittest.main()
