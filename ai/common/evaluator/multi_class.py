import numpy as np
import sklearn

from collections import defaultdict

from ai.common.evaluator.base import Evaluator


class MultiClassEvaluator(Evaluator):
    """
    Produces precision + recall metrics, ROC + Precision Recall curve graph and threshold decisions
    """
    name = "multiclass_evaluator"
    DESIRED_PRECISION = 0.8

    def evaluate(self, class_map, predictions, true_labels):
        """
        Provides valuation metrics for each class
        :param class_map: models class_map in form of {category: ix}
        :param predictions: predictions on test data [[0, 0.1. 0.6, 0.3]]
        :param true_labels: actual labels on test data [0, 1, 2, 0, 3]
        :return: dict of category metrics
        """
        category_metrics = {}
        indices = list(class_map.values())
        evaluation_metrics = defaultdict(dict)

        true_labels = [np.argmax(labels) for labels in true_labels]
        multi_class_confusion_matrix = self._get_multi_class_confusion_matrix(predictions, true_labels, indices)

        for category, category_ix in class_map.items():
            y_proba = predictions[:, category_ix]
            y_pred = [
                1 if prediction == category_ix else 0
                for prediction in predictions.argmax(axis=-1)
            ]
            y_true = np.array(
                [
                    1 if true_category_ix == category_ix else 0
                    for true_category_ix in true_labels
                ],
                dtype=np.uint8,
            )
            category_metrics[category] = self._get_metrics(category_ix, y_true, y_pred, y_proba, class_map,
                                                           multi_class_confusion_matrix)

        evaluation_metrics["coin_toss_data"] = self._get_coin_toss_data(class_map)
        evaluation_metrics["category_metrics"] = category_metrics
        evaluation_metrics["confusion_matrix"] = multi_class_confusion_matrix

        return evaluation_metrics

    def _get_coin_toss_data(self, class_map):
        num_classes = len(class_map)
        tp_rate = 1 / float(num_classes)
        fp_rate = 1 - tp_rate
        tn_rate = 1 / float(num_classes)
        fn_rate = 1 - tn_rate

        return dict(
            tp_rate=tp_rate,
            fp_rate=fp_rate,
            tn_rate=tn_rate,
            fn_rate=fn_rate
        )

    def _get_metrics(self, category_ix, y_true, y_pred, y_proba, class_map, multi_class_confusion_matrix):
        """
        {
            fire: {
                precision_score: 1.0,
                recall_score: 1.0,
                f1_score: 1.0,
                precision_recall_curve: (precision, recall, thresholds),
                roc_curve: (fpr, tpr, thresholds),
                recall_at_p80: 1.0
                threshold_at_p80: 1.0
            }
        }
        """
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        auc = sklearn.metrics.roc_auc_score(y_true, y_proba)
        pr_curve = sklearn.metrics.precision_recall_curve(y_true, y_proba)
        roc_curve = sklearn.metrics.roc_curve(y_true, y_proba)
        confusion_matrix = self._get_category_confusion_matrix(y_true, y_pred)
        most_commonly_confused_class = self._get_most_commonly_confused_class(category_ix, class_map,
                                                                              multi_class_confusion_matrix)

        precision_at_threshold, recall_at_precision, threshold_at_precision = self._get_point_at_desired_precision(
            pr_curve, desired_precision=self.DESIRED_PRECISION
        )
        return dict(
            y_proba=y_proba,
            y_pred=y_pred,
            y_true=y_true,
            most_commonly_confused_class=most_commonly_confused_class,
            confusion_matrix=confusion_matrix,
            precision_score=precision,
            recall_score=recall,
            f1_score=f1,
            auc_score=auc,
            curve=dict(
                precision=precision_at_threshold,
                recall=recall_at_precision,
                threshold=threshold_at_precision,
            ),
            precision_recall_curve=pr_curve,
            roc_curve=roc_curve,
            recall_at_p80=recall_at_precision,
            threshold_at_p80=threshold_at_precision,
        )

    def _get_most_commonly_confused_class(self, category_ix, class_map, multi_class_confusion_matrix):
        inverted_class_map = {ix: category for category, ix in class_map.items()}
        category_confusion_matrix = multi_class_confusion_matrix[category_ix]
        most_commonly_confused_class_ix = 0
        num_false_predictions = 0

        for ix, entry in enumerate(category_confusion_matrix):
            if ix == category_ix:
                continue

            if entry >= num_false_predictions:
                most_commonly_confused_class_ix = ix
                num_false_predictions = entry

        most_commonly_confused_class = inverted_class_map[most_commonly_confused_class_ix]
        return most_commonly_confused_class

    def _get_multi_class_confusion_matrix(self, predictions, y_true, labels):
        y_pred = [np.argmax(pred) for pred in predictions]
        return sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    def _get_category_confusion_matrix(self, y_true, y_pred):
        confusion_matrix = defaultdict(int)

        for yi_true, yi_pred in zip(y_true, y_pred):
            confusion_matrix["sample_size"] += 1

            if yi_true == 1 and yi_pred == 1:
                confusion_matrix["tp"] += 1

            elif yi_true == 1 and yi_pred == 0:
                confusion_matrix["fn"] += 1

            elif yi_true == 0 and yi_pred == 0:
                confusion_matrix["tn"] += 1

            elif yi_true == 0 and yi_pred == 1:
                confusion_matrix["fp"] += 1

        confusion_matrix["tp_rate"] = confusion_matrix["tp"] / float(confusion_matrix["sample_size"])
        confusion_matrix["fp_rate"] = confusion_matrix["fp"] / float(confusion_matrix["sample_size"])
        confusion_matrix["tn_rate"] = confusion_matrix["tn"] / float(confusion_matrix["sample_size"])
        confusion_matrix["fn_rate"] = confusion_matrix["fn"] / float(confusion_matrix["sample_size"])

        return confusion_matrix

    def _get_point_at_desired_precision(
            self, precision_recall_curve, desired_precision=0.8
    ):
        for precision, recall, threshold in zip(*precision_recall_curve):
            if precision >= desired_precision:
                return precision, recall, threshold

        return -1, -1, -1
