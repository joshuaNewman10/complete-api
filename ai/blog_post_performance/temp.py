plot_prediction_comparison(coin_toss_validation_data, target_zero_validation_data)


def view_detailed_prediction(category, x, bucket, prediction_type):
    category_prediction_bucket_samples = category_prediction_buckets[category]

    prediction_bucket = category_prediction_bucket_samples[bucket_ix]
    sample_bucket = prediction_bucket['samples'][prediction_type]
    num_samples = len(sample_bucket)
    if x >= num_samples:
        x = num_samples - 1

    sample = sample_bucket[x]
    sequence = sample['x']
    prediction = sample['y_proba']

    display(HTML(html))



def plot_prediction_distributions(predictions):
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.hist(predictions, alpha=0.7)
    title = '%s | Prediction Distribution' % (category)
    plt.title(title)
    plt.xlabel('y_proba')
    plt.ylabel('count')
    plt.show()

    def initialize_curve_figures(self, categories):
        figures = defaultdict(dict)
        for category in categories:
            figures[category]["precision_recall"] = self.get_precision_recall_figure(
                category
            )
            figures[category]["roc"] = self.get_roc_figure(category)
        return figures

    def get_precision_recall_figure(self, category):
        figure = plt.figure(figsize=(8, 8))
        title = category + " | " + "Precision - Recall"
        ax = figure.gca()
        ax.set_title(title)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        return figure

    def get_roc_figure(self, category):
        figure = plt.figure(figsize=(8, 8))
        title = category + " | " + "ROC"
        ax = figure.gca()
        ax.set_title(title)
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Coin Flip", alpha=.5
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        return figure

    def save_figures(self, category, figures):
        figure_file_paths = []
        for figure_type, figure in figures.items():
            figure.tight_layout()
            ax = figure.gca()
            ax.legend(loc="lower right")
            filename = "_".join([category, figure_type]) + ".png"
            figure_path = os.path.join(self.model_dir, filename)
            figure.savefig(figure_path)
            plt.close(figure)
            figure_file_paths.append(figure_path)
        return figure_file_paths


    def plot_metrics(self, metrics_by_category):
        categories = metrics_by_category.keys()
        figures_by_category = self.initialize_curve_figures(categories)
        for category, metrics in metrics_by_category.items():
            model_label = self.experiment_name + "@" + category
            figures = figures_by_category[category]
            precision, recall, _ = metrics["precision_recall_curve"]
            f1 = metrics["f1_score"]
            auc = metrics["auc_score"]
            precision_recall_figure = figures["precision_recall"]
            precision_recall_ax = precision_recall_figure.gca()
            precision_recall_ax.plot(
                recall,
                precision,
                lw=1,
                alpha=0.8,
                label="%s (F1 = %0.4f)" % (model_label, f1),
            )
            fpr, tpr, _ = metrics["roc_curve"]
            roc_figure = figures["roc"]
            roc_ax = roc_figure.gca()
            roc_ax.plot(
                fpr, tpr, lw=1, alpha=0.8, label="%s (AUC = %0.4f)" % (model_label, auc)
            )
            figure_file_paths = self.save_figures(category, figures)
            metrics_by_category[category]["figures"] = figure_file_paths
        return metrics_by_category

class MNISTValidationDataProvider(ValidationDataProvider):
    def _get_confusion_matrix(self, model, X_test, y_test):
        validation_data = defaultdict(dict)
        confusion_matrix = defaultdict(dict)

        for Xi, yi in zip(X_test, y_test):
            predictions = model.predict_proba([xi])[0]  # only 1 input image
            top_predicted_target_ix = np.argmax(predictions)
            chosen_target = TARGETS[top_predicted_target_ix]
            confusion_matrix[chosen_target][prediction_type].append(xi)

    def _get_predictions(self, model, X_test, y_test):
        predictions = defaultdict(dict)
        for Xi, yi in zip(X_test, y_test):
            prediction = model.predict_proba([xi])[0]  # only 1 input image
            top_prediction = np.max(prediction)
            top_predicted_target_ix = np.argmax(prediction)
            chosen_target = TARGETS[top_predicted_target_ix]
            predictions[chosen_target].append(top_prediction)

        return predictions


def view_prediction(target, confidence):
    pass


TARGETS = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
PREDICTION_TYPES = ["True_Positive", "True_Negative", "False_Positive", "False_Negative"]
CONFIDENCE_LEVELS = ["Confident", "Unsure", "Not_Confident"]