import logging
import os
import time

from argparse import ArgumentParser

import numpy as np
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from nltk import TweetTokenizer
from sklearn.externals import joblib
from tqdm import tqdm

from ai.blog_post_visualize_performance.experiment.text_triplet_hiearchal_rnn_embedding import TextSentimentTripletExperiment
from ai.blog_post_visualize_performance.provider.text_embedding_data import TextEmbeddingDataProvider
from ai.common.callback.keras import GradientDebugger
from ai.common.data_provider.text_triplet import TripletDataProvider
from ai.common.util.io import load_jsonl
from ai.text.transformer.triplet import TwitterHierarchicalTripletCharacterLevelTransformer, Vocabulary

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def get_tokenized_samples(training_data_file_path, tokenizer):
    tokenized_samples = []
    samples = load_jsonl(training_data_file_path)
    samples = samples

    for sample in tqdm(samples, desc="loading_tokens"):
        text = sample["text"]
        tokenized_sample = tokenizer.tokenize(text)
        tokenized_samples.append(tokenized_sample)

    return tokenized_samples


def get_max_length(tokenized_samples):
    document_lengths = sorted([len(tokens) for tokens in tokenized_samples])
    token_lengths = sorted([len(token) for tokens in tokenized_samples for token in tokens])
    max_document_length = np.percentile(document_lengths, 95)
    max_token_length = np.percentile(token_lengths, 95)
    return int(max_document_length), int(max_token_length)


def load_vocabulary(vocabulary_file_path):
    return joblib.load(vocabulary_file_path)


def run_experiment(experiment_directory, training_data_file_path, vocabulary_file_path=None, max_document_length=None, max_token_length=None):
    tokenizer = TweetTokenizer()

    if vocabulary_file_path:
        vocabulary = load_vocabulary(vocabulary_file_path)
    else:
        vocabulary = Vocabulary()
        tokenized_samples = get_tokenized_samples(training_data_file_path, tokenizer)
        all_tokens = (c for tokens in tokenized_samples for token in tokens for c in token)
        max_document_length, max_token_length = get_max_length(tokenized_samples)
        vocabulary.fit(all_tokens)
        joblib.dump(vocabulary, os.path.join(experiment_directory, "vocab.pkl"))


    vocab_size = len(vocabulary)
    experiment = TextSentimentTripletExperiment(experiment_directory=experiment_directory,
                                                max_document_length=max_document_length,
                                                max_token_length=max_token_length,
                                                vocab_size=vocab_size,
                                                evaluator=None)

    experiment_name = experiment.name
    max_document_length = experiment.max_document_length
    batch_size = experiment.batch_size
    max_token_length = experiment.max_token_length


    transformer = TwitterHierarchicalTripletCharacterLevelTransformer(vocabulary=vocabulary, tokenizer=tokenizer)

    log_dir = os.path.join(experiment_directory, experiment_name)
    encoder, triplet = experiment.get_model()

    training_data_provider = TripletDataProvider(
        jsonl_file=training_data_file_path,
        transformer=transformer,
        batch_size=batch_size,
        max_document_length=max_document_length,
        max_token_length=max_token_length,
        vocab_size=vocab_size
    )

    training_data_generators = training_data_provider.get_training_data_generators()
    train_generator = training_data_generators.training_data_generator
    val_generator = training_data_generators.validation_data_generator

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir=log_dir),
        ReduceLROnPlateau(factor=0.01, verbose=1)
    ]

    LOGGER.info("Beginning training for experiment %s", experiment)

    history = triplet.fit_generator(
        generator=iter(train_generator),
        steps_per_epoch=500,
        validation_data=iter(val_generator),
        validation_steps=500,
        epochs=5,
        verbose=1,
        callbacks=callbacks,
    )

    history = history.history
    experiment.persist_experiment(encoder, history)

    embedding_data_provider = TextEmbeddingDataProvider(
        experiment_directory,
        experiment_name,
        encoder,
        transformer,
        max_document_length,
        max_token_length
    )

    embedding_data_provider.store_embedding_data(training_data_provider.testing_samples)


def main(experiment_directory, training_data_file_path, vocabulary_file_path):
    start = time.time()
    run_experiment(experiment_directory, training_data_file_path, vocabulary_file_path)
    end = time.time()

    LOGGER.info("Ran experiment in %s seconds", end - start)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_directory", default="training_runs")
    parser.add_argument("--training_data_file_path", required=True)
    parser.add_argument("--vocabulary_file_path", required=False)

    args = parser.parse_args()
    main(args.experiment_directory, args.training_data_file_path, args.vocabulary_file_path)
