import logging
import numpy as np
import random

from collections import namedtuple, defaultdict

from ai.common.data_provider.base import TrainingDataProvider
from ai.common.model.sample import TextSample
from ai.common.model.training_data_generators import TrainingDataGenerators
from ai.common.util.io import load_jsonl

Triplet = namedtuple('Triplet', ('anchor', 'related', 'unrelated'))

LOGGER = logging.getLogger(__name__)
CLASS_NAME_NEGATIVE = "negative"
CLASS_NAME_POSITIVE = "positive"


class TripletProvider:
    def __init__(self, samples, shuffle=False):
        self._samples_by_label = self._group_samples_by_label(samples)
        self._shuffle = shuffle

    def __iter__(self):
        if self._shuffle:
            for samples in self._samples_by_label.values():
                random.shuffle(samples)

        yield from self._triplet_generator()

    def _group_samples_by_label(self, samples):
        samples_by_label = defaultdict(list)

        for sample in samples:
            class_name = sample.class_name
            samples_by_label[class_name].append(sample)

        return samples_by_label

    def _get_samples_by_sentiment(self, sentiment):
        return self._samples_by_label.get(sentiment, [])

    def _triplet_generator(self):
        positive_samples = self._get_samples_by_sentiment(CLASS_NAME_POSITIVE)
        negative_samples = self._get_samples_by_sentiment(CLASS_NAME_NEGATIVE)
        LOGGER.info("Num pos samples %s Num negative samples %s", len(positive_samples), len(negative_samples))

        while True:
            randint = random.randint(1, 2)
            if randint == 1:  # positive
                anchor_sample, related_sample = random.sample(positive_samples, 2)
                unrelated_sample = random.choice(negative_samples)
                yield Triplet(anchor_sample, related_sample, unrelated_sample)
            else:  # negative
                anchor_sample, related_sample = random.sample(negative_samples, 2)
                unrelated_sample = random.choice(positive_samples)
                yield Triplet(anchor_sample, related_sample, unrelated_sample)


class TripletBatchCharacterLevelGenerator:
    def __init__(self, provider, transformer, max_document_length, max_token_length, vocab_size, batch_size):
        self._provider = provider
        self._transformer = transformer
        self._batch_size = batch_size
        self._max_document_length = max_document_length
        self._max_token_length = max_token_length
        self._vocab_size = vocab_size

    def __iter__(self):
        while True:
            targets = np.ones(self._batch_size)
            anchor_batch = self._init_batch()
            related_batch = self._init_batch()
            unrelated_batch = self._init_batch()

            for i, triplet in enumerate(self._provider):
                idx = i % self._batch_size
                anchor, related, unrelated = self._transformer.transform(triplet)

                anchor_batch[idx, :anchor.shape[0], :anchor.shape[1]] = \
                    anchor[:self._max_document_length, :self._max_token_length]

                related_batch[idx, :related.shape[0], :related.shape[1]] = \
                    related[:self._max_document_length, :self._max_token_length]

                unrelated_batch[idx, :unrelated.shape[0], :unrelated.shape[1]] = \
                    unrelated[:self._max_document_length, :self._max_token_length]

                if idx == self._batch_size - 1:
                    yield [anchor_batch, related_batch, unrelated_batch], targets
                    anchor_batch = self._init_batch()
                    related_batch = self._init_batch()
                    unrelated_batch = self._init_batch()

    def _init_batch(self):
        return np.zeros((self._batch_size, self._max_document_length, self._max_token_length))


class TripletBatchWordLevelGenerator:
    def __init__(self, provider, transformer, max_document_length, vocab_size, batch_size):
        self._provider = provider
        self._transformer = transformer
        self._batch_size = batch_size
        self._max_document_length = max_document_length
        self._vocab_size = vocab_size

    def __iter__(self):
        while True:
            targets = np.ones(self._batch_size)
            anchor_batch = self._init_batch()
            related_batch = self._init_batch()
            unrelated_batch = self._init_batch()

            for i, triplet in enumerate(self._provider):
                idx = i % self._batch_size
                anchor, related, unrelated = self._transformer.transform(triplet)

                anchor_batch[idx, :anchor.shape[0]] = \
                    anchor[:self._max_document_length]

                related_batch[idx, :related.shape[0]] = \
                    related[:self._max_document_length]

                unrelated_batch[idx, :unrelated.shape[0]] = \
                    unrelated[:self._max_document_length]

                if idx == self._batch_size - 1:
                    yield [anchor_batch, related_batch, unrelated_batch], targets
                    anchor_batch = self._init_batch()
                    related_batch = self._init_batch()
                    unrelated_batch = self._init_batch()

    def _init_batch(self):
        return np.zeros((self._batch_size, self._max_document_length))


class TextTripletDataProvider(TrainingDataProvider):
    TRAIN_SPLIT = 0.6
    TEST_SPLIT = 0.2
    VAL_SPLIT = 0.2

    training_samples = []
    testing_samples = []
    validation_samples = []

    triplet_batch_generator = TripletBatchWordLevelGenerator

    def __init__(self, jsonl_file, transformer, max_document_length, vocab_size, batch_size):
        self._jsonl_file = jsonl_file
        self._transformer = transformer
        self._max_document_length = max_document_length
        self._vocab_size = vocab_size
        self._batch_size = batch_size

        samples = load_jsonl(self._jsonl_file)
        samples = map(self.deserialize_sample, samples)

        self._samples = list(samples)
        self._num_samples = len(self._samples)

    def get_training_data_generators(self):
        self.training_samples = list(self._get_training_samples(self._samples))
        self.testing_samples = list(self._get_testing_samples(self._samples))
        self.validation_samples = list(self._get_validation_samples(self._samples))

        training_provider = TripletProvider(self.training_samples)
        testing_provider = TripletProvider(self.testing_samples)
        validation_provider = TripletProvider(self.validation_samples)

        training_generator = self._get_triplet_batch_generator(provider=training_provider)
        testing_generator = self._get_triplet_batch_generator(provider=testing_provider)
        validation_generator = self._get_triplet_batch_generator(provider=validation_provider)

        return TrainingDataGenerators(
            training_generator,
            testing_generator,
            validation_generator
        )

    def deserialize_sample(self, sample):
        return TextSample(text=sample["text"], class_name=sample["target"], source=sample, vector=None)

    def _get_training_samples(self, samples):
        training_split = int(self.TRAIN_SPLIT * self._num_samples)
        return samples[0:training_split]

    def _get_testing_samples(self, samples):
        validation_split = self.TRAIN_SPLIT + self.VAL_SPLIT
        training_split = int(self.TRAIN_SPLIT * self._num_samples)
        validation_split = int(validation_split * self._num_samples)
        return samples[training_split:validation_split]

    def _get_validation_samples(self, samples):
        validation_split = self.TRAIN_SPLIT + self.VAL_SPLIT
        validation_split = int(validation_split * self._num_samples)
        return samples[validation_split:self._num_samples]

    def _get_triplet_batch_generator(self, provider):
        raise NotImplementedError()


class TextCharacterLevelTripletDataProvider(TextTripletDataProvider):
    triplet_batch_generator = TripletBatchCharacterLevelGenerator

    def __init__(self, jsonl_file, transformer, max_document_length, vocab_size, batch_size, max_token_length):
        self._max_token_length = max_token_length

        super(TextCharacterLevelTripletDataProvider, self).__init__(
            jsonl_file=jsonl_file,
            transformer=transformer,
            max_document_length=max_document_length,
            vocab_size=vocab_size,
            batch_size=batch_size
        )

    def _get_triplet_batch_generator(self, provider):
        return self.triplet_batch_generator(provider=provider,
                                            transformer=self._transformer,
                                            max_document_length=self._max_document_length,
                                            vocab_size=self._vocab_size,
                                            max_token_length=self._max_token_length,
                                            batch_size=self._batch_size)


class TextWordLevelTripletDataProvider(TextTripletDataProvider):
    triplet_batch_generator = TripletBatchWordLevelGenerator

    def _get_triplet_batch_generator(self, provider):
        return self.triplet_batch_generator(provider=provider,
                                            transformer=self._transformer,
                                            max_document_length=self._max_document_length,
                                            vocab_size=self._vocab_size,
                                            batch_size=self._batch_size)
