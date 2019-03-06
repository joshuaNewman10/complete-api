import json

from keras import Input, Model, backend as K
from keras.layers import Lambda, Concatenate, Embedding, Dropout, LSTM
from keras.optimizers import Adam
from keras.regularizers import l2

from ai.common.model.experiment import Experiment


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0]) - K.square(y_pred[:, 1]) + margin))


def triplet_accuracy(_, y_pred):
    return K.mean(y_pred[:, 0] < y_pred[:, 1])


def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


class TextSentimentLSTMEmbeddingExperiment(Experiment):
    _name = 'text_sentiment_lstm_embedding_experiment'

    batch_size = 64
    char_embedding = 100
    token_embedding = 128
    document_embedding = 156
    initializer = 'glorot_uniform'
    regularizer = l2(1e-4)

    def __init__(self, experiment_directory, evaluator, max_document_length, max_token_length, vocab_size,
                 embedding_dimensions, embedding_weights):

        super(TextSentimentLSTMEmbeddingExperiment, self).__init__(experiment_directory, evaluator)
        self.max_document_length = max_document_length
        self.max_token_length = max_token_length
        self.vocab_size = vocab_size
        self.embedding_dimensions = embedding_dimensions
        self.embedding_weights = embedding_weights

    def get_model(self, **kwargs):
        encoder = self._get_encoder()
        x_anchor = Input(shape=(self.max_document_length,), name='anchor')
        x_related = Input(shape=(self.max_document_length,), name='related')
        x_unrelated = Input(shape=(self.max_document_length,), name='unrelated')

        h_anchor = encoder(x_anchor)
        h_related = encoder(x_related)
        h_unrelated = encoder(x_unrelated)

        related_dist = Lambda(euclidean_distance, name='pos_dist')([h_anchor, h_related])
        unrelated_dist = Lambda(euclidean_distance, name='neg_dist')([h_anchor, h_unrelated])

        inputs = [x_anchor, x_related, x_unrelated]
        distances = Concatenate()([related_dist, unrelated_dist])

        triplet = Model(inputs=inputs, outputs=distances)
        triplet.compile(optimizer=Adam(), loss=triplet_loss, metrics=[triplet_accuracy])
        return encoder, triplet

    def _get_encoder(self):
        document = Input(shape=(self.max_document_length,))
        X = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dimensions,
                      input_length=self.max_document_length,
                      trainable=False, weights=[self.embedding_weights])(document)
        X = Dropout(0.5)(X)
        X = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(X)
        model = Model(document, X)

        return model

    def get_params(self):
        encoder, classifier = self.get_model()

        return dict(
            model_architecture=json.dumps(dict(
                autoencoder=json.dumps(encoder.summary()),
                classifier=json.dumps(classifier.summary())
            )),
            loss="triplet_loss",
            optimizer="Adam",
            char_embedding=self.char_embedding,
            token_embedding=self.token_embedding,
            max_document_length=1,
            batch_size=64,
            max_token_length=2,
            vocab_size=3,
            document_embedding=156,
        )
