import logging
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

from ai.blog_post_visualize_performance.provider.embedding_data import EmbeddingDataProvider

LOGGER = logging.getLogger(__name__)


class TextEmbeddingDataProvider(EmbeddingDataProvider):
    db_scan_eps = 0.25
    db_scan_min_samples = 2

    metadata_fields = ["text", "class_name", "cluster"]

    def __init__(self, out_dir, model_name, encoder, transformer, max_doc_length, max_token_length):
        super(TextEmbeddingDataProvider, self).__init__(out_dir=out_dir, model_name=model_name, encoder=encoder,
                                                        transformer=transformer)
        self._max_doc_length = max_doc_length
        self._max_token_length = max_token_length

    def _sink_metadata(self, samples, vectors):
        dbscan = DBSCAN(metric='precomputed', min_samples=self.db_scan_min_samples, eps=self.db_scan_eps)

        LOGGER.debug('Computing distances')
        distances = pairwise_distances(vectors, n_jobs=6)

        LOGGER.debug('Computing dbscan')
        clusters = dbscan.fit_predict(distances)

        with open(self._metadata_path, 'w') as f:
            f.write('\t'.join(self.metadata_fields) + '\n')
            for sample, cluster in zip(samples, clusters):
                sentiment = sample.class_name
                text = sample.text.replace('\n', ' ')
                cluster = str(cluster)
                f.write('\t'.join([text, sentiment, cluster]) + '\n')

    def _get_vector(self, sample):
        text = sample.text
        vector = np.zeros((1, self._max_doc_length))
        codes = self._transformer._transform(text.lower())
        vector[0, :codes.shape[0]] = codes[:self._max_doc_length]
        return self._encoder.predict(vector)[0]
