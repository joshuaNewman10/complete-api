import cv2
import logging
import os

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

from ai.blog_post_visualize_performance.provider.embedding_data import EmbeddingDataProvider

LOGGER = logging.getLogger(__name__)


class ImageEmbeddingDataProvider(EmbeddingDataProvider):
    db_scan_eps = 0.25

    metadata_fields = ["class_name", "cluster"]
    db_scan_min_samples = 2

    def __init__(self, out_dir, model_name, encoder, transformer):
        super(ImageEmbeddingDataProvider, self).__init__(out_dir=out_dir, model_name=model_name, encoder=encoder,
                                                         transformer=transformer)
        self._sprite_path = os.path.join(self._model_out_path, "sprite.png")

    def _sink_metadata(self, samples, vectors):
        dbscan = DBSCAN(metric='precomputed', min_samples=self.db_scan_min_samples, eps=self.db_scan_eps)

        LOGGER.debug('Computing distances')
        distances = pairwise_distances(vectors, n_jobs=6)
        LOGGER.debug('Computing dbscan')
        clusters = dbscan.fit_predict(distances)

        with open(self._metadata_path, 'w') as f:
            f.write('\t'.join(self.metadata_fields) + '\n')
            for sample, cluster in zip(samples, clusters):
                class_name = sample.class_name
                cluster = str(cluster)
                f.write('\t'.join([str(class_name), str(cluster)]) + '\n')

        self._sink_sprite(samples)

    def _get_vector(self, sample):
        image = sample.image
        image = self._transformer.transform(image)
        return self._encoder.predict(np.array([image]))[0]

    def _sink_sprite(self, samples):
        images = []

        for sample in samples:
            image = sample.image
            image = cv2.resize(image, (28, 28))
            images.append(image)

        images = np.array(images)

        if len(images.shape) == 3:
            images = np.tile(images[..., np.newaxis], (1, 1, 1, 3))

        images = images.astype(np.float32)
        min = np.min(images.reshape((images.shape[0], -1)), axis=1)
        images = (images.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
        max = np.max(images.reshape((images.shape[0], -1)), axis=1)
        images = (images.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

        n = int(np.ceil(np.sqrt(images.shape[0])))
        padding = ((0, n ** 2 - images.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (images.ndim - 3)
        images = np.pad(images, padding, mode='constant',
                        constant_values=0)
        # Tile the individual thumbnails into an image.
        images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3)
                                                                     + tuple(range(4, images.ndim + 1)))
        images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])
        images = (images * 255).astype(np.uint8)
        cv2.imwrite(self._sprite_path, images)
