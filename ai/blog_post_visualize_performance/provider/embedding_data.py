import numpy as np
import os


class EmbeddingDataProvider:
    def __init__(self, out_dir, model_name, encoder, transformer):
        self._encoder = encoder
        self._transformer = transformer
        self._out_dir = out_dir
        self._model_name = model_name

        self._model_out_path = out_dir
        self._vector_path = os.path.join(self._model_out_path, "vectors.tsv")
        self._metadata_path = os.path.join(self._model_out_path, "metadata.tsv")
        self._vector_path = os.path.join(self._model_out_path, "vectors.tsv")

    def store_embedding_data(self, samples):
        vectors = self._get_vectors(samples)
        self._sink_vectors(vectors)
        self._sink_metadata(samples, vectors)

    def _get_vectors(self, samples):
        vectors = []

        for sample in samples:
            vector = self._get_vector(sample)
            vectors.append(vector)

        return np.array(vectors)

    def _sink_vectors(self, vectors):
        np.savetxt(self._vector_path, vectors, delimiter="\t")
        return vectors

    def _sink_metadata(self, samples, vectors):
        raise NotImplementedError()

    def _get_vector(self, text):
        raise NotImplementedError()
