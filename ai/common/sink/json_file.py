import atexit
import json
import logging

from ai.common.provider.encoder import NumpyEncoder
from ai.common.sink.base import Sink

LOGGER = logging.getLogger(__name__)


class JSONFileSink(Sink):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(self.file_path, 'a')

        atexit.register(self._cleanup)

    def transform(self, item):
        return json.dumps(item, cls=NumpyEncoder)

    def sink(self, item, **kwargs):
        self.file.write(item)

    def flush(self):
        self.file.flush()

    def _cleanup(self):
        LOGGER.debug('closing file....')
        self.file.close()
