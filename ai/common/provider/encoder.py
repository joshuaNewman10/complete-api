import json

import numpy


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.int):
            return int(obj)
        elif isinstance(obj, numpy.int32):
            return int(obj)
        elif isinstance(obj, numpy.int64):
            return int(obj)
        elif isinstance(obj, numpy.float):
            return float(obj)
        elif isinstance(obj, numpy.float32):
            return float(obj)
        elif isinstance(obj, numpy.float64):
            return float(obj)

        return json.JSONEncoder.default(self, obj)
