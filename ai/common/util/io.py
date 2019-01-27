import json

from itertools import islice
from tqdm import tqdm


def load_jsonl(in_file_path):
    with open(in_file_path) as f:
        data = map(json.loads, f)
        #data = islice(data, 0, 100)
        for datum in tqdm(data, in_file_path):
            yield datum
