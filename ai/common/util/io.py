import json

from tqdm import tqdm


def load_jsonl(in_file_path):
    with open(in_file_path) as f:
        data = map(json.loads, f)
        for datum in tqdm(data, in_file_path):
            yield datum
