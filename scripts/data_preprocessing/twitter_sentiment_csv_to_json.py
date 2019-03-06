import csv

from argparse import ArgumentParser
from collections import defaultdict
from random import shuffle

from tqdm import tqdm

from ai.common.sink.jsonl_file import JSONLFileSink

SENTIMENT_MAP = {
    "0": "negative",
    "2": "neutral",
    "4": "positive"
}


def get_sentiment_from_target(target):
    return SENTIMENT_MAP[target]


def main(in_file_path, out_file_path):
    sink = JSONLFileSink(out_file_path)
    all_data = []
    counts = defaultdict(int)

    with open(in_file_path, mode='r', encoding="latin-1") as infile:
        reader = csv.DictReader(infile, fieldnames=["target", "id", "date", "flag", "user", "text"])
        for row in tqdm(reader):
            sentiment = get_sentiment_from_target(row["target"])
            row["target"] = sentiment
            counts[sentiment] += 1
            all_data.append(row)

        shuffle(all_data)
        print(counts)
        for row in all_data:
            sink.receive(row)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_file_path", required=True)
    parser.add_argument("--out_file_path", required=True)

    args = parser.parse_args()
    main(args.in_file_path, args.out_file_path)
