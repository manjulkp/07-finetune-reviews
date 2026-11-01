"""Download small samples of the Yelp and Amazon review datasets.

The datasets are streamed, so the full files (hundreds of MB) aren't downloaded.
Project 9 also uses these files, including the extra unlabeled sample.
"""

import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path("data")
SIZES = {"train": 4000, "val": 500, "test": 1000, "unlabeled": 6000}
SHUFFLE_BUFFER = 5000


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def yelp_row(item):
    return {"text": item["text"], "label": int(item["label"])}


def amazon_row(item):
    return {"text": f"{item['title']}. {item['content']}", "label": int(item["label"])}


def take(stream, count, convert):
    rows = []
    for item in stream:
        rows.append(convert(item))
        if len(rows) == count:
            return rows
    raise ValueError(f"expected {count} rows but the stream ended after {len(rows)}")


def main():
    yelp_train = load_dataset("fancyzhx/yelp_polarity", split="train", streaming=True)
    yelp_train = iter(yelp_train.shuffle(seed=0, buffer_size=SHUFFLE_BUFFER))
    # The same iterator feeds all three, so that these sets can't share a review.
    write_jsonl(DATA_DIR / "yelp_train.jsonl", take(yelp_train, SIZES["train"], yelp_row))
    write_jsonl(DATA_DIR / "yelp_val.jsonl", take(yelp_train, SIZES["val"], yelp_row))
    write_jsonl(DATA_DIR / "yelp_unlabeled.jsonl", take(yelp_train, SIZES["unlabeled"], yelp_row))

    yelp_test = load_dataset("fancyzhx/yelp_polarity", split="test", streaming=True)
    yelp_test = iter(yelp_test.shuffle(seed=0, buffer_size=SHUFFLE_BUFFER))
    write_jsonl(DATA_DIR / "yelp_test.jsonl", take(yelp_test, SIZES["test"], yelp_row))

    amazon_test = load_dataset("fancyzhx/amazon_polarity", split="test", streaming=True)
    amazon_test = iter(amazon_test.shuffle(seed=0, buffer_size=SHUFFLE_BUFFER))
    write_jsonl(DATA_DIR / "amazon_test.jsonl", take(amazon_test, SIZES["test"], amazon_row))

    for path in sorted(DATA_DIR.glob("*.jsonl")):
        rows = read_jsonl(path)
        positive = sum(row["label"] for row in rows) / len(rows)
        print(f"{path.name}: {len(rows)} reviews, {positive:.1%} positive")


if __name__ == "__main__":
    main()
