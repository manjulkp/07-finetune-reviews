"""Compare how much of DistilBERT to fine-tune, and test on both review domains."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import BagOfWords
from finetune import accuracy, count_trainable, set_trainable, train_model
from prepare_data import read_jsonl

MODEL_NAME = "distilbert-base-uncased"
# The whole-model learning rate is small, because large updates would wreck the
# pretrained weights. A new head starts from nothing, so it can go faster.
RUNS = [
    ("Head only", "head", 1e-3),
    ("Last 2 layers", "last2", 5e-5),
    ("Whole model", "full", 2e-5),
]

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.7,
    }
)


def plot(rows, path):
    rows = rows[::-1]
    names = [row["name"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(rows) + 1.6), sharey=True, sharex=True)
    for ax, key, title in (
        (axes[0], "yelp_test", "Yelp test reviews (same domain as training)"),
        (axes[1], "amazon_test", "Amazon test reviews (a domain it never saw)"),
    ):
        values = [row[key] for row in rows]
        ax.scatter(values, names, color="#2a78d6", s=46, edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
        for value, name in zip(values, names):
            ax.annotate(f"{value:.1%}", (value, name), xytext=(0, 8), textcoords="offset points", ha="center", color="#52514e", fontsize=9)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(title)
        ax.margins(x=0.1, y=0.3)
    fig.supxlabel("Accuracy", color="#52514e", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--model", default=MODEL_NAME, help="pretrained model to fine-tune")
    args = parser.parse_args(argv)

    data = Path(args.data)
    if not (data / "yelp_train.jsonl").exists():
        parser.error(f"{data}/yelp_train.jsonl not found. Run prepare_data.py first.")
    train_rows = read_jsonl(data / "yelp_train.jsonl")
    val_rows = read_jsonl(data / "yelp_val.jsonl")
    yelp_test = read_jsonl(data / "yelp_test.jsonl")
    amazon_test = read_jsonl(data / "amazon_test.jsonl")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bag = BagOfWords(train_rows)
    rows = [
        {
            "name": "Word counts (baseline)",
            "trainable": sum(p.numel() for p in bag.linear.parameters()),
            "val": bag.accuracy(val_rows),
            "yelp_test": bag.accuracy(yelp_test),
            "amazon_test": bag.accuracy(amazon_test),
            "seconds": None,
        }
    ]
    print(rows[0], flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    best_model, best_val = None, -1.0
    for name, mode, lr in RUNS:
        print(name, flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        set_trainable(model, mode)
        result = train_model(
            model, tokenizer, train_rows, val_rows, lr, args.epochs,
            log=lambda text: print(text, flush=True),
        )  # fmt: skip
        val = max(row["val_acc"] for row in result["history"])
        rows.append(
            {
                "name": name,
                "trainable": count_trainable(model),
                "val": val,
                "yelp_test": accuracy(model, tokenizer, yelp_test),
                "amazon_test": accuracy(model, tokenizer, amazon_test),
                "seconds": result["seconds"],
            }
        )
        print(rows[-1], flush=True)
        if val > best_val:
            best_model, best_val = model, val

    # Project 9 uses this as its teacher. It's picked by validation accuracy.
    best_model.save_pretrained(out / "best_model")
    tokenizer.save_pretrained(out / "best_model")

    lines = [
        "# Fine-tuning results",
        "",
        f"Trained on {len(train_rows):,} Yelp reviews. Validation: {len(val_rows):,}. "
        f"Test: {len(yelp_test):,} Yelp and {len(amazon_test):,} Amazon reviews.",
        "",
        "| Approach | Trainable parameters | Validation | Yelp test | Amazon test | Training time |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        seconds = "Under a minute" if row["seconds"] is None else f"{row['seconds'] / 60:.1f} min"
        lines.append(
            f"| {row['name']} | {row['trainable']:,} | {row['val']:.1%} | {row['yelp_test']:.1%} | {row['amazon_test']:.1%} | {seconds} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    plot(rows, out / "comparison.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
