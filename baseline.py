"""A word-count logistic regression, which fine-tuning has to beat to be worth it."""

import re
from collections import Counter

import torch
from torch.nn import functional as F

WORD = re.compile(r"[a-z']+")


def words(text):
    return WORD.findall(text.lower())


def build_vocab(rows, size=5000):
    counts = Counter(word for row in rows for word in set(words(row["text"])))
    return {word: i for i, (word, _) in enumerate(counts.most_common(size))}


def encode(rows, vocab):
    x = torch.zeros(len(rows), len(vocab))
    for i, row in enumerate(rows):
        for word in set(words(row["text"])):
            if word in vocab:
                x[i, vocab[word]] = 1.0
    return x, torch.tensor([row["label"] for row in rows])


class BagOfWords:
    def __init__(self, train_rows, epochs=30, lr=0.01, weight_decay=0.01, seed=0):
        torch.manual_seed(seed)
        self.vocab = build_vocab(train_rows)
        x, y = encode(train_rows, self.vocab)
        self.linear = torch.nn.Linear(len(self.vocab), 2)
        optimizer = torch.optim.AdamW(self.linear.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            order = torch.randperm(len(x))
            for start in range(0, len(x), 64):
                batch = order[start : start + 64]
                loss = F.cross_entropy(self.linear(x[batch]), y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def accuracy(self, rows):
        x, y = encode(rows, self.vocab)
        return (self.linear(x).argmax(dim=-1) == y).float().mean().item()
