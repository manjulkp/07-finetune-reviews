"""Fine-tuning helpers: choose which layers train, then run the training loop."""

import time

import torch
from torch.nn import functional as F

MODES = ("head", "last2", "full")


def set_trainable(model, mode):
    """Freeze the pretrained body, leaving only the chosen parts trainable."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode != "full":
        for parameter in model.pre_classifier.parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    if mode == "last2":
        for layer in model.distilbert.transformer.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batches(rows, batch_size, generator=None):
    order = (
        torch.randperm(len(rows), generator=generator).tolist() if generator else range(len(rows))
    )
    order = list(order)
    for start in range(0, len(order), batch_size):
        yield [rows[i] for i in order[start : start + batch_size]]


def tokenize(tokenizer, rows, max_length):
    return tokenizer(
        [row["text"] for row in rows],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


@torch.no_grad()
def predict_logits(model, tokenizer, rows, max_length=128, batch_size=32):
    model.eval()
    # Sort by length so that each batch needs less padding, then restore the order.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    results = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        inputs = tokenize(tokenizer, [rows[i] for i in indices], max_length)
        logits = model(**inputs).logits
        for index, row_logits in zip(indices, logits):
            results[index] = row_logits
    return torch.stack(results)


def accuracy(model, tokenizer, rows, max_length=128, batch_size=32):
    logits = predict_logits(model, tokenizer, rows, max_length, batch_size)
    labels = torch.tensor([row["label"] for row in rows])
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def train_model(model, tokenizer, train_rows, val_rows, lr, epochs=2, batch_size=16, max_length=128, seed=0, log=print):
    """Fine-tune the model and leave it holding the weights from its best validation epoch."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = epochs * ((len(train_rows) + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1 - step / max(1, total_steps))
    )

    started = time.perf_counter()
    best_val, best_state = -1.0, None
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, count = 0.0, 0
        for batch in batches(train_rows, batch_size, generator):
            inputs = tokenize(tokenizer, batch, max_length)
            labels = torch.tensor([row["label"] for row in batch])
            loss = F.cross_entropy(model(**inputs).logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch)
            count += len(batch)
        val_acc = accuracy(model, tokenizer, val_rows, max_length)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / count, "val_acc": val_acc})
        if log:
            log(f"  epoch {epoch + 1}: loss {total_loss / count:.4f}, validation accuracy {val_acc:.1%}, {time.perf_counter() - started:.0f}s")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"history": history, "seconds": time.perf_counter() - started}
