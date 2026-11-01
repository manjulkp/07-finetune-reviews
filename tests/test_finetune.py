import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from baseline import BagOfWords, build_vocab, words
from finetune import accuracy, count_trainable, predict_logits, set_trainable, train_model
from prepare_data import amazon_row, read_jsonl, take, write_jsonl, yelp_row


class StubTokenizer:
    """Maps "good" and "bad" to fixed IDs, so that a tiny model can learn sentiment."""

    def __call__(self, texts, truncation, max_length, padding, return_tensors):
        tokens = [[{"good": 5, "bad": 6}.get(w, 7) for w in t.split()][:max_length] for t in texts]
        longest = max(len(t) for t in tokens)
        ids = [t + [0] * (longest - len(t)) for t in tokens]
        mask = [[1] * len(t) + [0] * (longest - len(t)) for t in tokens]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def tiny_model():
    torch.manual_seed(0)
    config = DistilBertConfig(vocab_size=10, n_layers=3, n_heads=2, dim=16, hidden_dim=32, max_position_embeddings=32, num_labels=2)
    return DistilBertForSequenceClassification(config)


ROWS = [{"text": "good day", "label": 1}, {"text": "bad day", "label": 0}] * 20


def trainable_names(model):
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_head_mode_only_trains_the_head():
    model = tiny_model()
    set_trainable(model, "head")
    names = trainable_names(model)
    assert names and all(n.startswith(("pre_classifier", "classifier")) for n in names)


def test_last2_mode_adds_only_the_last_two_layers():
    model = tiny_model()
    set_trainable(model, "last2")
    names = trainable_names(model)
    assert any("layer.1." in n for n in names) and any("layer.2." in n for n in names)
    assert not any("layer.0." in n or "embeddings" in n for n in names)


def test_full_mode_trains_everything():
    model = tiny_model()
    set_trainable(model, "head")
    head_count = count_trainable(model)
    set_trainable(model, "full")
    assert count_trainable(model) == sum(p.numel() for p in model.parameters()) > head_count


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        set_trainable(tiny_model(), "everything")


def test_training_learns_and_only_changes_trainable_weights():
    model = tiny_model()
    set_trainable(model, "head")
    frozen = model.distilbert.embeddings.word_embeddings.weight.clone()
    head = model.classifier.weight.clone()
    train_model(model, StubTokenizer(), ROWS, ROWS, lr=0.01, epochs=4, batch_size=8, log=None)
    assert torch.equal(frozen, model.distilbert.embeddings.word_embeddings.weight)
    assert not torch.equal(head, model.classifier.weight)
    assert accuracy(model, StubTokenizer(), ROWS) == 1.0


def test_predictions_stay_in_input_order_despite_length_sorting():
    model = tiny_model().eval()
    rows = [{"text": "good day and more words", "label": 1}, {"text": "bad", "label": 0}]
    together = predict_logits(model, StubTokenizer(), rows, batch_size=1)
    single = predict_logits(model, StubTokenizer(), rows[:1], batch_size=1)
    assert torch.allclose(together[0], single[0], atol=1e-5)


def test_row_converters():
    assert yelp_row({"text": "nice", "label": 1}) == {"text": "nice", "label": 1}
    assert amazon_row({"title": "Bad", "content": "Broke fast", "label": 0}) == {"text": "Bad. Broke fast", "label": 0}


def test_take_raises_when_the_stream_is_too_short():
    assert take(iter([1, 2, 3]), 2, str) == ["1", "2"]
    with pytest.raises(ValueError, match="stream ended"):
        take(iter([1]), 2, str)


def test_jsonl_round_trip(tmp_path):
    rows = [{"text": "café", "label": 1}]
    write_jsonl(tmp_path / "a" / "b.jsonl", rows)
    assert read_jsonl(tmp_path / "a" / "b.jsonl") == rows


def test_baseline_learns_word_sentiment():
    rows = [{"text": "great tasty food", "label": 1}, {"text": "awful rude staff", "label": 0}] * 30
    assert BagOfWords(rows, epochs=5).accuracy(rows) == 1.0
    assert words("It's GOOD!") == ["it's", "good"]
    assert len(build_vocab(rows, size=2)) == 2
