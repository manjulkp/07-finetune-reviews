# Project 7: Fine-tune on customer reviews

Fine-tunes DistilBERT to classify review sentiment. It trains on Yelp reviews, then
tests on both Yelp and Amazon reviews to see how well the model handles a domain it
never saw.

| Approach | What trains |
| --- | --- |
| Word counts (baseline) | A simple logistic regression, with no pretrained model |
| Head only | Only the new classification layers |
| Last 2 layers | The classification layers and the last 2 transformer layers |
| Whole model | Everything |

The data is 4,000 training, 500 validation and 1,000 test reviews per domain.

## Results

Training the whole model gave the best Yelp accuracy, 90.5%. On Amazon reviews, which no
model trained on, the word-count baseline fell to 77.5%, while the last-2-layers and
whole models both stayed near 89%.

| Approach | Trainable parameters | Validation | Yelp test | Amazon test | Training time |
| --- | --- | --- | --- | --- | --- |
| Word counts (baseline) | 10,002 | 86.6% | 87.8% | 77.5% | Under a minute |
| Head only | 592,130 | 85.4% | 86.0% | 83.5% | 21.5 min |
| Last 2 layers | 14,767,874 | 88.4% | 88.7% | 89.8% | 45.8 min |
| Whole model | 66,955,010 | 88.6% | 90.5% | 89.3% | 51.7 min |

Each test set has 1,000 reviews and each approach ran once, so treat small differences
as ties, such as the two best models on Amazon. The chart is `outputs\comparison.png`.

## Run it

```powershell
uv sync
uv run python prepare_data.py     # stream small samples of each dataset
uv run python experiment.py       # downloads DistilBERT the first time
uv run pytest
```

The best model by validation accuracy is saved to `outputs\best_model`, which Project 9
uses as its teacher.

## Files

| File | What it does |
| --- | --- |
| `prepare_data.py` | Downloads the review samples |
| `finetune.py` | Chooses which layers train, and runs training |
| `baseline.py` | Word-count baseline |
| `experiment.py` | Runs all four approaches and writes the results |
| `tests/` | Uses a tiny random model, so no download is needed |
