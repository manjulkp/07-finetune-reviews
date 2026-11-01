# Fine-tuning results

Trained on 4,000 Yelp reviews. Validation: 500. Test: 1,000 Yelp and 1,000 Amazon reviews.

| Approach | Trainable parameters | Validation | Yelp test | Amazon test | Training time |
| --- | --- | --- | --- | --- | --- |
| Word counts (baseline) | 10,002 | 86.6% | 87.8% | 77.5% | Under a minute |
| Head only | 592,130 | 85.4% | 86.0% | 83.5% | 21.5 min |
| Last 2 layers | 14,767,874 | 88.4% | 88.7% | 89.8% | 45.8 min |
| Whole model | 66,955,010 | 88.6% | 90.5% | 89.3% | 51.7 min |
