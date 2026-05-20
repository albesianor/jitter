# Jitter

#### Goal
Develop a natural language processing tool that measures "world instability" from news headlines.

#### User interface
- User provides RSS feeds.
- Tool computes each headline "instability" score between 0 and 1.
- Interface returns a score distribution and summary metrics.
- User can give feedback on each headline.
- Model retrains with user feedback.

## Implementation
### Data collection
- Load feed urls from disk.
- Download and parse feeds (`feedparser`).
- Clean feed and concatenate title and description.

### Naive scoring
- Use a BERT model to embed sequences.
- Compute the centroid of the embeddings of some preset "instability" headlines.
- Score the embeddings against the centroid by cosine-similarity.

This is easy to implement ([00_naive.ipynb](00_naive.ipynb)) and fast, but has two main issues:
- relevant and irrelevant headlines are treated similarly (irrelevant headlines contribute to the overall score),
- having a single centroid doesn't allow to capture nuances.

### Improved model
Three-stages model:
```
cleaned headlines --(BERT)--> embedded vectors --(NN1)--> relevant vectors --(NN2)--> score
```

Training for `NN1` and `NN2` is done by knowledge distillation from a zero-shot BERT classifier (on a separate dataset initially, then taking into account user feedback).

### Notebooks
| Function | Notebook |
|---|---|
| preliminary exploration and naive model | [00_naive.ipynb](00_naive.ipynb)                     |
| data collection                         | [01_data_collection.ipynb](01_data_collection.ipynb) |
| relevance labeling (for `NN1` training) | [02_labeling.ipynb](02_labeling.ipynb)               |
| `NN1` implementation and training       | [03_filter_training.ipynb](03_filter_training.ipynb) |
| scoring (for `NN2` training)            | |
| `NN2` implementation and training       | |
| complete inference pipeline             | |