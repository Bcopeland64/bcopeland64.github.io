# SI Demo Walkthrough — Module 7A Integration Lab

**Audience:** Student Instructors preparing to deliver the 90-minute Day 4 integration session  
**Source model:** Fine-tuned app-review sentiment classifier from Lab 7A (loaded from HuggingFace Hub)  
**Target corpus:** 20 Newsgroups forum posts

---

> **How to use this document**  
> Each segment below explains *what every line is doing* in plain language, then closes with the **full code block** for that segment so you can copy-paste into your notebook cleanly. Run segments in order — later ones depend on earlier ones.

---

## Segment 0 — Environment Setup

### What this segment does
Installs the required libraries and loads all the Python modules needed for the integration lab.
It also defines the key constants — the HuggingFace Hub model repo path, the newsgroup categories to use,
and the tuning knobs (max token length, batch size) — in one place at the top so they are easy to change.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `# !pip install transformers ...` | A commented-out install command. Remove the `#` if your environment is missing any package. The `-q` flag suppresses most of the installation output to keep the notebook tidy. |
| `import torch.nn.functional as F` | Loads the functional toolkit from PyTorch. We use it to call `F.softmax`, which converts raw model scores into probabilities. |
| `from transformers import AutoTokenizer, AutoModelForSequenceClassification` | The two HuggingFace classes we need: the tokenizer (converts text to numbers) and the model (converts numbers to predictions). `Auto` means HuggingFace will figure out the right architecture from the Hub repo's config file. |
| `from sklearn.datasets import fetch_20newsgroups` | Loads scikit-learn's built-in downloader for the 20 Newsgroups dataset. The first call downloads the data from the internet and caches it; subsequent calls load from disk. |
| `HF_USERNAME = "your-hf-username"` | A placeholder constant. Learners replace this with the HuggingFace username that ran Lab 7A. Storing it as a constant means only one line needs to change. |
| `MODEL_REPO = f"{HF_USERNAME}/m7-app-review-sentiment"` | Builds the full Hub address: `"username/m7-app-review-sentiment"`. This is exactly what you would type on huggingface.co to find the repo. |
| `CATEGORIES = [...]` | The four newsgroup categories to load. Each is a string that matches one of the 20 Newsgroups folder names. We chose these four because they span very different writing styles — sports, politics, tech, and medicine. |
| `POSTS_PER_CATEGORY = 200` | Caps how many posts we take from each category. 200 × 4 = 800 posts total — enough to see patterns but fast enough to run in a demo session. |
| `MAX_LENGTH = 128` | How many tokens the tokenizer sends to the model per post. Newsgroup posts can be thousands of words; we truncate to 128, which is what the model was trained on with app reviews. This deliberate mismatch is part of what we will analyze. |
| `BATCH_SIZE = 32` | How many posts we run through the model at one time during inference. Larger batches are faster but need more memory. 32 is a safe default. |
| `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Picks GPU if one is available, otherwise falls back to CPU. The model and all input tensors must be on the same device. |

### Full code block

```python
# Uncomment if packages are not installed in your environment
# !pip install transformers datasets scikit-learn seaborn matplotlib torch -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.datasets import fetch_20newsgroups

HF_USERNAME = "your-hf-username"          # <-- CHANGE THIS
MODEL_REPO  = f"{HF_USERNAME}/m7-app-review-sentiment"

CATEGORIES = [
    "rec.sport.hockey",
    "talk.politics.guns",
    "comp.graphics",
    "sci.med",
]
POSTS_PER_CATEGORY = 200
MAX_LENGTH = 128
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Model repo: {MODEL_REPO}")
print("All imports successful!")
```

---

## Segment 1 — Load Model from the HuggingFace Hub

### What this segment does
Loads the tokenizer and model that were trained in Lab 7A directly from the HuggingFace Hub.
No checkpoint files are needed — one line per component fetches everything: weights, configuration, and label names.
It also demonstrates the `id2label` pattern, which is how we recover human-readable class names from the saved model config.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `AutoTokenizer.from_pretrained(MODEL_REPO)` | Downloads the tokenizer (vocabulary file and splitting rules) that matches the model in the Hub repo. `Auto` reads the repo's `config.json` to figure out which tokenizer class to use. After the first download the result is cached on disk. |
| `AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)` | Downloads the model weights (the `.safetensors` or `.bin` file) and rebuilds the model architecture in memory. The classification head (the layer that outputs one score per class) is loaded with the fine-tuned weights from Lab 7A, not random weights. |
| `model.to(device)` | Moves all model parameters from RAM to the device (GPU if available). After this call, the model lives on the GPU and expects input tensors on the GPU too. |
| `model.eval()` | Switches the model to evaluation mode. This disables two training-only behaviors: (1) dropout layers stop randomly zeroing neurons, and (2) batch normalization uses running statistics instead of batch statistics. Both changes make predictions deterministic and accurate. |
| `id2label = model.config.id2label` | Reads the label mapping from the model's config file. When we pushed the model in Lab 7A, HuggingFace saved the mapping `{0: "negative", 1: "neutral", 2: "positive"}` (or similar) inside `config.json`. Reading it here means we do not hardcode label names in the application code — if the model is retrained with different labels, this line automatically picks up the new mapping. |
| `label2id = model.config.label2id` | The reverse mapping — useful when we need to look up a class ID by name. |
| `model.num_parameters()` | Returns the total number of trainable weights in the model (roughly 66 million for DistilBERT). |

### Why model registries matter
Committing a 265 MB checkpoint file to git is an anti-pattern:
- It bloats the repository history permanently.
- Every clone downloads the binary, even for developers who never run inference.
- Versioning and rollback are awkward with binary blobs.

The HuggingFace Hub is a public example of a **model registry** — an addressed, versioned store where models live separately from code. Production teams use the same pattern with internal tools like MLflow Model Registry, W&B Artifacts, or AWS SageMaker Model Registry. Loading via `from_pretrained("username/repo")` is the correct production handoff.

### Full code block

```python
print(f"Loading tokenizer and model from: {MODEL_REPO}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.to(device)
model.eval()

id2label   = model.config.id2label
label2id   = model.config.label2id
num_labels = model.config.num_labels

print(f"Number of labels: {num_labels}")
print(f"id → label mapping: {id2label}")
print()
print("Model architecture (classification head):")
print(model.classifier)
print()
print(f"Total parameters: {model.num_parameters():,}")
```

---

## Segment 2 — Load and Inspect the Target Corpus

### What this segment does
Downloads the 20 Newsgroups dataset and builds a clean pandas DataFrame capped at `POSTS_PER_CATEGORY` posts per category.
It also filters out very short posts (less than 20 characters) and reports basic statistics about post lengths.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `fetch_20newsgroups(subset='all', ...)` | Downloads all posts (train + test splits combined) for the selected categories. `subset='all'` combines the two splits into one pool so we can sample freely. |
| `remove=('headers', 'footers', 'quotes')` | Strips the email metadata (From, Subject, Date lines), signature blocks, and quoted reply text from each post. Without this, the model would often predict on author names or email addresses instead of the actual message. |
| `random_state=42` | Makes the download reproducible — everyone gets the same posts in the same order. |
| `newsgroups.data` | A Python list of strings — one string per post, containing the cleaned message body. |
| `newsgroups.target` | A NumPy integer array — one integer per post, indicating which category that post belongs to. |
| `newsgroups.target_names` | A list of category name strings, e.g. `["comp.graphics", "rec.sport.hockey", ...]`. We use it to convert the integer category IDs back to readable names. |
| `df_all['text'].str.strip().str.len() >= 20` | Filters out posts whose cleaned text is fewer than 20 characters long. These are usually empty or just whitespace after the metadata is removed, and they carry no useful signal for sentiment analysis. |
| `groupby('category').apply(lambda g: g.sample(...))` | For each category group independently, sample up to `POSTS_PER_CATEGORY` rows. This is the pandas pattern for capping per-group counts. `min(len(g), POSTS_PER_CATEGORY)` handles categories that have fewer posts than the cap. |
| `df['char_len'] = df['text'].str.len()` | Adds a column with each post's character count. We use this later to compare post lengths to model confidence. |

### Full code block

```python
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=CATEGORIES,
    remove=('headers', 'footers', 'quotes'),
    random_state=42,
)

df_all = pd.DataFrame({
    'text':     newsgroups.data,
    'cat_id':   newsgroups.target,
    'category': [newsgroups.target_names[t] for t in newsgroups.target],
})

df_all = df_all[df_all['text'].str.strip().str.len() >= 20].copy()

df = (
    df_all.groupby('category', group_keys=False)
    .apply(lambda g: g.sample(min(len(g), POSTS_PER_CATEGORY), random_state=42))
    .reset_index(drop=True)
)

print(f"Total posts after capping: {len(df)}")
print(df['category'].value_counts().to_string())

df['char_len'] = df['text'].str.len()
print(df['char_len'].describe().round(0).astype(int).to_string())
```

---

## Segment 3 — Side-by-Side Domain Gap Inspection

### What this segment does
Prints a small selection of app review examples (what the model was trained on) next to sample newsgroup posts
(what we are applying the model to). Reading these side-by-side makes the domain gap tangible before we run any numbers.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `app_review_examples` | A hardcoded list of three short, opinionated sentences that represent the style of consumer app reviews. The model learned to associate this kind of language with sentiment labels. |
| `df[df['category'] == cat]['text'].iloc[0][:400]` | Grabs the first post from the filtered category and trims it to 400 characters for display. The slicing `[:400]` does not change the DataFrame — it only affects what we print. |
| `zip(CATEGORIES, newsgroup_examples)` | Pairs each category name with its sample post so we can print them together in one loop. |
| `text.strip()[:300]` | Removes leading/trailing whitespace, then shows only the first 300 characters — enough to see the writing style without scrolling. |

### What to look for when reading the output
- **Length**: reviews are one or two sentences; forum posts run for paragraphs.
- **Purpose**: reviews express a verdict ("amazing", "useless"); forum posts inform, argue, or ask questions.
- **Vocabulary**: tech posts mention *crash* and *error* in a neutral, problem-solving tone — the same words that signal negativity in an app review.
- **Register**: medical and scientific posts use clinical language that has no counterpart in consumer reviews.

### Full code block

```python
app_review_examples = [
    "This app is absolutely amazing! Best purchase I've made all year.",
    "Keeps crashing every time I open it. Completely unusable after the last update.",
    "It's okay. Does what it says but nothing special. 3 stars.",
]

newsgroup_examples = [
    df[df['category'] == cat]['text'].iloc[0][:400]
    for cat in CATEGORIES
]

print("=" * 60)
print("TRAINING DOMAIN — App Reviews (what the model learned from)")
print("=" * 60)
for i, text in enumerate(app_review_examples, 1):
    print(f"\n  Review {i}: {text}")

print()
print("=" * 60)
print("TARGET DOMAIN — 20 Newsgroups (what we are applying the model to)")
print("=" * 60)
for cat, text in zip(CATEGORIES, newsgroup_examples):
    print(f"\n  [{cat}]")
    print(f"  {text.strip()[:300]}...")
```

---

## Segment 4 — Build the Batch Prediction Pipeline

### What this segment does
Defines the `predict_batch` function that tokenizes and classifies a list of texts,
then calls it repeatedly over the full corpus in chunks of `BATCH_SIZE` posts at a time.
Results are merged back into the main DataFrame as new columns.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `def predict_batch(texts: list[str]) -> list[dict]:` | Defines a function that takes a list of text strings and returns a list of dictionaries — one dictionary per text, containing the predicted class, label name, confidence score, and full probability vector. |
| `tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')` | Tokenizes a whole batch at once. `truncation=True` cuts posts that exceed 128 tokens. `padding=True` pads shorter posts so all sequences in the batch have the same length. `return_tensors='pt'` returns PyTorch tensors instead of Python lists — the model requires tensors. |
| `{k: v.to(device) for k, v in encoded.items()}` | A dictionary comprehension that moves every tensor in the `encoded` dictionary to the same device as the model. If the model is on the GPU, the inputs must also be on the GPU. |
| `with torch.no_grad():` | A context manager that tells PyTorch to skip recording math operations for gradient computation. During inference we will never call `.backward()`, so there is no reason to record gradients. This saves roughly 20–30% of memory and makes inference slightly faster. |
| `model(**encoded)` | Runs the forward pass. `**encoded` unpacks the dictionary as keyword arguments, which is how HuggingFace models expect their inputs. The result has an attribute `.logits` of shape `(batch_size, num_labels)`. |
| `F.softmax(outputs.logits, dim=-1)` | Applies the softmax function along the last dimension. Each row of logits (one per example) is converted to a set of probabilities that sum to 1.0. A logit of `[1.5, -0.3, 0.8]` becomes something like `[0.65, 0.10, 0.25]`. |
| `.cpu().numpy()` | Moves the tensor from GPU back to CPU (if needed) and converts it to a NumPy array for easier manipulation with pandas and matplotlib. |
| `probs.argmax(axis=-1)` | For each example (row), picks the column index with the highest probability. That index is the predicted class ID. |
| `probs.max(axis=-1)` | For each example, returns the highest probability value — the model's confidence score for its top prediction. |
| `id2label[int(pred_ids[i])]` | Converts the integer class ID back to a human-readable label name using the mapping we loaded from `model.config.id2label`. |
| The batching loop | `range(0, len(texts), BATCH_SIZE)` steps through the text list in chunks. `texts[start : start + BATCH_SIZE]` slices the current chunk. We call `results.extend(...)` (not `.append`) because `predict_batch` returns a list — we want to flatten it into the results list. |

### Full code block

```python
def predict_batch(texts: list[str]) -> list[dict]:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    probs      = F.softmax(outputs.logits, dim=-1).cpu().numpy()
    pred_ids   = probs.argmax(axis=-1)
    confidence = probs.max(axis=-1)

    return [
        {
            'pred_id':    int(pred_ids[i]),
            'pred_label': id2label[int(pred_ids[i])],
            'confidence': float(confidence[i]),
            'probs':      probs[i].tolist(),
        }
        for i in range(len(texts))
    ]

texts   = df['text'].tolist()
results = []

for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start : start + BATCH_SIZE]
    results.extend(predict_batch(batch))
    if start % 200 == 0:
        print(f"  Processed {start + len(batch)} / {len(texts)} posts...")

df['pred_label'] = [r['pred_label'] for r in results]
df['pred_id']    = [r['pred_id']    for r in results]
df['confidence'] = [r['confidence'] for r in results]

print(f"\nInference complete. Total predictions: {len(df)}")
print(df[['category', 'pred_label', 'confidence']].head().to_string())
```

---

## Segment 5 — Prediction Distribution

### What this segment does
Counts how many posts were assigned each predicted label and plots a bar chart.
A heavily skewed distribution — where most posts are predicted as one class — indicates that the model
is pattern-matching on surface features rather than reading genuine sentiment.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `df['pred_label'].value_counts()` | Counts how many times each label string appears in the column. Returns a Series sorted by count descending. |
| `[id2label[i] for i in sorted(id2label.keys())]` | Creates a list of label names in canonical order (sorted by class ID: 0, 1, 2...). We use this so the bar chart always displays labels in the same order regardless of which class is most common. |
| `overall_counts.reindex(label_order, fill_value=0)` | Reorders the value counts to match `label_order`. If a label was never predicted, `fill_value=0` inserts a zero count rather than dropping it. |
| The ASCII bar inside the print loop | `'#' * int(pct / 2)` draws a simple text bar whose length is proportional to the percentage. Dividing by 2 keeps it from wrapping at 100%. |
| `ax.bar_label(bars, fmt='%d', padding=3)` | Adds a count number on top of each bar. `fmt='%d'` formats the number as an integer. `padding=3` lifts the label slightly above the bar top. |
| `ax.set_ylim(0, overall_counts.max() * 1.15)` | Adds 15% headroom above the tallest bar so the bar labels are not clipped. |

### What to look for in the output
If the distribution is roughly even across classes, the model is uncertain — it cannot map forum vocabulary onto app-review sentiment.
If one class (usually negative) dominates, the model is pattern-matching on alarm words that are common in tech/political forum posts but carry a different meaning there.

### Full code block

```python
overall_counts = df['pred_label'].value_counts()
label_order    = [id2label[i] for i in sorted(id2label.keys())]
overall_counts = overall_counts.reindex(label_order, fill_value=0)

print("Overall prediction distribution across the full corpus:")
for label, count in overall_counts.items():
    pct = count / len(df) * 100
    bar = '#' * int(pct / 2)
    print(f"  {label:12s}  {count:4d}  ({pct:5.1f}%)  {bar}")

colors = ['#d62728', '#aec7e8', '#2ca02c'][:len(label_order)]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(label_order, overall_counts.values, color=colors, edgecolor='black', linewidth=0.6)
ax.bar_label(bars, fmt='%d', padding=3)
ax.set_title('Predicted Label Distribution — 20 Newsgroups Corpus', fontsize=13)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Count')
ax.set_ylim(0, overall_counts.max() * 1.15)
plt.tight_layout()
plt.show()
```

---

## Segment 6 — Confidence Histogram

### What this segment does
Plots two histograms side by side: one showing the overall distribution of model confidence scores,
and one breaking confidence down by predicted label. Together they reveal whether the model is uncertain
(low confidence spread across all classes) or overconfident (high confidence on out-of-domain text).

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `axes[0].hist(df['confidence'], bins=20, ...)` | Draws a histogram with 20 evenly spaced bins between 0 and 1. Each bar shows how many posts fell into that confidence range. |
| `ax.axvline(df['confidence'].median(), ...)` | Draws a vertical dashed line at the median confidence value. This makes the "typical" confidence visible at a glance. |
| The right subplot loop | Draws one histogram per predicted label, overlaid on the same axes. `alpha=0.6` makes each histogram slightly transparent so they do not completely hide each other. |
| `high_conf_pct = (df['confidence'] >= 0.90).sum() / len(df) * 100` | Calculates what fraction of posts have a confidence of 90% or higher. A high number on forum text is a warning sign — the model is overconfident. |

### Reading the histograms
- **Left-skewed peak near 1.0** — the model is committing strongly to most predictions. On out-of-domain text, this usually means overconfidence, not accuracy.
- **Uniform spread** — the model is uncertain across the board, which is appropriate for unfamiliar text.
- **Bimodal distribution** — some posts match the training distribution well (high confidence); others do not (low confidence). The bimodal case is the most informative for error analysis — the low-confidence posts are the ones where the domain shift is most severe.

### Full code block

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df['confidence'], bins=20, color='steelblue', edgecolor='black', linewidth=0.5)
axes[0].axvline(df['confidence'].median(), color='red', linestyle='--',
                label=f"Median: {df['confidence'].median():.2f}")
axes[0].set_title('Overall Confidence Distribution', fontsize=12)
axes[0].set_xlabel('Max Softmax Probability')
axes[0].set_ylabel('Number of Posts')
axes[0].legend()

for label in label_order:
    subset = df[df['pred_label'] == label]['confidence']
    axes[1].hist(subset, bins=20, alpha=0.6, label=label, edgecolor='black', linewidth=0.3)
axes[1].set_title('Confidence by Predicted Label', fontsize=12)
axes[1].set_xlabel('Max Softmax Probability')
axes[1].set_ylabel('Number of Posts')
axes[1].legend()

plt.suptitle('Model Confidence on Out-of-Domain (Newsgroups) Text', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

print("Confidence statistics:")
print(df.groupby('pred_label')['confidence'].describe().round(3).to_string())

high_conf_pct = (df['confidence'] >= 0.90).sum() / len(df) * 100
print(f"\nFraction of posts with confidence >= 0.90: {high_conf_pct:.1f}%")
```

---

## Segment 7 — Per-Category Prediction Breakdown

### What this segment does
Breaks the prediction distribution down by newsgroup category and displays it as a 100% stacked bar chart.
This is where the transfer gap becomes concrete — different categories show different skews because of their
different vocabulary overlap with app-review training data.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `df.groupby(['category', 'pred_label']).size()` | Counts how many posts fall into each (category, label) pair. The result is a Series with a two-level index. |
| `.unstack(fill_value=0)` | Pivots the inner index (pred_label) into columns, creating a 2D table. `fill_value=0` fills any missing combinations with zero. |
| `.reindex(columns=label_order, fill_value=0)` | Reorders the columns to match the canonical label order so the stacked bar colors are consistent. |
| `pivot.div(pivot.sum(axis=1), axis=0) * 100` | Converts raw counts to percentages within each category row. `pivot.sum(axis=1)` sums across columns (labels) for each row (category). `div(..., axis=0)` divides each row by its sum. |
| `kind='bar', stacked=True` | Makes pandas draw a stacked bar chart instead of a grouped bar chart. Each bar reaches 100% and the segments show the proportion for each predicted label. |
| `bbox_to_anchor=(1.01, 1), loc='upper left'` | Positions the legend just outside the plot area on the right side so it does not overlap the bars. |
| `df.groupby('category')['confidence'].mean()` | Computes the average confidence score within each category. Categories where the model is less confident on average are the ones where the domain shift is most severe. |

### What to discuss with learners
- `comp.graphics` often skews negative because technical vocabulary (*crash*, *error*, *failed*, *broken*) overlaps with complaint language in app reviews.
- `talk.politics.guns` often skews negative because aggressive debate language overlaps with negative review language.
- `sci.med` often shows lower average confidence — clinical language has little overlap with either positive or negative app review vocabulary.
- `rec.sport.hockey` may skew positive because fan enthusiasm language ("great game", "amazing play") overlaps with positive review language.

### Full code block

```python
pivot = (
    df.groupby(['category', 'pred_label'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=label_order, fill_value=0)
)

pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

print("Predicted label distribution per category (%):")
print(pivot_pct.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
pivot_pct.plot(kind='bar', stacked=True, ax=ax,
               color=colors[:len(label_order)], edgecolor='black', linewidth=0.4)
ax.set_title('Predicted Sentiment Distribution by Newsgroup Category', fontsize=13)
ax.set_xlabel('')
ax.set_ylabel('% of Posts in Category')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.legend(title='Predicted Label', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_ylim(0, 115)
plt.tight_layout()
plt.show()

print("\nAverage prediction confidence per category:")
print(df.groupby('category')['confidence'].mean().round(3).sort_values().to_string())
```

---

## Segment 8 — Error Analysis

### What this segment does
This segment has three parts.

**Part A** counts and groups the high-confidence predictions — those where the model committed to its answer with ≥ 90% probability.

**Part B** defines `show_confident_examples`, a helper that retrieves and prints the most confidently predicted posts for any given label. We read these manually to judge whether the prediction makes sense.

**Part C** runs a focused analysis on `comp.graphics` posts predicted as the negative class — the clearest example of **technical false negatives** — and draws a scatter plot of confidence vs. post length.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `df[df['confidence'] >= HIGH_CONF_THRESHOLD]` | Boolean indexing — keeps only the rows where the confidence column exceeds the threshold. |
| `high_conf.groupby(['pred_label', 'category']).size().unstack(fill_value=0)` | Cross-tabulates high-confidence predictions by label and category. This shows *which* categories are generating most of the high-confidence (potentially overconfident) predictions. |
| `def show_confident_examples(pred_label, n)` | A helper function that filters by label, sorts by confidence descending, and prints the top `n` results. Writing it as a function means we can call it once per label cleanly. |
| `.nlargest(n, 'confidence')` | Selects the `n` rows with the highest values in the `confidence` column without sorting the full DataFrame. |
| `NEGATIVE_LABEL = id2label[0]` | Assumes the class with ID 0 is the most negative class. If your Lab 7A model uses a different ordering, adjust this. |
| The scatter plot | `ax.set_xscale('log')` uses a log scale on the x-axis because post lengths span a wide range (20 to 5000+ characters). Log scale makes short and long posts both visible in the same chart. |
| `ax.axhline(0.90, ...)` | Draws a horizontal reference line at the 90% confidence threshold so you can see which points are in the "overconfident" zone. |

### The two failure modes to highlight
1. **Technical false negative:** A `comp.graphics` post asks "Why does my program crash when I call glDrawArrays?" The model predicts negative with 94% confidence because *crash* is a strong negative-sentiment word in app reviews. The author is not unhappy — they are just debugging.
2. **Register mismatch:** A `sci.med` post discusses a clinical trial result in formal academic language. The model predicts positive with 88% confidence because it picked up "significant improvement" — a phrase that appears in positive app reviews too. The meaning is entirely different.

### Full code block

```python
# Part A — count high-confidence predictions
HIGH_CONF_THRESHOLD = 0.90

high_conf = df[df['confidence'] >= HIGH_CONF_THRESHOLD].copy()
print(f"Posts predicted with >= {HIGH_CONF_THRESHOLD:.0%} confidence: {len(high_conf)}")
print(f"  ({len(high_conf)/len(df)*100:.1f}% of corpus)")

print("\nHigh-confidence predictions by label and category:")
print(
    high_conf.groupby(['pred_label', 'category'])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

# Part B — examine the most confident predictions per label
def show_confident_examples(pred_label: str, n: int = 3) -> None:
    subset = (
        df[df['pred_label'] == pred_label]
        .nlargest(n, 'confidence')
        [['category', 'pred_label', 'confidence', 'text']]
    )
    print(f"\n{'='*60}")
    print(f"Most confident '{pred_label.upper()}' predictions")
    print(f"{'='*60}")
    for _, row in subset.iterrows():
        print(f"  Category:   {row['category']}")
        print(f"  Confidence: {row['confidence']:.3f}")
        print(f"  Text (first 250 chars):")
        print(f"    {row['text'].strip()[:250]}")
        print()

for label in label_order:
    show_confident_examples(label, n=3)

# Part C — technical false negatives + confidence vs. length scatter
NEGATIVE_LABEL = id2label[0]

tech_negative = df[
    (df['category'] == 'comp.graphics') &
    (df['pred_label'] == NEGATIVE_LABEL) &
    (df['confidence'] >= 0.80)
].nlargest(5, 'confidence')

print(f"comp.graphics posts predicted '{NEGATIVE_LABEL}' with >= 80% confidence: {len(tech_negative)}")

for _, row in tech_negative.head(3).iterrows():
    print(f"\n  Confidence: {row['confidence']:.3f}")
    print(f"  Text: {row['text'].strip()[:300]}")

fig, ax = plt.subplots(figsize=(9, 4))
for label in label_order:
    subset = df[df['pred_label'] == label]
    ax.scatter(subset['char_len'], subset['confidence'],
               alpha=0.3, s=12, label=label)
ax.set_xscale('log')
ax.set_xlabel('Post Length (characters, log scale)')
ax.set_ylabel('Model Confidence')
ax.set_title('Confidence vs. Post Length — by Predicted Label', fontsize=12)
ax.legend(title='Predicted Label')
ax.axhline(0.90, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
plt.tight_layout()
plt.show()
```

---

## Segment 9 — Transfer Gap Summary and Learner Reflection

### What this segment does
Prints a structured written summary of the transfer gap findings — what the numbers showed, what caused it,
and what would be needed to close it. The final markdown cell is left blank for learners to fill in with their
own written analysis.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| The printed summary block | A structured text report using plain `print` statements. It collects results from all previous cells and presents them in a coherent narrative. In a production context, this kind of summary would go into a model card or a readiness review document. |
| `df.groupby('category')['confidence'].mean().sort_values()` | Sorts categories from lowest average confidence (most uncertain) to highest (most certain). The order itself tells you which categories are hardest for the model. |
| `(df['confidence'] >= 0.90).sum()` | Counts the number of True values in the boolean Series produced by the comparison. Each True means one post exceeded the 90% threshold. |
| The reflection markdown cell | An open text cell with four structured questions. Learners answer these by reviewing the charts and example predictions from the earlier cells. The questions mirror what a real ML practitioner would write in a transfer learning report. |

### The four reflection questions explained

| Question | What it is really testing |
|---|---|
| Which categories were most/least confident? | Did the learner read the per-category confidence table and connect it to the vocabulary overlap hypothesis? |
| What linguistic patterns caused mispredictions? | Can the learner identify specific examples from the `show_confident_examples` output and articulate why the model was misled? |
| How did MAX_LENGTH=128 affect results? | Does the learner understand that truncation means the model only read the beginning of each post, which may not contain representative content? |
| What would be needed to deploy reliably? | Can the learner propose concrete mitigations: domain adaptation, confidence thresholding, longer max_length, labeled forum data? |

### Full code block

```python
print("=" * 65)
print("TRANSFER GAP SUMMARY")
print("=" * 65)
print()
print("Training domain : Consumer app reviews (Lab 7A)")
print("Target domain   : 20 Newsgroups forum posts")
print(f"Model           : {MODEL_REPO}")
print(f"Corpus size     : {len(df)} posts across {len(CATEGORIES)} categories")
print()

print("--- Predicted label distribution ---")
for label, count in overall_counts.items():
    print(f"  {label:12s}: {count:4d}  ({count/len(df)*100:.1f}%)")
print()

print("--- Average confidence by category ---")
for cat, conf in df.groupby('category')['confidence'].mean().sort_values().items():
    print(f"  {cat:30s}: {conf:.3f}")
print()

n_overconf = (df['confidence'] >= 0.90).sum()
print(f"--- High-confidence (>= 0.90) posts ---")
print(f"  Count  : {n_overconf} ({n_overconf/len(df)*100:.1f}% of corpus)")
print()

print("--- Key observations ---")
print("  1. The model applies app-review patterns to forum vocabulary.")
print("     Words like 'crash', 'error', 'failed' in comp.graphics posts")
print("     trigger negative predictions even when no dissatisfaction is expressed.")
print()
print("  2. Posts were truncated to 128 tokens. Forum posts are much longer.")
print("     The model only read a small window of each post.")
print()
print("  3. Domain shift is real: the model's confidence does not correlate")
print("     with prediction quality on out-of-domain text.")
print()
print("--- What would help ---")
print("  - Fine-tune on domain-labeled forum posts (adapt, don't just apply)")
print("  - Use a longer max_length or a model trained on longer sequences")
print("  - Add a threshold: refuse to predict if confidence < 0.70")
print("  - Build a calibration curve to understand what 'confidence' really means")
```

---

## Quick Reference — Key Concepts Introduced in This Lab

| Concept | One-sentence definition |
|---|---|
| **Model registry** | A versioned, addressable store for model artifacts — separate from git, accessed by name and version. |
| **`id2label` pattern** | Reading predicted class names from `model.config.id2label` instead of hardcoding them in application code. |
| **Domain transfer** | Applying a model trained on one type of text to a different type of text; results depend on how much vocabulary and style overlap the two domains share. |
| **Transfer gap** | The drop in prediction quality that occurs when training and target domains differ; measured qualitatively (error analysis) when labeled target data is unavailable. |
| **Overconfidence** | When a model assigns high softmax probabilities on out-of-domain text despite making incorrect predictions; high confidence alone does not indicate accuracy. |
| **Technical false negative** | A prediction error caused by vocabulary overlap between a neutral technical term (e.g., *crash* in a forum) and a sentiment-loaded term (e.g., *crash* in a complaint review). |
| **`torch.no_grad()`** | A context manager that disables gradient tracking during inference, saving memory and time. |
| **`F.softmax`** | Converts raw logit scores into probabilities that sum to 1.0, making confidence values interpretable. |

---

*End of walkthrough. Refer to `SI_Demo_Integration_7A_Notebook.ipynb` for the runnable notebook version.*
