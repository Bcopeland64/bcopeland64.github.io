# SI Demo Walkthrough — Sentiment Analysis with DistilBERT on SST-2

**Audience:** Student Instructors preparing to deliver the 90-minute live demo  
**Dataset:** SST-2 (Stanford Sentiment Treebank, binary: Negative / Positive)  
**Model:** `distilbert-base-uncased`

---

> **How to use this document**  
> Each segment below explains *what every line is doing* in plain language, then closes with the **full code block** for that segment so you can copy-paste into your notebook cleanly. Run segments in order — later ones depend on earlier ones.

---

## Segment 0 — Environment Setup

### What this segment does
Installs the required libraries and loads all the Python modules we will need throughout the demo. Running this first means every tool is ready before we need it.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `!pip install transformers ...` | Downloads and installs four libraries from the internet: **transformers** (HuggingFace's model toolkit), **datasets** (HuggingFace's data loader), **scikit-learn** (classical ML metrics), **seaborn / matplotlib** (charts), **torch** (the math engine that runs the neural network), and **accelerate** (makes the Trainer work smoothly on CPUs and GPUs). The `#` comment keeps this line dormant if packages are already installed. |
| `import numpy as np` | Loads NumPy, a library for fast math on arrays. We give it the short nickname `np` so we can write `np.argmax` instead of `numpy.argmax`. |
| `import matplotlib.pyplot as plt` | Loads Matplotlib's plotting module. `plt` is the nickname. We use it to draw the confusion matrix. |
| `import seaborn as sns` | Loads Seaborn, a charting library built on top of Matplotlib. It makes `heatmap()` pretty with one line of code. |
| `from collections import Counter` | Imports a counting tool from Python's built-in library. `Counter([0,1,1,0,1])` returns `{1: 3, 0: 2}`. We use it to count labels. |
| `import torch` | Loads PyTorch, the numerical computing engine. The Trainer uses this under the hood for gradient descent. |
| `from datasets import load_dataset` | Pulls in the function that downloads and caches HuggingFace datasets. |
| `from transformers import AutoTokenizer, ...` | Pulls in four HuggingFace classes: the tokenizer, the model, the config object for training, and the training loop runner. |
| `from sklearn.metrics import ...` | Imports four measurement tools: raw accuracy, F1 score, confusion matrix builder, and a formatted text report. |
| `MODEL_CHECKPOINT = 'distilbert-base-uncased'` | Stores the model name as a constant. Writing it once at the top means if we want to swap to a different model (e.g., `'bert-base-uncased'`) we only change one line. |
| `print('All imports successful!')` | A sanity check. If something is missing, Python will throw an error *above* this line and this message will never print. |

### Full code block

```python
# Uncomment the line below if packages are not yet installed
# !pip install transformers datasets scikit-learn seaborn matplotlib torch accelerate -q

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

MODEL_CHECKPOINT = 'distilbert-base-uncased'
print('All imports successful!')
```

---

## Segment 1 — Data Loading and Inspection

### What this segment does
Downloads the SST-2 dataset, peeks at a few examples so we can see its structure, and counts how many Negative vs. Positive reviews are in the training set. We want to *know our data* before doing anything else.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `dataset = load_dataset('glue', 'sst2')` | Downloads the SST-2 slice of the GLUE benchmark. Think of `load_dataset` as a smart `open()` that fetches from the internet, unzips, and caches locally. After the first run, subsequent calls load from disk. The result is a dictionary-like object with keys `'train'`, `'validation'`, and `'test'`. |
| `print(dataset)` | Prints a summary table showing the three splits and their sizes (67k train, 872 validation, 1.8k test). |
| `for i in range(3):` | Loops over the first three examples in the training set so we can see what a raw example looks like. |
| `ex = dataset['train'][i]` | Fetches one example by integer index. Each example is a Python dictionary with keys `'sentence'`, `'label'`, and `'idx'`. |
| `'Positive' if ex['label'] == 1 else 'Negative'` | The labels are stored as integers (0 = Negative, 1 = Positive). This ternary expression converts them back to words for readability. |
| `train_labels = dataset['train']['label']` | Fetches the entire `'label'` column as a list of integers. This is a column-slice, not a row-slice. |
| `Counter(train_labels)` | Counts occurrences of each label value. Output looks like `{1: 37569, 0: 29780}`. |
| `len(dataset['train'])` | Returns how many rows are in the training split (67,349 for full SST-2). |
| `CAP` and `.select(range(CAP))` | We optionally cap the dataset to ~2 K examples for faster demo runs. `.select()` returns a new dataset containing only the specified integer indices. |

### Full code block

```python
# --- Section 1: Data Loading and Inspection ---
CAP = 2000  # Reduce to ~2K for faster demo; set to None to use full dataset

dataset = load_dataset('glue', 'sst2')
print(dataset)

# Peek at the first three training examples
print('\n--- First 3 Training Examples ---')
for i in range(3):
    ex = dataset['train'][i]
    label_word = 'Positive' if ex['label'] == 1 else 'Negative'
    print(f"  [{label_word}] {ex['sentence']}\n")

# Label distribution
train_labels = dataset['train']['label']
label_counts = Counter(train_labels)
print('Label distribution (train):')
for label_id, count in sorted(label_counts.items()):
    label_word = 'Positive' if label_id == 1 else 'Negative'
    print(f'  {label_id} ({label_word}): {count} examples')

# Optional cap — keeps the demo fast
if CAP:
    dataset['train'] = dataset['train'].select(range(CAP))
    print(f'\nDataset capped at {CAP} training examples for demo speed.')

print(f'\nFinal split sizes:')
for split in dataset:
    print(f'  {split}: {len(dataset[split])} examples')
```

---

## Segment 2 — Tokenization

### What this segment does
Converts raw text strings into numbers that the model can read. Transformers do not understand words — they understand sequences of integers called *token IDs*. This segment shows a single-sentence demo first, then applies the transformation to the entire dataset efficiently using `map`.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)` | Downloads the same vocabulary and splitting rules that were used when DistilBERT was originally trained. "Auto" means HuggingFace will figure out the right tokenizer class automatically. Using the *same* tokenizer is critical — if you change the vocabulary, the pretrained weights no longer make sense. |
| `tokenizer(sample)` | Converts a single sentence into a dictionary with keys `input_ids` (the integer sequence), `attention_mask` (1 = real token, 0 = padding), and sometimes `token_type_ids`. |
| `tokenizer.decode(encoded['input_ids'])` | Reverses the tokenization. You can see the special `[CLS]` and `[SEP]` tokens that DistilBERT adds automatically. |
| `def tokenize_fn(examples):` | Defines a function that accepts a *batch* of examples (a dictionary where values are lists). When `batched=True`, `map()` sends many examples at once for speed. |
| `truncation=True` | If a sentence is longer than 128 tokens, cut it at 128. Without this, very long sentences would crash the model. |
| `padding='max_length'` | If a sentence is shorter than 128 tokens, pad it with zeros up to 128. Every example in a batch must be the same length. |
| `max_length=128` | The target length for all sequences. 128 is a common choice for short texts — long enough for most movie reviews, short enough to train quickly. |
| `dataset.map(tokenize_fn, batched=True)` | Applies `tokenize_fn` to every row in every split of the dataset. `batched=True` sends rows in chunks (default 1000 at a time) which is much faster than one-by-one. The result is a new dataset with extra columns. |
| `rename_column('label', 'labels')` | The HuggingFace `Trainer` looks for a column called **`labels`** (plural). SST-2 ships with a column called `label` (singular). This renames it. |
| `remove_columns(['sentence', 'idx'])` | Deletes the raw text and index columns. The model only needs `input_ids`, `attention_mask`, and `labels`. Leaving extra columns causes the Trainer to complain. |
| `set_format('torch')` | Tells the dataset to return PyTorch tensors instead of Python lists when items are accessed. The model expects tensors, not lists. |

### Full code block

```python
# --- Section 2: Tokenization ---

# Load the tokenizer that matches our model
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Demo: tokenize a single sentence and inspect the output
sample_sentence = dataset['train'][0]['sentence']
print('Original sentence:')
print(' ', sample_sentence)

encoded = tokenizer(sample_sentence)
print('\nToken IDs:', encoded['input_ids'])
print('Attention mask:', encoded['attention_mask'])
print('Decoded back:', tokenizer.decode(encoded['input_ids']))
print(f'Token count: {len(encoded["input_ids"])}')

# Define the batch tokenization function
def tokenize_fn(examples):
    return tokenizer(
        examples['sentence'],
        truncation=True,
        padding='max_length',
        max_length=128,
    )

# Show columns before
print('\nColumns BEFORE tokenization:', dataset['train'].column_names)

# Apply tokenization to every split
tokenized_dataset = dataset.map(tokenize_fn, batched=True)
print('Columns AFTER tokenization: ', tokenized_dataset['train'].column_names)

# Prepare for the Trainer: rename label → labels, drop unused columns, set tensor format
tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
tokenized_dataset = tokenized_dataset.remove_columns(['sentence', 'idx'])
tokenized_dataset.set_format('torch')

print('\nFinal columns ready for Trainer:', tokenized_dataset['train'].column_names)
print('Sample item keys:', list(tokenized_dataset['train'][0].keys()))
```

---

## Segment 3 — Model Construction

### What this segment does
Loads the DistilBERT model with a *new, randomly initialized classification head* on top. The body of the model already knows language from pretraining on Wikipedia and BooksCorpus. We only need to teach the head to predict Negative vs. Positive — that is the fine-tuning task.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `AutoModelForSequenceClassification` | A wrapper class that takes any HuggingFace text encoder and attaches a classification layer on top. "Auto" selects the right architecture based on the checkpoint name. |
| `.from_pretrained(MODEL_CHECKPOINT, num_labels=2)` | Downloads 66M pretrained weights from HuggingFace Hub, then adds a fresh two-neuron output layer. The `num_labels=2` tells it there are two possible classes. **This is the number you change to 3 in the learner assignment.** |
| The warning about "randomly initialized weights" | Expected behavior. It says: "I loaded the body weights from the checkpoint, but the classification head (`pre_classifier` and `classifier`) were initialized randomly because they did not exist in the pretrained model." This is correct — we will train those layers during fine-tuning. |
| `model.classifier` | Inspects the classification head — a single linear layer that takes a 768-dimensional vector and outputs 2 scores (one per class). |
| `model.num_parameters()` | Returns ~67 million. Of those, most are frozen in early fine-tuning and only the head changes significantly. |

### Full code block

```python
# --- Section 3: Model Construction ---

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2,           # Binary: Negative=0, Positive=1
                            # ← LEARNER LAB: change this to 3 for your dataset
)

# Inspect the classification head that was just randomly initialized
print('Classification head architecture:')
print(model.classifier)
print()

# Count total parameters
total_params = model.num_parameters()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters:     {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
```

---

## Segment 4 — TrainingArguments

### What this segment does
Creates a configuration object that controls *every aspect* of the training loop: how long to train, how big each batch is, how fast to learn, when to evaluate, and where to save results. Nothing actually trains yet — this is just the plan.

### Concept explanations

| Argument | What it means in plain English |
|----------|-------------------------------|
| `output_dir='./sst2_results'` | The folder where model checkpoints are saved at the end of each epoch. If training crashes, you can resume from the latest checkpoint. |
| `num_train_epochs=3` | Go through the entire training set 3 times. One full pass is called an *epoch*. More epochs → more learning, but also more risk of memorizing the training data. |
| `per_device_train_batch_size=16` | Feed 16 examples to the model at a time during training. Larger batches are more stable but need more GPU memory. |
| `per_device_eval_batch_size=64` | Feed 64 examples at a time during evaluation. We can use a bigger number here because evaluation does *not* compute gradients — it uses less memory. |
| `warmup_steps=100` | For the first 100 optimization steps, gradually increase the learning rate from near-zero to the full `learning_rate`. Jumping straight to a high learning rate can destabilize the pretrained weights. |
| `weight_decay=0.01` | A regularization penalty that pushes small weights toward zero. Helps prevent the model from overfitting (memorizing training data). |
| `learning_rate=2e-5` | Controls how large each weight update is. `2e-5` means 0.00002 — intentionally tiny so we do not destroy the pretrained knowledge. Standard range for fine-tuning transformers: 1e-5 to 5e-5. |
| `logging_dir='./sst2_logs'` | Where TensorBoard log files go. Open TensorBoard with `tensorboard --logdir ./sst2_logs` to watch loss curves live. |
| `logging_steps=50` | Print a training log line every 50 gradient update steps. |
| `eval_strategy='epoch'` | Run evaluation on the validation set at the end of every epoch. Options: `'no'`, `'steps'`, `'epoch'`. |
| `save_strategy='epoch'` | Save a checkpoint to disk at the end of every epoch. Must match `eval_strategy` when using `load_best_model_at_end=True`. |
| `load_best_model_at_end=True` | After all epochs finish, automatically reload the checkpoint that had the best validation metric. Prevents the final checkpoint from being worse than an earlier one. |
| `metric_for_best_model='f1'` | "Best" is defined by F1 score, not by loss. Since we pass `compute_metrics`, the Trainer knows how to compute F1. |
| `report_to='none'` | Suppresses automatic logging to Weights & Biases, MLflow, etc. Good for demos so you are not prompted to log in. |

### Full code block

```python
# --- Section 4: TrainingArguments Walkthrough ---

training_args = TrainingArguments(
    output_dir='./sst2_results',        # Checkpoint save location
    num_train_epochs=3,                  # Epochs to train
    per_device_train_batch_size=16,      # Train batch size per device
    per_device_eval_batch_size=64,       # Eval batch size (larger is fine, no gradients)
    warmup_steps=100,                    # Linear LR warmup to protect pretrained weights
    weight_decay=0.01,                   # L2 regularization
    learning_rate=2e-5,                  # Fine-tuning LR — keep small
    logging_dir='./sst2_logs',           # TensorBoard log directory
    logging_steps=50,                    # Log every N gradient steps
    eval_strategy='epoch',               # Evaluate at end of each epoch
    save_strategy='epoch',               # Save checkpoint each epoch
    load_best_model_at_end=True,         # Reload best checkpoint after training
    metric_for_best_model='f1',          # Use F1 to pick the best checkpoint
    report_to='none',                    # Silence W&B / MLflow prompts
)

print('TrainingArguments created successfully.')
print(f'  Epochs:          {training_args.num_train_epochs}')
print(f'  Train batch:     {training_args.per_device_train_batch_size}')
print(f'  Learning rate:   {training_args.learning_rate}')
print(f'  Output dir:      {training_args.output_dir}')
```

---

## Segment 5 — compute_metrics

### What this segment does
Defines the function that the Trainer will call automatically at the end of every evaluation run. The Trainer passes raw model outputs (logits) and correct answers (labels); this function converts them into human-readable scores.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `def compute_metrics(eval_pred):` | Defines a function with one parameter. The Trainer will call it and pass a named tuple containing two arrays: `predictions` (raw scores) and `label_ids` (ground-truth integers). |
| `logits, labels = eval_pred` | Unpacks the tuple. `logits` is a 2D array of shape `(num_examples, num_classes)` — each row is one example, each column is the model's raw confidence score for one class. They are *not* probabilities yet. |
| `np.argmax(logits, axis=-1)` | For each row (example), find the column index with the highest value. `axis=-1` means "look along the last dimension." A row like `[2.1, -0.4]` → predicted class `0`; a row like `[-0.3, 1.8]` → predicted class `1`. |
| `accuracy_score(labels, predictions)` | Fraction of examples where the predicted class matched the true class. Simple but misleading for imbalanced datasets. |
| `f1_score(labels, predictions, average='binary')` | The harmonic mean of precision and recall. `average='binary'` is only valid for 2-class problems — it treats class 1 as the "positive" class. **For the learner lab (3 classes), change to `average='weighted'` or `'macro'`.** |
| `return {'accuracy': ..., 'f1': ...}` | Returns a dictionary. The Trainer will log these keys and compare them using `metric_for_best_model`. |

### Argmax intuition demo

```
Sample logits (3 examples, 2 classes):
  [[ 2.3, -1.1],   → predict class 0 (Negative)
   [-0.5,  3.2],   → predict class 1 (Positive)
   [ 0.1,  0.2]]   → predict class 1 (Positive, barely)

np.argmax(..., axis=-1) → [0, 1, 1]
```

### Full code block

```python
# --- Section 5: compute_metrics for Binary Classification ---

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)   # Convert raw scores → class indices
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    # ← LEARNER LAB: change average='binary' to average='weighted' for 3-class
    return {'accuracy': accuracy, 'f1': f1}

# Visual intuition: watch argmax in action
sample_logits = np.array([
    [ 2.3, -1.1],   # Model is very confident this is class 0
    [-0.5,  3.2],   # Model is very confident this is class 1
    [ 0.1,  0.2],   # Model is barely leaning toward class 1
])
print('Sample logits:\n', sample_logits)
print('Predicted classes:', np.argmax(sample_logits, axis=-1))
print('Expected output: [0, 1, 1]')
```

---

## Segment 6 — Smoke Test Training

### What this segment does
Runs a tiny end-to-end training loop on just 50 training examples and 20 validation examples for 1 epoch. The goal is to verify the pipeline works (no crashes) and watch the loss decrease live. This is not enough data to produce a useful model — it is a plumbing check.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `tokenized_dataset['train'].select(range(50))` | Creates a new dataset containing only the first 50 rows. `range(50)` generates `[0, 1, 2, ..., 49]`. We use this tiny slice so training finishes in about 30 seconds on a CPU. |
| `TrainingArguments(... save_strategy='no' ...)` | A separate, minimal config just for the smoke test. `save_strategy='no'` means we skip writing checkpoints — no point saving a model trained on 50 examples. |
| `Trainer(model=model, args=smoke_args, ...)` | Assembles the training loop. The Trainer takes the model, the hyperparameter config, the data splits, and the metric function — and handles everything else (batching, gradient updates, evaluation, logging). |
| `trainer.train()` | Starts the training loop. Watch the log output: the `loss` column should decrease over steps. Even on 50 examples over 1 epoch, you should see the loss drop from ~0.7 toward ~0.5. |
| Watching the log table | Each row is one logging checkpoint. Columns: `step` (how many gradient updates so far), `training_loss` (lower is better), `epoch` (which pass through the data we are on). |

### Full code block

```python
# --- Section 6: Smoke Test Training (50 examples, 1 epoch) ---

# Carve out tiny subsets
smoke_train = tokenized_dataset['train'].select(range(50))
smoke_val   = tokenized_dataset['validation'].select(range(20))

print(f'Smoke train size: {len(smoke_train)} examples')
print(f'Smoke val size:   {len(smoke_val)} examples')

# Minimal TrainingArguments for the smoke test
smoke_args = TrainingArguments(
    output_dir='./smoke_results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    logging_steps=5,
    eval_strategy='epoch',
    save_strategy='no',
    report_to='none',
)

# Build the Trainer
trainer = Trainer(
    model=model,
    args=smoke_args,
    train_dataset=smoke_train,
    eval_dataset=smoke_val,
    compute_metrics=compute_metrics,
)

# Fire! Watch the loss decrease in the log table
print('\nStarting smoke test...')
trainer.train()
print('\nSmoke test complete.')
```

---

## Segment 7 — Inference and Confusion Matrix

### What this segment does
Uses the trained model to make predictions on unseen validation examples, then builds a confusion matrix to visualize *where* the model makes mistakes. A confusion matrix shows every combination of (true label, predicted label) as a grid of counts.

### Concept explanations

| Line | What it means in plain English |
|------|-------------------------------|
| `tokenized_dataset['validation'].select(range(100))` | Grabs 100 validation examples to run inference on. |
| `trainer.predict(test_subset)` | Runs the model on all 100 examples in batches and collects the output. Returns a `PredictionOutput` object with three attributes: `.predictions` (raw logits), `.label_ids` (true labels), and `.metrics`. |
| `np.argmax(predictions_output.predictions, axis=-1)` | Converts logit arrays into class index predictions, same as in `compute_metrics`. |
| `predictions_output.label_ids` | The ground-truth labels for the 100 examples. |
| `classification_report(true_labels, preds, ...)` | Prints precision, recall, and F1 for each class plus macro and weighted averages. More informative than a single accuracy number. |
| `confusion_matrix(true_labels, preds)` | Returns a 2D NumPy array. Row = true label, column = predicted label. Diagonal = correct predictions. Off-diagonal = mistakes. |
| `sns.heatmap(cm, annot=True, fmt='d', ...)` | Draws the confusion matrix as a colored grid. `annot=True` writes the count inside each cell. `fmt='d'` formats them as integers. `cmap='Blues'` uses a blue color scale. |
| `xticklabels=label_names` | Replaces column indices (0, 1) with human-readable names ('Negative', 'Positive'). |

### How to read the confusion matrix

```
                 Predicted
                 Neg    Pos
True   Neg  [  TN  |  FP  ]
       Pos  [  FN  |  TP  ]

TN = True Negative  (correctly said Negative)
TP = True Positive  (correctly said Positive)
FP = False Positive (said Positive when really Negative)
FN = False Negative (said Negative when really Positive)
```

For the learner lab (3 classes), the matrix becomes 3×3. There are no longer "TP/TN/FP/FN" in the binary sense — instead, look at the diagonal for per-class accuracy and the off-diagonal for which classes get confused with each other.

### Full code block

```python
# --- Section 7: Inference and Confusion Matrix ---

# Run predictions on 100 validation examples
test_subset = tokenized_dataset['validation'].select(range(100))
predictions_output = trainer.predict(test_subset)

preds      = np.argmax(predictions_output.predictions, axis=-1)
true_labels = predictions_output.label_ids
label_names = ['Negative', 'Positive']
# ← LEARNER LAB: update label_names to your 3 class names

# Text report
print('--- Classification Report ---')
print(classification_report(true_labels, preds, target_names=label_names))

# Confusion matrix heatmap
cm = confusion_matrix(true_labels, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=label_names,
    yticklabels=label_names,
    cmap='Blues',
)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix — SST-2 Smoke Test (50 train examples)')
plt.tight_layout()
plt.show()
```

---

## Segment 8 — Pivot to the Learner Lab

### What this segment does
Summarizes exactly what changes when the learner moves from binary SST-2 (2 classes) to their multiclass dataset (3 classes). The rest of the pipeline is identical — only four things need to change.

### The four changes

| Location | Binary (demo) | Multiclass (learner lab) |
|----------|--------------|--------------------------|
| `AutoModelForSequenceClassification` | `num_labels=2` | `num_labels=3` |
| `f1_score` | `average='binary'` | `average='weighted'` or `'macro'` |
| `label_names` | `['Negative', 'Positive']` | Your three class names |
| Confusion matrix | 2×2 grid | 3×3 grid |

### Why `weighted` vs `macro`?

- **`average='weighted'`** — weights each class's F1 by how many examples it has. Good default; fair to imbalanced datasets.
- **`average='macro'`** — averages each class's F1 equally, regardless of class size. Use this if minority classes matter as much as the majority class.

### Full code block

```python
# --- Section 8: Pivot to the Learner Lab ---

print("""
=== YOUR ASSIGNMENT: MULTICLASS CLASSIFICATION ===

Your dataset has 3 classes instead of 2. Make these 4 changes:

  1. Model construction — change num_labels:
       num_labels=3   # was 2

  2. compute_metrics — change f1_score averaging:
       f1 = f1_score(labels, predictions, average='weighted')
       # was average='binary'  (binary only works for 2-class problems)

  3. Update label_names in the confusion matrix cell:
       label_names = ['Class A', 'Class B', 'Class C']   # your actual names

  4. Your confusion matrix will now be a 3x3 grid.
     Look at the diagonal for correct predictions and
     off-diagonal cells to see which classes are confused.

Everything else — data loading, tokenization, model loading,
TrainingArguments, Trainer, predict() — stays exactly the same.
""")

# Template for your 3-class compute_metrics
def compute_metrics_multiclass(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')   # ← key change
    return {'accuracy': accuracy, 'f1': f1}
```

---

## Quick Reference — Binary → Multiclass Diff

```python
# BINARY (this demo)                    # MULTICLASS (learner lab)
num_labels=2                            num_labels=3
average='binary'                        average='weighted'
label_names = ['Neg', 'Pos']           label_names = ['A', 'B', 'C']
```

---

*End of walkthrough. Refer to `SI_Demo_SST2_Lab_Notebook.ipynb` for the runnable notebook version.*
