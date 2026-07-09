# Walkthrough: TriviaQA QA Pipeline Lab (SI Demo)

**Module 7B Applied Lab — Day 2, 90-minute session**

This document walks through every line of the SI demo notebook
(`si_demo_triviaqa.ipynb`), section by section. Each section ends with
the complete code block for that section so you can copy-paste or
cross-reference quickly.

---

## Overview

The demo uses the **TriviaQA `rc.nocontext`** validation split (150 examples).
`rc.nocontext` means there is no passage — only a question and a set of
valid answer strings. We deliberately pass the question itself as the
context for the extractive QA pipeline. This forces the model to rely
entirely on its parametric (trained) knowledge, which surfaces
**knowledge-gap failures** in a clean, controlled way.

The demo covers five skills that appear in the learner assignment:

| Step | Skill |
|---|---|
| 1 | Loading a dataset from HuggingFace |
| 2 | Instantiating and running a `pipeline()` |
| 3 | Writing normalization + EM + token-F1 functions |
| 4 | Looping, scoring, and writing a CSV |
| 5 | Diagnosing failures from the result table |

---

## Section 1 — Setup & Imports

### What we're doing

Before any ML work, we need four things:

- **`datasets`** — HuggingFace library that downloads and streams
  benchmark datasets (TriviaQA, SQuAD, etc.).
- **`transformers`** — HuggingFace library that wraps pre-trained models
  behind a high-level `pipeline()` API.
- **`pandas`** — the workhorse for tabular result inspection and CSV I/O.
- **Standard library** — `string`, `re`, and `Counter` power the
  SQuAD normalization and token-F1 functions.

### Line-by-line

```
%pip install -q datasets transformers torch pandas
```
The `%pip` magic installs into the active kernel's environment.
`-q` (quiet) suppresses verbose install logs.
`torch` is the backend PyTorch engine that `transformers` calls when
running the model.

```python
import string
```
`string.punctuation` is a pre-built constant:
`'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'`.
We use it to build a translation table that strips punctuation from
answer strings.

```python
import re
```
The `re` (regular expressions) module lets us write concise patterns
like `\b(a|an|the)\b` to match and remove articles that SQuAD scoring
ignores.

```python
from collections import Counter
```
`Counter` turns a list of tokens into a frequency dictionary.
The `&` (intersection) operator between two `Counter` objects gives us
the count of tokens shared between the prediction and the gold answer —
the core of the token-F1 calculation.

```python
import pandas as pd
```
`pd` is the conventional alias for pandas.
We use `pd.DataFrame` to collect per-example results and `DataFrame.to_csv`
to write them to disk for later analysis.

```python
from datasets import load_dataset
```
This is the entry point for downloading any HuggingFace Hub dataset.
We pass a dataset name, a configuration name, and a split string.

```python
from transformers import pipeline
```
`pipeline` is a one-line factory: you pass a task name and optional model
identifier, and it handles tokenization, inference, and post-processing.

### Full Code — Section 1

```python
# ── install once if running in a fresh environment ──────────────────────────
%pip install -q datasets transformers torch pandas             # install core libs

# ── standard library ────────────────────────────────────────────────────────
import string                                                  # punctuation chars
import re                                                      # regex engine
from collections import Counter                                # token frequency

# ── third-party ─────────────────────────────────────────────────────────────
import pandas as pd                                            # tabular data ops
from datasets import load_dataset                              # HuggingFace datasets
from transformers import pipeline                              # HF inference pipeline
```

---

## Section 2 — QA Pipeline Construction + One-Passage Inference

### What we're doing

We build the pipeline, load the first 150 examples of TriviaQA
`rc.nocontext`, inspect one example, and run end-to-end inference.

**Why `deepset/roberta-base-squad2`?**
This checkpoint is a RoBERTa-base model fine-tuned on SQuAD 2.0. It is
widely used as a baseline for extractive QA because it is small enough to
run on a CPU in reasonable time and well-understood in the literature.

**Why pass the question as context?**
The `rc.nocontext` split has no passage. The `pipeline()` API requires a
`context` string. By passing the question itself, we observe how the model
behaves when no answer is actually present in the text — a pure
knowledge-gap scenario.

### Line-by-line

```python
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
)
```
`pipeline("question-answering", ...)` downloads the model weights and
tokenizer on first run (cached to `~/.cache/huggingface` afterward).
The return value, `qa_pipeline`, is a callable: you pass it a `question`
and `context` and it returns a dict.

```python
dataset = load_dataset(
    "trivia_qa",
    "rc.nocontext",
    split="validation[:150]",
    trust_remote_code=True
)
```
- `"trivia_qa"` is the dataset identifier on HuggingFace Hub.
- `"rc.nocontext"` selects the reading-comprehension subset without
  passage context (as opposed to `"rc"` which includes Wikipedia paragraphs).
- `"validation[:150]"` is a slice string — only the first 150 examples of
  the validation split are downloaded and loaded into memory.
- `trust_remote_code=True` is required because TriviaQA uses a custom
  loading script.

```python
example = dataset[0]
```
`dataset` behaves like a list; index 0 returns the first record as a
Python dictionary with keys `question`, `answer`, `question_id`, etc.

```python
print("Question :", example["question"])
print("Gold     :", example["answer"]["value"])
```
`example["answer"]` is itself a dict. `"value"` is the canonical gold
answer string. `"aliases"` (used later) holds all acceptable alternative
spellings and forms.

```python
result = qa_pipeline(
    question=example["question"],
    context=example["question"],
)
print(result)
```
`result` is a dict with four keys:
- `"answer"` — the extracted span (a substring of `context`).
- `"score"` — the model's softmax confidence for this span.
- `"start"` — character offset of the span start in `context`.
- `"end"` — character offset of the span end in `context`.

Because the context is the question itself, the extracted span will be
some fragment of the question text — almost never the correct trivia
answer. This is the knowledge-gap failure in action.

### Full Code — Section 2

```python
# ── 2a. Build the pipeline ───────────────────────────────────────────────────
qa_pipeline = pipeline(                                        # create HF pipeline
    "question-answering",                                      # extractive QA task
    model="deepset/roberta-base-squad2",                       # SQuAD2-tuned RoBERTa
)

# ── 2b. Load the TriviaQA subset ────────────────────────────────────────────
dataset = load_dataset(                                        # fetch from HF Hub
    "trivia_qa",                                               # dataset identifier
    "rc.nocontext",                                            # no-passage config
    split="validation[:150]",                                  # first 150 examples
    trust_remote_code=True,                                    # allow dataset script
)

# ── 2c. Inspect one example ──────────────────────────────────────────────────
example = dataset[0]                                           # first record
print("Question :", example["question"])                       # show the question
print("Gold     :", example["answer"]["value"])                # canonical answer

# ── 2d. Run one-passage inference ────────────────────────────────────────────
result = qa_pipeline(                                          # run the pipeline
    question=example["question"],                              # trivia question
    context=example["question"],                               # question = context
)
print("\nResult dict:")                                        # section label
print(result)                                                  # answer, score, start, end
```

---

## Section 3 — Evaluation Harness: Normalization, EM, and Token-F1

### What we're doing

Every QA benchmark since SQuAD 2016 uses the same three-step harness:

1. **Normalize** both prediction and gold answer (lowercase, strip
   punctuation, strip articles, collapse whitespace).
2. **Exact Match (EM)**: 1 if the normalized strings are identical, else 0.
3. **Token-F1**: precision and recall over shared tokens, then harmonic mean.

We implement each as a standalone function so it can be unit-tested and
reused in the learner assignment.

### Line-by-line: `normalize_answer`

```python
def normalize_answer(text):
```
A pure function: takes any string, returns a cleaned string. No side
effects. Both EM and token-F1 call this before any comparison.

```python
    text = text.lower()
```
Step 1: Lowercase. "United States" and "united states" should compare
as equal. This is the most impactful single normalization step.

```python
    text = text.translate(str.maketrans("", "", string.punctuation))
```
Step 2: Strip punctuation. `str.maketrans("", "", chars)` builds a
translation table that maps every character in `chars` to `None` (delete).
`text.translate(table)` applies it in one pass. This handles periods,
commas, apostrophes, and hyphens simultaneously.

```python
    text = re.sub(r"\b(a|an|the)\b", " ", text)
```
Step 3: Remove articles. `\b` is a word-boundary anchor so we only match
standalone words — "an" inside "hand" is not removed. The replacement is
a space (not empty string) to avoid fusing adjacent words.

```python
    text = " ".join(text.split())
```
Step 4: Collapse whitespace. `str.split()` (no argument) splits on any
run of whitespace and discards empty strings. Re-joining with a single
space normalizes tabs, multiple spaces, and leading/trailing whitespace.

```python
    return text
```
Returns the four-step-cleaned string.

### Line-by-line: `exact_match`

```python
def exact_match(prediction, gold_answers):
```
`gold_answers` is a list because TriviaQA provides many valid aliases.
We return 1 if the prediction matches *any* of them.

```python
    pred_norm = normalize_answer(prediction)
    for gold in gold_answers:
        if pred_norm == normalize_answer(gold):
            return 1
    return 0
```
Short-circuit: returns 1 on first match. If no alias matches, returns 0.
Both sides are normalized before comparison so casing and punctuation
differences do not penalize semantically correct answers.

### Line-by-line: `token_f1`

```python
def token_f1(prediction, gold_answers):
```
Measures partial credit. "United States" vs. "United States of America"
gets F1 > 0 even though EM = 0.

```python
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
```
Guard against empty predictions. An empty string would cause
`len(pred_tokens) == 0` and a ZeroDivisionError in the precision formula.

```python
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
```
`Counter(tokens)` maps each token to its count. The `&` (min) intersection
gives us, for each shared token, the minimum count between prediction and
gold. `sum(common.values())` is the total number of shared tokens (with
multiplicity for repeated tokens).

```python
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall    = n_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
```
- **Precision**: what fraction of predicted tokens were correct?
- **Recall**: what fraction of gold tokens were predicted?
- **F1**: harmonic mean, penalizes extremes in precision/recall.
- We iterate all aliases and keep the best F1 so we don't penalize
  predictions that match a non-canonical but valid form.

### Full Code — Section 3

```python
def normalize_answer(text):                                    # SQuAD-style normalizer
    text = text.lower()                                        # step 1: lowercase
    text = text.translate(                                     # step 2: strip punctuation
        str.maketrans("", "", string.punctuation)              #   using translate table
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)               # step 3: drop articles
    text = " ".join(text.split())                              # step 4: collapse whitespace
    return text                                                # return cleaned string


def exact_match(prediction, gold_answers):                     # binary correctness metric
    pred_norm = normalize_answer(prediction)                   # normalize prediction
    for gold in gold_answers:                                  # try every valid alias
        if pred_norm == normalize_answer(gold):                # compare after normalizing
            return 1                                           # found a match → EM = 1
    return 0                                                   # no match → EM = 0


def token_f1(prediction, gold_answers):                        # token-overlap F1 metric
    pred_tokens = normalize_answer(prediction).split()         # tokenize prediction
    if not pred_tokens:                                        # guard: empty prediction
        return 0.0                                             # return 0 immediately
    best_f1 = 0.0                                              # accumulator for best F1
    for gold in gold_answers:                                  # try every valid alias
        gold_tokens = normalize_answer(gold).split()           # tokenize gold answer
        if not gold_tokens:                                    # guard: empty gold
            continue                                           # skip this alias
        common = Counter(pred_tokens) & Counter(gold_tokens)   # shared token counts
        n_common = sum(common.values())                        # total shared tokens
        if n_common == 0:                                      # skip if no overlap
            continue
        precision = n_common / len(pred_tokens)                # shared / predicted
        recall    = n_common / len(gold_tokens)                # shared / gold
        f1 = 2 * precision * recall / (precision + recall)     # harmonic mean
        best_f1 = max(best_f1, f1)                             # keep best alias score
    return best_f1                                             # return best F1


# ── Worked example ──────────────────────────────────────────────────────────
pred_demo = "The United States of America"                     # raw prediction string
gold_demo = "USA"                                              # canonical gold answer
print("norm pred :", normalize_answer(pred_demo))              # see normalized form
print("norm gold :", normalize_answer(gold_demo))              # see normalized form
print("EM  :", exact_match(pred_demo, [gold_demo]))            # EM = 0 (different)
print("F1  :", round(token_f1(pred_demo, [gold_demo]), 3))     # F1 = 0 (no overlap)
print("F1 with alias 'United States' :",
      round(token_f1(pred_demo, [gold_demo, "United States"]), 3))  # F1 > 0
```

---

## Section 4 — Run Pipeline + Harness over the Full Subset

### What we're doing

We loop over all 150 examples, run the pipeline on each, score the
prediction with EM and token-F1, and collect everything into a DataFrame
that gets written to `triviaqa_results.csv`.

### Line-by-line

```python
records = []
```
An empty list that will grow to 150 dicts, one per example.
Appending to a list then calling `pd.DataFrame(records)` at the end is
more efficient than growing a DataFrame row-by-row.

```python
for ex in dataset:
```
`dataset` is iterable; each iteration yields one record dict.

```python
    question     = ex["question"]
    gold_value   = ex["answer"]["value"]
    gold_aliases = list(ex["answer"]["aliases"])
    if gold_value not in gold_aliases:
        gold_aliases.append(gold_value)
```
We build the full alias list by combining `"aliases"` (all valid surface
forms) with `"value"` (the canonical form). `list(...)` copies the alias
list so we don't mutate the dataset in place. The guard prevents
duplicating the canonical value if it's already in the alias list.

```python
    try:
        out   = qa_pipeline(question=question, context=question)
        pred  = out["answer"]
        score = out["score"]
    except Exception:
        pred  = ""
        score = 0.0
```
Wrapping the pipeline call in `try/except` makes the loop robust: if any
example causes a tokenizer error (e.g., an unusually long question), we
record an empty prediction and continue rather than crashing mid-loop.

```python
    em = exact_match(pred, gold_aliases)
    f1 = token_f1(pred, gold_aliases)
```
Score the prediction against all valid aliases.

```python
    records.append({
        "question": question,
        "gold":     gold_value,
        "prediction": pred,
        "score":    round(score, 4),
        "em":       em,
        "f1":       round(f1, 4),
    })
```
Store all fields we'll need for analysis: the question (for reading
qualitative examples), both gold and prediction (for manual inspection),
the model's confidence (`score`), and both metrics.

```python
df = pd.DataFrame(records)
df.to_csv("triviaqa_results.csv", index=False)
```
`pd.DataFrame(records)` converts the list of dicts into a table with
column names inferred from the dict keys. `to_csv(..., index=False)`
writes without a row-number column, which makes the file cleaner to
open in Excel or share with teammates.

### Full Code — Section 4

```python
records = []                                                   # list of result dicts

for ex in dataset:                                             # iterate all examples
    question     = ex["question"]                              # question text
    gold_value   = ex["answer"]["value"]                       # canonical answer
    gold_aliases = list(ex["answer"]["aliases"])               # all valid aliases
    if gold_value not in gold_aliases:                         # deduplicate
        gold_aliases.append(gold_value)                        # ensure value included

    try:
        out   = qa_pipeline(question=question, context=question)  # run inference
        pred  = out["answer"]                                  # predicted span
        score = out["score"]                                   # model confidence
    except Exception:                                          # catch pipeline errors
        pred  = ""                                             # fallback: empty string
        score = 0.0                                            # fallback: zero score

    em = exact_match(pred, gold_aliases)                       # compute EM (0 or 1)
    f1 = token_f1(pred, gold_aliases)                          # compute token-F1

    records.append({                                           # save all fields
        "question":   question,
        "gold":       gold_value,                              # canonical answer
        "prediction": pred,                                    # model output
        "score":      round(score, 4),                         # confidence (4 dp)
        "em":         em,                                      # exact match flag
        "f1":         round(f1, 4),                            # F1 score
    })

df = pd.DataFrame(records)                                     # build DataFrame
df.to_csv("triviaqa_results.csv", index=False)                 # persist to disk
print(f"Done — {len(df)} rows saved to triviaqa_results.csv")  # confirm save
df.head(3)                                                     # preview first rows
```

---

## Section 5 — Inspect Aggregate Metrics

### What we're doing

We compute the headline EM and F1 numbers, interpret the gap between
them, and sort by F1 ascending to surface the worst predictions.

### Key concept: the EM–F1 gap

| Gap size | What it means |
|---|---|
| Large (> 0.05) | Many near-misses: surface-form or normalization failures dominant |
| Small (≤ 0.05) | Mostly binary right/wrong: knowledge-gap or distractor failures dominant |

In a no-context TriviaQA run, the gap is typically small — predictions
are usually fragments of the question, not near-correct answer strings.

### Line-by-line

```python
mean_em = df["em"].mean()
mean_f1 = df["f1"].mean()
gap     = mean_f1 - mean_em
```
`DataFrame.mean()` computes the column average. EM and F1 are both in
[0, 1], so the gap is a fraction. A gap of 0.10 means there are roughly
10 percentage points' worth of examples where the model partially overlaps
with the gold but doesn't match exactly.

```python
worst10 = df.sort_values("f1").head(10)
display(worst10[["question", "gold", "prediction", "score", "em", "f1"]])
```
`sort_values("f1")` sorts ascending by default (lowest first), so the
first 10 rows are the worst predictions. Column subsetting limits the
display to the columns useful for failure diagnosis.

### Full Code — Section 5

```python
mean_em  = df["em"].mean()                                     # aggregate EM rate
mean_f1  = df["f1"].mean()                                     # aggregate F1 rate
gap      = mean_f1 - mean_em                                   # F1 − EM gap

print(f"Mean EM  : {mean_em:.3f}")                             # print EM
print(f"Mean F1  : {mean_f1:.3f}")                             # print F1
print(f"F1–EM gap: {gap:.3f}")                                 # print gap
print()
print("Interpretation:")
if gap > 0.05:                                                 # threshold for discussion
    print("  Large gap → near-misses: surface-form / normalization failures")
else:
    print("  Small gap → mostly binary right/wrong: knowledge-gap dominant")

print("\n── Worst 10 predictions (sorted by F1 ascending) ──")
worst10 = df.sort_values("f1").head(10)                        # 10 lowest-F1 rows
display(worst10[["question", "gold", "prediction", "score", "em", "f1"]])
```

---

## Section 6 — Three TriviaQA Failure Modes

### What we're doing

For each failure mode we:
1. Name the pattern and explain what is happening mechanically.
2. Filter the DataFrame to find concrete examples.
3. State what a production team would do about it.

These three modes serve as a **template for the learner assignment** —
not to be copied, but to show the depth of analysis expected.

---

### Failure Mode 1: Knowledge-Gap Failures

**Pattern:** The question requires world knowledge the model has not
internalized. With no passage, the model extracts some span from the
question text itself. The extraction is syntactically plausible but
factually wrong.

**Filter logic:**
```python
kg_mask = (df["em"] == 0) & (df["f1"] < 0.3) & (df["score"] > 0.3)
```
We want examples where:
- The prediction is wrong (`em == 0`).
- There is almost no token overlap with the gold (`f1 < 0.3`).
- The model is at least somewhat confident (`score > 0.3`), which makes
  the failure more interesting than a low-confidence "I don't know."

### Full Code — Failure Mode 1

```python
print("=" * 60)
print("FAILURE MODE 1: Knowledge-Gap Failures")
print("=" * 60)
print("""
Pattern : The question requires world knowledge the model does not
          have. With no passage, the model extracts a span from the
          question text, which rarely contains the answer. Confidence
          (score) can be moderate to high; the answer is still wrong.
""")

kg_mask = (df["em"] == 0) & (df["f1"] < 0.3) & (df["score"] > 0.3)  # define filter
kg_examples = df[kg_mask].head(3)                              # top 3 examples

for _, row in kg_examples.iterrows():                          # iterate examples
    print(f"Q:    {row['question']}")                          # question text
    print(f"Gold: {row['gold']}")                              # correct answer
    print(f"Pred: {row['prediction']}  (score={row['score']})") # wrong prediction
    print(f"EM={row['em']}  F1={row['f1']}")                   # both near zero
    print("-" * 60)

print("""
Production implication:
  → Add retrieval (RAG): fetch a Wikipedia passage before running
    the extractive reader. Without context, recall is bounded by
    the model's parametric knowledge at training-cutoff.
""")
```

---

### Failure Mode 2: Surface-Form / Normalization Failures

**Pattern:** The prediction is semantically correct but uses a different
surface form from any gold alias. After normalization, EM = 0 but
F1 > 0. Classic examples: "USA" vs "United States", "1985" vs
"nineteen eighty-five".

**Why this matters:** A large EM–F1 gap in your aggregate metrics is the
signal that this failure mode is active. Expanding the alias set (e.g.,
from Wikidata Q-item labels) or switching to an NLI-based equivalence
metric resolves it.

**Filter logic:**
```python
sf_mask = (df["em"] == 0) & (df["f1"] > 0.4)
```
EM = 0 but meaningful token overlap suggests a near-correct answer.

### Full Code — Failure Mode 2

```python
print("=" * 60)
print("FAILURE MODE 2: Surface-Form / Normalization Failures")
print("=" * 60)
print("""
Pattern : The prediction is semantically correct but uses a different
          surface form from the gold answer. After normalization,
          EM = 0 but F1 > 0 (partial token overlap).
          Examples: 'USA' vs 'United States', '1985' vs 'nineteen eighty-five'.
""")

sf_mask = (df["em"] == 0) & (df["f1"] > 0.4)                  # surface-form filter
sf_examples = df[sf_mask].head(3)                              # top 3 examples

for _, row in sf_examples.iterrows():                          # iterate examples
    print(f"Q:    {row['question']}")
    print(f"Gold: {row['gold']}")
    print(f"Pred: {row['prediction']}")
    print(f"EM={row['em']}  F1={row['f1']:.3f}")              # F1 > 0, EM = 0
    print("-" * 60)

# Manual worked example to guarantee illustration
print("\n── Worked Normalization Example ──")
ex_pred = "United States of America"                           # prediction form
ex_gold = "USA"                                                # gold form
print(f"Prediction : '{ex_pred}'")
print(f"Gold       : '{ex_gold}'")
print(f"norm(pred) : '{normalize_answer(ex_pred)}'")           # after normalization
print(f"norm(gold) : '{normalize_answer(ex_gold)}'")           # after normalization
print(f"EM : {exact_match(ex_pred, [ex_gold])}")               # still 0
print(f"F1 with alias 'United States' : "
      f"{token_f1(ex_pred, [ex_gold, 'United States']):.3f}")  # F1 > 0

print("""
Production implication:
  → Expand the alias list (Wikidata Q-items carry many surface forms).
  → Use model-based equivalence (NLI score) instead of string EM.
""")
```

---

### Failure Mode 3: Distractor-Pull Failures

**Pattern:** The question contains a salient near-answer — a famous name,
year, or place — and the model latches onto it instead of the correct
(often rarer) answer. The model's confidence (`score`) can be
misleadingly high. This demonstrates why **score alone is not a
reliability signal**.

**Filter logic:**
```python
dp_mask = (df["em"] == 0) & (df["prediction"].str.split().str.len() <= 2) & (df["score"] > 0.5)
```
We look for short predictions (1–2 tokens, typical of a grabbed named
entity) that are wrong but confident.

### Full Code — Failure Mode 3

```python
print("=" * 60)
print("FAILURE MODE 3: Distractor-Pull Failures")
print("=" * 60)
print("""
Pattern : The question contains a salient near-answer (famous name,
          year, or place) and the model latches onto it instead of the
          correct rarer answer. Score can be misleadingly high.
          This shows why score alone is not a reliability signal.
""")

dp_mask = (
    (df["em"] == 0) &                                          # wrong answer
    (df["prediction"].str.split().str.len() <= 2) &            # short prediction
    (df["score"] > 0.5)                                        # confident model
)
dp_examples = df[dp_mask].head(3)                              # top 3 examples

for _, row in dp_examples.iterrows():                          # iterate examples
    print(f"Q:    {row['question']}")
    print(f"Gold: {row['gold']}")
    print(f"Pred: '{row['prediction']}'  (score={row['score']:.3f})")  # confident + wrong
    print(f"EM={row['em']}  F1={row['f1']:.3f}")
    print("-" * 60)

print("""
Production implication:
  → Calibrate confidence: a well-calibrated model's score should
    track accuracy. Plot score vs. EM across buckets. If score > 0.8
    rows have EM < 0.5, the model is overconfident and score cannot
    be used as a threshold for automated acceptance.
""")
```

---

## Section 7 — Pivot to the Learner Lab

This section is a markdown cell. No code.

### What it does

The pivot cell closes the SI demo and frames the learner assignment.
It explicitly contrasts the TriviaQA demo dataset (no context, knowledge-gap
dominant, low EM expected) with the learner's news QA dataset (full
passages, span-extractive, higher EM expected, span-boundary errors
dominate).

The table prevents learners from directly copying the SI's failure mode
analysis: the datasets are structurally different, so the same three
modes will not appear in the same proportion in the learner's results.

---

## How the Notebook Connects to the Learner Assignment

| SI Demo element | Learner replaces with |
|---|---|
| `"trivia_qa"`, `"rc.nocontext"` | Their curated news QA dataset |
| `split="validation[:150]"` | Full ~1,000-example set |
| `deepset/roberta-base-squad2` | Same model (or upgraded) |
| Three failure modes (knowledge-gap, surface-form, distractor-pull) | Three failure modes from their own data |
| `triviaqa_results.csv` | `news_qa_results.csv` |

The `normalize_answer`, `exact_match`, and `token_f1` functions are
identical — learners implement them themselves rather than copy-pasting.
