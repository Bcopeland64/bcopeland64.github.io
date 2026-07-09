# M7B SI Demo Walkthrough: ROUGE Evaluation on XSum with BART-CNN

**Audience:** Supplemental Instruction leader running the 90-min Day 4 lab demo  
**Corpus:** XSum validation subset (50 examples) — BBC news, single-sentence abstractive references  
**Model:** `facebook/bart-large-cnn` — trained on CNN/DailyMail  
**Core tension being demonstrated:** CNN/DM-trained BART produces paragraph summaries; XSum expects one compressed sentence. This length and style mismatch is what drives the ROUGE scores you will see — and discussing it is the pedagogical heart of the demo.

---

## Section 1 — Environment Setup

### What we are doing and why

Before any ML work can run in a fresh environment we need four libraries. Installing them at the top of the notebook makes the dependency list visible to anyone who opens the file.

| Library | Purpose |
|---|---|
| `transformers` | HuggingFace library that hosts the BART-CNN model and the `pipeline` abstraction |
| `datasets` | HuggingFace library for streaming benchmark datasets like XSum from the Hub |
| `rouge-score` | Google's reference implementation of ROUGE — the same scorer used in ACL papers |
| `pandas` / `matplotlib` | Tabular storage and visualization of per-example scores |

### Line-by-line explanation

**Install cell**

```python
import sys
!{sys.executable} -m pip install -q "transformers<5.0" datasets rouge-score pandas matplotlib
print("Restart the kernel now if transformers was just downgraded, then re-run from the top.")
```

`{sys.executable}` interpolates the path of the Python interpreter that is actually running the kernel. This ensures pip installs into the kernel's own environment rather than whichever `python3` happens to be first on the system PATH — a common source of "package installed but not importable" errors in shared environments.

`"transformers<5.0"` pins the version below 5.0 because the `"summarization"` pipeline task was removed in Transformers v5.0. Without this pin, a fresh install could pull in v5 and immediately raise a `ValueError` when the pipeline is constructed. The print reminder asks learners to restart the kernel if the downgrade just ran — a stale in-memory import of a newer version will persist until the kernel restarts.

The `-q` flag suppresses the verbose pip output so cell output stays readable. Re-running this cell is always safe — pip checks whether a package is already installed before doing anything.

**Import cell**

```python
from transformers import pipeline        # line 1
```
`pipeline` is a high-level wrapper that bundles tokenisation, model inference, and de-tokenisation into a single callable. You pass it a task name and model ID; it returns a callable that accepts raw text.

```python
from datasets import load_dataset       # line 2
```
`load_dataset` connects to the HuggingFace Hub (or a local cache) and returns a `DatasetDict`. The slice syntax we use later (`"validation[:50]"`) means only the first 50 examples are downloaded.

```python
from rouge_score import rouge_scorer as rs   # line 3
```
We alias the module `rs` so that constructing the scorer later reads as `rs.RougeScorer(...)`. The scorer object is thread-safe and cheap to construct, so we build it once and reuse it.

```python
import pandas as pd                     # line 4
import numpy as np                      # line 5
import matplotlib.pyplot as plt         # line 6
```
Standard scientific Python stack. `numpy` is imported even though we rarely call it directly — `pandas` occasionally returns numpy types that are easier to format if numpy is in scope.

```python
import warnings
warnings.filterwarnings("ignore")       # line 8
```
Suppresses deprecation warnings from older tokeniser code inside Transformers. These warnings do not indicate errors; hiding them keeps the demo cell output focused on the outputs we want to discuss.

### Full code block

```python
import sys
!{sys.executable} -m pip install -q "transformers<5.0" datasets rouge-score pandas matplotlib
print("Restart the kernel now if transformers was just downgraded, then re-run from the top.")

from transformers import pipeline
from datasets import load_dataset
from rouge_score import rouge_scorer as rs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

---

## Section 2 — Load XSum Demo Subset

### What we are doing and why

XSum (Extreme Summarization) is a BBC News dataset where each article is paired with a single human-written summary sentence. It was designed specifically to test abstractive summarization — summaries that paraphrase and compress rather than copy. Using 50 examples keeps inference fast enough to finish in a live 90-minute session while still producing a meaningful distribution of ROUGE scores.

### Line-by-line explanation

```python
ds = load_dataset("xsum", split="validation[:50]")
```

`"xsum"` is the dataset identifier on the HuggingFace Hub. `split="validation[:50]"` is a slice expression that streams only the first 50 rows of the 11,332-example validation split — no full download required. The result is a `Dataset` object with three columns: `"document"` (the BBC article), `"summary"` (the one-sentence reference), and `"id"` (a BBC article ID).

```python
print(f"Dataset size   : {len(ds)} examples")
print(f"Column names   : {ds.column_names}")
```

Confirmation prints. The first confirms we got 50 rows. The second surfaces the exact column names we will use to access data in later cells.

```python
print(ds[0]["document"][:300])
```

`ds[0]` accesses the first example as a Python dict. `[:300]` slices the first 300 characters of the article — enough to see the topic and writing style without flooding the output.

```python
print(ds[0]["summary"])
```

Prints the XSum gold reference summary. **Draw attention here:** it is almost certainly a single sentence of 20–30 words. This is the target our model is being measured against, and it is very different from the multi-sentence paragraph the CNN/DM-trained model will produce.

### Full code block

```python
ds = load_dataset("xsum", split="validation[:50]")

print(f"Dataset size   : {len(ds)} examples")
print(f"Column names   : {ds.column_names}")
print()
print("=== Article (first 300 chars) ===")
print(ds[0]["document"][:300])
print()
print("=== XSum Reference Summary ===")
print(ds[0]["summary"])
```

---

## Section 3 — Pipeline Construction + One-Document Inference

### What we are doing and why

This section shows learners the full API surface of the HuggingFace pipeline so they know exactly what object they are working with and why the output is structured the way it is. The output structure — a list of dicts, accessed via `result[0]["summary_text"]` — is a constant source of confusion if it is not explicitly shown.

### Line-by-line explanation

```python
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)
```

`"summarization"` tells HuggingFace which task head and default tokeniser to load. `model="facebook/bart-large-cnn"` downloads the CNN/DailyMail fine-tuned BART checkpoint (~1.6 GB on first run; cached after that). `device=-1` forces CPU inference. Change to `device=0` to use the first available GPU, which reduces inference time from ~20 seconds per article to ~1–2 seconds.

```python
result = summarizer(article, max_length=130, min_length=30, do_sample=False)
```

Calling `summarizer` tokenises the text, runs the encoder-decoder forward pass, applies beam search, and de-tokenises back to a string. `max_length=130` and `min_length=30` set the token-count bounds on the output. `do_sample=False` forces greedy/beam decoding — no randomness — which is required for reproducibility.

```python
print("Return type      :", type(result))      # <class 'list'>
print("Element type     :", type(result[0]))   # <class 'dict'>
print("Dict keys        :", result[0].keys())  # dict_keys(['summary_text'])
```

These three prints exist purely to make the data structure explicit. The pipeline returns a **list** because it supports batched inputs (a list of articles). Element zero is a **dict** with one key: `"summary_text"`. Learners often write `result["summary_text"]` and get a `TypeError` — this cell prevents that.

```python
print(result[0]["summary_text"])
print(ds[0]["summary"])
```

Side-by-side comparison. The model output will be noticeably longer and more descriptive; the XSum reference will be one tight, often highly rephrased, sentence. **This difference is the central observation of the demo — revisit it throughout.**

### Full code block

```python
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

article = ds[0]["document"]

result = summarizer(article, max_length=130, min_length=30, do_sample=False)

print("Return type      :", type(result))
print("Element type     :", type(result[0]))
print("Dict keys        :", result[0].keys())
print()
print("=== Generated Summary (BART-CNN) ===")
print(result[0]["summary_text"])
print()
print("=== Reference Summary (XSum style) ===")
print(ds[0]["summary"])
print()
print("Notice: BART-CNN produces a multi-sentence paragraph; XSum expects one sentence.")
```

---

## Section 4 — Generation Hyperparameters

### What we are doing and why

Evaluation results are only reproducible if generation is deterministic. This section gives learners an intuition for how three parameters — `max_length`, `num_beams`, and `do_sample` — each change the output, and it establishes `do_sample=False` as the required setting for any evaluation run.

### Line-by-line explanation

```python
configs = [
    {"max_length": 60,  "num_beams": 4, "do_sample": False, "label": "Short  / beam=4 / greedy"},
    {"max_length": 130, "num_beams": 4, "do_sample": False, "label": "Default / beam=4 / greedy"},
    {"max_length": 130, "num_beams": 1, "do_sample": False, "label": "Default / beam=1 / greedy"},
    {"max_length": 130, "num_beams": 4, "do_sample": True,  "label": "Default / beam=4 / sample"},
]
```

Each dict in the list is a single experiment. Changing ONE variable per experiment follows the scientific method — learners can attribute output differences to the variable that changed. The `label` key is our description; we pop it off before unpacking the rest as kwargs.

```python
label = cfg.pop("label")
out   = summarizer(article, **cfg)
```

`cfg.pop("label")` removes the label from the dict and returns it. `**cfg` then unpacks the remaining keys (`max_length`, `num_beams`, `do_sample`) as keyword arguments to `summarizer`. If we did not pop `label`, the pipeline would raise a `TypeError` about an unexpected argument.

**`max_length`** — caps the number of output tokens. Reducing it forces shorter (and often more extractive) summaries.

**`num_beams`** — controls beam search width. `num_beams=1` is pure greedy decoding: always pick the single highest-probability next token. `num_beams=4` maintains four candidate sequences in parallel and returns the one with the highest overall probability. Higher beams generally produce more fluent output at higher computational cost.

**`do_sample`** — when `True`, tokens are drawn from the probability distribution rather than taken greedily. This introduces randomness: running the same config twice gives different outputs. **Never use `do_sample=True` in an evaluation run** because you lose reproducibility.

### Full code block

```python
article = ds[1]["document"]

configs = [
    {"max_length": 60,  "num_beams": 4, "do_sample": False, "label": "Short  / beam=4 / greedy"},
    {"max_length": 130, "num_beams": 4, "do_sample": False, "label": "Default / beam=4 / greedy"},
    {"max_length": 130, "num_beams": 1, "do_sample": False, "label": "Default / beam=1 / greedy"},
    {"max_length": 130, "num_beams": 4, "do_sample": True,  "label": "Default / beam=4 / sample"},
]

for cfg in configs:
    label = cfg.pop("label")
    out   = summarizer(article, **cfg)
    print(f"[{label}]")
    print(out[0]["summary_text"])
    print()

print("-" * 70)
print("Key takeaway: do_sample=False makes generation deterministic.")
print("Always use do_sample=False when running evaluation runs.")
```

---

## Section 5 — ROUGE Harness on a Fixture

### What we are doing and why

Before running ROUGE over 50 examples, we verify the harness on two hand-crafted sentences where we can reason about the expected scores ourselves. This "fixture test" catches the most common bug — swapping the argument order of `scorer.score()` — before it silently corrupts all 50 results.

### What ROUGE measures

| Metric | What it counts |
|---|---|
| ROUGE-1 | Unigram (single word) overlap between reference and prediction |
| ROUGE-2 | Bigram (two consecutive word) overlap |
| ROUGE-L | Longest Common Subsequence — sensitive to word order but not strict contiguity |

Each metric returns three values: **Precision** (fraction of predicted n-grams found in the reference), **Recall** (fraction of reference n-grams found in the prediction), and **F1** (harmonic mean of precision and recall). In most NLP papers the **F1** is the reported number — in this code it is accessed as `.fmeasure`.

### Line-by-line explanation

```python
scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
```

Instantiates the scorer and tells it which metrics to compute. `use_stemmer=True` applies the Porter stemmer so that "running" and "run" are treated as the same token — this reduces surface-form variation without changing meaning.

```python
reference = "The president signed the bill into law on Monday."
predicted  = "On Monday, the president signed the legislation."
```

Two sentences with identical meaning but different word choice ("bill" vs "legislation") and different word order. This makes the fixture interesting: ROUGE-1 should be high (most words overlap), ROUGE-2 should drop (bigrams are different once word order changes), and ROUGE-L should be moderate (the LCS "the president signed … on Monday" exists but is non-contiguous).

```python
scores = scorer.score(reference, predicted)
```

**The argument order is `(reference, predicted)`, not `(predicted, reference)`.** Swapping them swaps precision and recall. With symmetric data this is invisible, but with length-mismatched data (like our 50-example run) swapping silently inverts which direction you are penalising length. Always verify this order against the library documentation.

```python
for metric, score in scores.items():
    print(f"{metric:<10}{score.precision:.3f}      {score.recall:.3f}     {score.fmeasure:.3f}")
```

`scores` is a dict keyed by metric name. Each value is a `Score` named-tuple with `.precision`, `.recall`, and `.fmeasure` attributes. `.fmeasure` is the F1.

### Full code block

```python
scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

reference = "The president signed the bill into law on Monday."
predicted  = "On Monday, the president signed the legislation."

scores = scorer.score(reference, predicted)

print("Metric    Precision  Recall    F1")
print("-" * 40)
for metric, score in scores.items():
    print(f"{metric:<10}{score.precision:.3f}      {score.recall:.3f}     {score.fmeasure:.3f}")

print()
print("'legislation' ≠ 'bill' → ROUGE-2 drops because the bigrams differ.")
print("ROUGE-L uses longest common subsequence — robust to word-order changes.")
```

---

## Section 6 — Run Pipeline + Harness Over XSum Subset

### What we are doing and why

This is the core data collection loop: generate a summary for each of the 50 articles, score it against the XSum gold reference, and store every result in a list that becomes a DataFrame. The loop is intentionally simple — no batching, no async — so learners can follow it line by line.

### Line-by-line explanation

```python
scorer  = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
records = []
```

Re-instantiate the scorer (safe to do; it is stateless) and create an empty list. Each iteration will append one dict to `records`.

```python
for i, example in enumerate(ds):
```

`enumerate` gives us the numeric index `i` alongside the dict `example`. We need `i` for two reasons: progress printing and a stable row identifier in the final DataFrame.

```python
    tok_inputs = summarizer.tokenizer(
        example["document"],
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    safe_doc = summarizer.tokenizer.decode(
        tok_inputs["input_ids"][0],
        skip_special_tokens=True
    )
```

BART has a hard positional embedding limit of 1,024 tokens. Some XSum articles (typically around example 40) exceed this and trigger an `IndexError` deep in the encoder. Passing `truncation=True` directly to the pipeline is unreliable in some Transformers versions — the flag is not always forwarded to the tokenizer before the encoder runs. The fix is to truncate manually: tokenize with a 1,024-token cap, then decode back to a plain string. `skip_special_tokens=True` removes the BOS/EOS markers so `safe_doc` looks like a normal string to the pipeline.

```python
    gen = summarizer(
        safe_doc,
        max_length=130,
        min_length=30,
        do_sample=False
    )
    generated = gen[0]["summary_text"]
```

Generates one summary from the safely truncated document. The settings match the "default" config from Section 4 — `do_sample=False` ensures the same article always produces the same summary. We immediately unwrap the list-of-dicts to get the plain string.

```python
    reference = example["summary"]
    scores    = scorer.score(reference, generated)
```

`example["summary"]` is the XSum gold sentence. `scorer.score(reference, generated)` — reference first, generated second.

```python
    records.append({
        "id"        : i,
        "article"   : example["document"],
        "reference" : reference,
        "generated" : generated,
        "r1_f"      : scores["rouge1"].fmeasure,
        "r2_f"      : scores["rouge2"].fmeasure,
        "rL_f"      : scores["rougeL"].fmeasure
    })
```

We store the full article text in `records` so we can pull it back in Section 8 for the faithfulness check without going back to the dataset.

```python
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} / 50...")
```

Progress marker every 10 examples. On CPU each article takes ~5–20 seconds, so 50 examples takes up to ~15 minutes. The progress print reassures learners the kernel has not hung.

```python
df = pd.DataFrame(records)
print(df[["r1_f", "r2_f", "rL_f"]].describe().round(3))
```

Converts the list of dicts to a DataFrame. `.describe()` immediately gives us count, mean, std, min, quartiles, and max for each metric — a quick sanity check before we plot.

### Full code block

```python
scorer  = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
records = []

for i, example in enumerate(ds):
    # Pre-truncate to BART's hard 1024-token position limit using the tokenizer directly.
    # Passing truncation=True to the pipeline is unreliable in some transformers versions —
    # the flag may not be forwarded before the encoder runs, causing an IndexError on long
    # documents (typically around example 40).
    tok_inputs = summarizer.tokenizer(
        example["document"],
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    safe_doc = summarizer.tokenizer.decode(
        tok_inputs["input_ids"][0],
        skip_special_tokens=True
    )

    gen       = summarizer(
        safe_doc,
        max_length=130,
        min_length=30,
        do_sample=False
    )
    generated = gen[0]["summary_text"]
    reference = example["summary"]

    scores = scorer.score(reference, generated)

    records.append({
        "id"        : i,
        "article"   : example["document"],
        "reference" : reference,
        "generated" : generated,
        "r1_f"      : scores["rouge1"].fmeasure,
        "r2_f"      : scores["rouge2"].fmeasure,
        "rL_f"      : scores["rougeL"].fmeasure
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} / 50...")

df = pd.DataFrame(records)
print("\nDone. Shape:", df.shape)
print(df[["r1_f", "r2_f", "rL_f"]].describe().round(3))
```

---

## Section 7 — Aggregate ROUGE & Per-Summary Distribution

### What we are doing and why

The aggregate means are the single-number headline that would appear in a paper or leaderboard. The distribution plot shows whether the mean is representative or whether a few outliers are dragging it in one direction. Both are needed for an honest evaluation.

### Line-by-line explanation

**Aggregate means cell**

```python
for col, label in [("r1_f", "ROUGE-1"), ("r2_f", "ROUGE-2"), ("rL_f", "ROUGE-L")]:
    print(f"  {label} mean F1: {df[col].mean():.3f}")
```

Loops over the three metric columns and prints each mean to three decimal places. The printed values will be noticeably lower than CNN/DM benchmarks (~0.44 ROUGE-1). That gap is the teaching moment: it is not because the model is bad — it is because the reference style is completely different.

**Distribution plot cell**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
```

Creates a figure with three side-by-side panels, each 5 inches wide and 4 inches tall. `axes` is a numpy array of three `Axes` objects.

```python
for ax, col, title in zip(axes, ["r1_f", "r2_f", "rL_f"], ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"]):
    ax.hist(df[col], bins=15, color="steelblue", edgecolor="black", alpha=0.75)
```

`zip` pairs each axis with a column name and a display title. `bins=15` gives enough granularity to see shape without being too noisy for 50 points. `alpha=0.75` makes bars slightly transparent.

```python
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.3f}")
```

Draws a vertical dashed red line at the mean. If the distribution is skewed, the mean line will visually sit off-centre — a good discussion point about whether mean is the right summary statistic.

```python
plt.savefig("rouge_distributions.png", bbox_inches="tight")
```

Saves the figure to disk so it can be dropped into slides or the integration report without re-running the notebook.

```python
best_idx  = df["r1_f"].idxmax()
worst_idx = df["r1_f"].idxmin()
```

`.idxmax()` / `.idxmin()` return the DataFrame **index** (not the value) of the row with the highest/lowest ROUGE-1. We use these indices in Section 8 to pull out the specific example for the faithfulness check.

### Full code block

```python
print("=== Aggregate ROUGE — BART-CNN on XSum (n=50) ===")
for col, label in [("r1_f", "ROUGE-1"), ("r2_f", "ROUGE-2"), ("rL_f", "ROUGE-L")]:
    print(f"  {label} mean F1: {df[col].mean():.3f}")

print()
print("Context: CNN/DM-trained BART scores ~44 ROUGE-1 on CNN/DM test set.")
print("XSum references are ~1 sentence; BART-CNN outputs ~3–4 sentences.")
print("Length mismatch → ROUGE recall is penalised → lower F1 than CNN/DM benchmark.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col, title in zip(
    axes,
    ["r1_f", "r2_f", "rL_f"],
    ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"]
):
    ax.hist(df[col], bins=15, color="steelblue", edgecolor="black", alpha=0.75)
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.3f}")
    ax.set_title(title)
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Count")
    ax.legend()

plt.suptitle("ROUGE Score Distribution — BART-CNN on XSum (n=50)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("rouge_distributions.png", bbox_inches="tight")
plt.show()

best_idx  = df["r1_f"].idxmax()
worst_idx = df["r1_f"].idxmin()
print(f"Best  ROUGE-1: {df.loc[best_idx,  'r1_f']:.3f}  (example #{df.loc[best_idx,  'id']})")
print(f"Worst ROUGE-1: {df.loc[worst_idx, 'r1_f']:.3f}  (example #{df.loc[worst_idx, 'id']})")
```

---

## Section 8 — Faithfulness Check

### What we are doing and why

This section demonstrates ROUGE's most important limitation: it measures lexical overlap with a reference string. A summary that uses the same words as the reference can score well while asserting things that are not in the source article. A summary that paraphrases accurately can score poorly. ROUGE does not read the article at all.

### Line-by-line explanation

```python
top = df.loc[df["r1_f"].idxmax()]
```

`.idxmax()` finds the integer index of the row with the highest ROUGE-1 F1. `.loc[...]` retrieves that row as a pandas Series. We use the best-scoring example on the hypothesis that "the model felt most confident here" — which sometimes correlates with overconfident hallucination.

```python
print(f"Example #{int(top['id'])}  |  ROUGE-1 F1 = {top['r1_f']:.3f}")
print(top["article"][:600])
print(top["reference"])
print(top["generated"])
```

Side-by-side display. 600 characters is usually enough to cover the first two paragraphs of a BBC article — where the key facts are. The faithfulness check is manual: read the generated summary, find each factual claim in the article text, and flag any claim that cannot be found.

**What to look for:**
- Named entities that appear in the summary but not the article (hallucinated names)
- Numeric claims (statistics, dates, amounts) that differ from the article
- Causal relationships that reverse what the article says
- Events described as confirmed that the article presents as alleged

```python
print("""
--- Manual Faithfulness Check Protocol ---
1. Read the generated summary.
2. For each factual claim, highlight it in the article text.
3. Any claim with no highlight = potential hallucination.
...
""")
```

Gives learners a repeatable protocol they can apply to their own corpus in the assignment.

**Word count comparison cell**

```python
df["ref_len"] = df["reference"].apply(lambda x: len(x.split()))
df["gen_len"] = df["generated"].apply(lambda x: len(x.split()))
```

`.apply(lambda x: len(x.split()))` splits each string on whitespace and counts the tokens — a rough but fast word count. We add two new columns to the DataFrame so we can compare mean lengths.

```python
print(f"Reference avg length : {df['ref_len'].mean():.1f} words")
print(f"Generated avg length : {df['gen_len'].mean():.1f} words")
```

Expected output: reference ~20–25 words, generated ~60–80 words. The ratio (roughly 3:1) directly explains the ROUGE recall gap seen in Section 7: only a fraction of the generated tokens can possibly overlap with a one-sentence reference.

### Full code block

```python
top = df.loc[df["r1_f"].idxmax()]

print(f"Example #{int(top['id'])}  |  ROUGE-1 F1 = {top['r1_f']:.3f}")
print("=" * 70)

print("\n--- SOURCE ARTICLE (first 600 chars) ---")
print(top["article"][:600])

print("\n--- REFERENCE SUMMARY (XSum gold) ---")
print(top["reference"])

print("\n--- GENERATED SUMMARY (BART-CNN) ---")
print(top["generated"])

print("""
--- Manual Faithfulness Check Protocol ---
1. Read the generated summary.
2. For each factual claim, highlight it in the article text.
3. Any claim with no highlight = potential hallucination.
4. Record: entity errors, relation errors, out-of-article facts.

Key lesson: ROUGE only measures n-gram overlap with the reference.
It cannot detect facts that are fluent but absent from the source.
A summary can score 0.40+ ROUGE-1 and still hallucinate.
""")

df["ref_len"] = df["reference"].apply(lambda x: len(x.split()))
df["gen_len"] = df["generated"].apply(lambda x: len(x.split()))

print("=== Word Count Comparison ===")
print(f"Reference avg length : {df['ref_len'].mean():.1f} words")
print(f"Generated avg length : {df['gen_len'].mean():.1f} words")
print()
print("BART-CNN was trained on CNN/DM where references are paragraph-length.")
print("It has no incentive to compress to XSum's single-sentence style.")
print("This length ratio directly suppresses ROUGE recall.")
```

---

## Section 9 — Pivot to the Integration

### What we are doing and why

The final section bridges the demo to the learner assignment. The SI's role ends here: we have shown the measurement tool and its failure modes. The learner's job is to run the same harness on their own corpus and synthesise the result with the two other M7 measurements into a decision recommendation.

### Line-by-line explanation

```python
summary_table = pd.DataFrame({
    "Corpus"           : [...],
    "Reference style"  : [...],
    "ROUGE-1 F1 (est.)": [...],
    "Key failure mode" : [...]
})
```

A comparison table with three rows: CNN/DM (the model's home domain), XSum (this demo), and the learner's corpus. The third row has "Run and report" for the ROUGE score — a deliberate gap that frames what the learner must contribute. Presenting this as a DataFrame makes it easy to add or modify rows in future semesters.

```python
print(summary_table.to_string(index=False))
```

`.to_string(index=False)` prints the table without the default row-number index, which looks cleaner in a terminal-style output cell.

**The integration print block** explains the three-measurement structure of the Module 7 assignment:
- BERTScore measures semantic similarity in embedding space — it can match paraphrases that ROUGE misses
- Readability scores (Flesch, Gunning Fog) measure whether a non-expert audience can understand the output
- ROUGE measures lexical overlap — this lab

The decision matrix question at the end is the engineering act: given these three measurements, would you deploy this model? The answer is not in the notebook — it lives in the learner's written report.

### Full code block

```python
summary_table = pd.DataFrame({
    "Corpus"           : ["CNN/DailyMail (BART-CNN native)", "XSum (SI demo — this notebook)", "Your corpus (M7 assignment)"],
    "Reference style"  : ["Paragraph (3–5 sentences)",        "Single abstractive sentence",    "Paragraph — tech/entertainment news"],
    "ROUGE-1 F1 (est.)": ["~0.44",                            "~0.18–0.25",                    "Run and report"],
    "Key failure mode" : ["Extractive drift",                  "Length mismatch + abstraction gap", "TBD — your analysis"]
})

print(summary_table.to_string(index=False))

print("""
=== Your Integrated Report (Module 7 Assignment) ===

You will run the SAME harness on your assigned corpus and report:
  - ROUGE-1, ROUGE-2, ROUGE-L mean F1
  - Per-summary distribution
  - At least one faithfulness failure ROUGE did not catch

Then synthesise all three M7 measurements:
  [M7A] BERTScore      — semantic similarity (embedding space)
  [M7B] Readability    — Flesch / Gunning Fog on generated text
  [M7B] ROUGE          — lexical overlap with reference (this lab)

Decision matrix question:
  'Given these three measurements, would you deploy this model for
   your domain? What trade-offs would you document for stakeholders?'

The SI does NOT write this report — that integration is YOUR work.
""")
```

---

## Quick Reference: Common Mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| `scorer.score(predicted, reference)` | Precision and recall are swapped | Always: `scorer.score(reference, predicted)` |
| `result["summary_text"]` | `TypeError` — result is a list | Always: `result[0]["summary_text"]` |
| `do_sample=True` in evaluation | Different output every run | Use `do_sample=False` for all evaluation runs |
| Comparing XSum ROUGE to CNN/DM benchmarks | Apples to oranges | Note domain and reference-style differences |
| Using `.fmeasure` only | Missing the precision/recall story | Print all three; discuss what a recall gap means |

---

## Differentiation Note for SI

This notebook uses the **XSum** corpus. The learner assignment uses a **tech/entertainment news corpus with paragraph-length references**. The SI does **not** compose the integrated report — that three-metric synthesis across M7A BERTScore, M7B Readability, and M7B ROUGE is the individual learner deliverable. The SI's role is to demonstrate the measurement tool and its failure modes so that learners can make informed engineering decisions in their own reports.
