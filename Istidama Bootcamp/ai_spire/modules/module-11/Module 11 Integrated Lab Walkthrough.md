# Module 11 Integrated Lab — Full Walkthrough (Simplified)

This explains the code in the lab, grouped by the same six sections as the 90-minute session. A complete, runnable code block appears at the end of each section.

---

## Section 1 — Setup and Orientation (10 min)

### What you are looking at

The lab gives you two pre-built pieces you don't touch:

- **`sentiment-api/app.py`** — a FastAPI service. Send it `{"text": "..."}`, it sends back `{"sentiment": "...", "confidence": ..., "latency_ms": ...}`.
- **`data/sentiment_demo.json`** — 15 hand-labeled product reviews (the "correct answers").

Your job: send every review to the service, collect its guesses, compare them to the correct answers, and write up a report.

---

### Imports and environment setup

```python
%pip install fastapi uvicorn httpx pytest --quiet
```
Installs the packages this lab needs: `fastapi`/`uvicorn` run the API, `httpx` calls it, `pytest` runs the tests. `--quiet` hides the install noise.

```python
import sys
```
Lets us edit where Python looks for modules to import.

```python
import json
```
For turning JSON text into Python objects and back.

```python
import pathlib
```
A cleaner way to work with file paths than raw strings — e.g. `Path('data/file.json').read_text()` reads a whole file in one line.

```python
import subprocess
```
Lets Python start and run other programs. We use it to start the server in the background and to run scripts/tests.

```python
import time
```
Used for a short pause (`time.sleep(2)`) so the server has time to start before we send it requests.

```python
import httpx
```
An HTTP client. `httpx.post(url, json=body)` sends data as JSON and gives back a response we can read with `.json()`.

```python
sys.path.insert(0, '.')
```
Tells Python "also look in this folder for imports," so `from lib.scorers import accuracy` works.

---

### Starting the API server

```python
server = subprocess.Popen(
    ['python', '-m', 'uvicorn',
     'app:app',
     '--port', '8000',
     '--log-level', 'error'],
    cwd='sentiment-api',
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
```
This starts the server in the background — the notebook keeps running while it starts up.

- `'app:app'` — run the `app` object found inside `app.py`.
- `'--port', '8000'` — run it on port 8000.
- `'--log-level', 'error'` — keep the server quiet; change to `'info'` if you need to debug it.
- `cwd='sentiment-api'` — start it from inside that folder so it can find its own files.
- The `DEVNULL` lines just hide the server's console output. Swap in `None` if you want to see it.

```python
time.sleep(2)
```
Give the server 2 seconds to finish starting before we send it any requests.

```python
BASE_URL = 'http://localhost:8000'
```
The server's address, stored once so we don't repeat it everywhere.

```python
print(f'Server PID: {server.pid} — ready at {BASE_URL}')
```
Shows the server's process ID — useful if you need to stop it manually later.

---

### Loading the labeled dataset

```python
DATA_PATH = pathlib.Path('data/sentiment_demo.json')
```
Stores the file location as a path object (gives us handy methods like `.read_text()`).

```python
reviews = json.loads(DATA_PATH.read_text())
```
Reads the file's text, then turns that text into a Python list of dictionaries — one per review.

```python
print(f'Loaded {len(reviews)} reviews\n')
```
Quick check that the file loaded correctly and isn't empty.

```python
for r in reviews[:5]:
    rid   = r['id']
    label = r['label']
    text  = r['text'][:45]
    print(f'{rid:3d}  {label:8s}  {text}')
```
Prints the first 5 reviews neatly: ID, label, and a short preview of the text, lined up in columns.

---

### The first API call

```python
sample = reviews[0]
```
Grab one review just to see what the API's response looks like.

```python
response = httpx.post(
    f'{BASE_URL}/predict',
    json={'text': sample['text']},
    timeout=5.0,
)
```
Sends the review's text to the `/predict` endpoint. `timeout=5.0` makes it fail loudly (instead of hanging forever) if the server doesn't respond within 5 seconds.

```python
payload = response.json()
pred    = payload['sentiment']
truth   = sample['label']
```
Pulls the model's guess and the correct answer out of the response.

```python
print(f'Match : {truth == pred}')
```
Compares the two — `True` if the model got it right. We'll do this for all 15 reviews in Section 2.

---

### Complete code — Section 1

```python
%pip install fastapi uvicorn httpx pytest --quiet

import sys, json, pathlib, subprocess, time
import httpx

sys.path.insert(0, '.')

# ── Start the API server ───────────────────────────────────────────────────
server = subprocess.Popen(
    ['python', '-m', 'uvicorn', 'app:app', '--port', '8000', '--log-level', 'error'],
    cwd='sentiment-api',
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(2)
BASE_URL = 'http://localhost:8000'
print(f'Server PID: {server.pid} — ready at {BASE_URL}')

# ── Load the labeled dataset ───────────────────────────────────────────────
DATA_PATH = pathlib.Path('data/sentiment_demo.json')
reviews   = json.loads(DATA_PATH.read_text())
print(f'Loaded {len(reviews)} reviews\n')
for r in reviews[:5]:
    rid, label, text = r['id'], r['label'], r['text'][:45]
    print(f'{rid:3d}  {label:8s}  {text}')

# ── First API call ─────────────────────────────────────────────────────────
sample   = reviews[0]
response = httpx.post(f'{BASE_URL}/predict', json={'text': sample['text']}, timeout=5.0)
payload  = response.json()
pred, truth = payload['sentiment'], sample['label']
print(json.dumps(payload, indent=2))
print(f'\nGround truth : {truth}')
print(f'Predicted    : {pred}')
print(f'Match        : {truth == pred}')
```

---

## Section 2 — The Load → Iterate → Score → Accumulate → Report Pattern (15 min)

### Why a script, not a notebook?

Notebooks let you run cells out of order, so re-running one can change your results. Scripts always run top to bottom the same way — same input, same output, every time.

That matters here because evaluation reports need to be **reproducible**: running the harness again tomorrow should give the same numbers. The notebook is for *understanding* the code; the script (`eval_sentiment.py`) is what actually gets *run* for real.

---

### Displaying the harness

```python
harness_src = pathlib.Path('eval_sentiment.py').read_text()
print(harness_src)
```
Prints the script's code so you can see its overall shape before diving into the details.

---

### Step 1 — LOAD

```python
reviews = json.loads(
    pathlib.Path('data/sentiment_demo.json').read_text()
)
```
Same as before: read the file's text, then parse it into a list of reviews.

---

### Steps 2–4 — ITERATE → POST → ACCUMULATE

```python
results = []
```
An empty list that we'll fill with one result per review.

```python
for review in reviews[:3]:
```
Looping over just 3 reviews here to keep the demo short. The real script loops over all 15.

```python
    resp = httpx.post(
        f'{BASE_URL}/predict',
        json={'text': review['text']},
        timeout=5.0,
    )
```
One request per review — we never batch these together, to keep things simple and traceable.

```python
    payload   = resp.json()
    label     = review['label']
    predicted = payload['sentiment']
    correct   = (label == predicted)
```
Pulls out the correct label and the model's prediction, and checks if they match.

```python
    results.append({
        'id':        rid,
        'text':      review['text'],
        'label':     label,
        'predicted': predicted,
        'correct':   correct,
    })
```
Saves everything about this review in one dictionary. We keep the original `text` too — not for scoring, but so the final report can show exactly which reviews failed and why.

---

### Complete code — Section 2

```python
# Step 1: LOAD
reviews = json.loads(pathlib.Path('data/sentiment_demo.json').read_text())
print(f'LOAD complete: {len(reviews)} reviews ready')

# Steps 2–4: ITERATE → POST → ACCUMULATE
results = []

for review in reviews[:3]:
    resp      = httpx.post(f'{BASE_URL}/predict', json={'text': review['text']}, timeout=5.0)
    payload   = resp.json()
    rid       = review['id']
    label     = review['label']
    predicted = payload['sentiment']
    correct   = (label == predicted)

    results.append({
        'id': rid, 'text': review['text'],
        'label': label, 'predicted': predicted, 'correct': correct,
    })

    status = 'OK  ' if correct else 'FAIL'
    print(f'[{status}] id={rid:2d}  label={label:8s}  pred={predicted}')

print(f'\nAccumulated {len(results)} results')
```

---

## Section 3 — The Methodology Paragraph (10 min)

### Why the same words in three places?

This paragraph guards against a common problem: the spec says one thing should be measured, but the actual code measures something slightly different. Pasting the exact same paragraph in three places means any mismatch shows up immediately.

The three places, and who reads each one:
- **Script docstring** — the engineer running the harness
- **Evaluation spec** — the person grading the results
- **Learner guide** — the student building the harness

Same text everywhere means everyone agrees on what's actually being measured.

---

### The methodology paragraph

```python
METHODOLOGY = (
    'This harness evaluates a black-box sentiment classification service '
    'against a 15-item labeled dataset of product reviews. ...'
)
```
Python automatically joins strings written next to each other like this, so we can write one long paragraph across several lines without special syntax.

The paragraph answers four questions:
1. What data was used → "a 15-item labeled dataset"
2. How predictions were collected → "submitted via HTTP POST to /predict"
3. What metrics were computed → "accuracy and macro-averaged F1"
4. Whether it's repeatable → "The harness is deterministic"

---

### Complete code — Section 3

```python
METHODOLOGY = (
    'This harness evaluates a black-box sentiment classification service '
    'against a 15-item labeled dataset of product reviews.  Each review is '
    'submitted via HTTP POST to /predict.  Predictions are collected, then '
    'scored using accuracy and macro-averaged F1.  A per-item breakdown is '
    'included in the output report so that individual failures are auditable.  '
    "Latency is measured via the service's own /metrics endpoint (p95).  "
    'The harness is deterministic: given the same dataset and the same service '
    'state, it will produce the same scores.'
)

print('=== Evaluation Methodology ===\n')
print(METHODOLOGY)
print('\nThis exact paragraph appears in:')
print('  1. eval_sentiment.py  (module docstring)')
print('  2. the evaluation spec  (rubric)')
print('  3. the learner-facing guide')
```

---

## Section 4 — Scorer Functions Live in `lib/` (10 min)

### The pure-function principle

A **pure function**:
1. Only depends on its inputs — no hidden state or outside data.
2. Has no side effects — doesn't write files, make network calls, or change anything else.

Keeping the scoring logic pure is useful because it means these functions can be:
- **Tested easily** — no server or files needed, just plug in sample data.
- **Reused anywhere** — not tied to this one script.
- **Understood on their own** — all the math lives in one place.

Simple rule: the actual math lives in `lib/`; the harness script just coordinates the pieces.

---

### `lib/scorers.py` — `accuracy`

```python
from typing import Sequence
```
`Sequence` is a type hint meaning "any ordered collection" (list, tuple, etc.) — a flexible way to describe the expected input.

```python
def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
```
The type hints just document what goes in and out — they're not enforced by Python itself, but they help readers and tools like mypy.

```python
    if len(y_true) != len(y_pred):
        raise ValueError(...)
```
If the two lists are different lengths, something's wrong upstream — better to fail clearly than compute a meaningless number.

```python
    if not y_true:
        raise ValueError('Cannot compute accuracy on empty sequences')
```
An empty list would cause a divide-by-zero later, so we catch that early with a clear error instead of a confusing crash.

```python
    matches = sum(t == p for t, p in zip(y_true, y_pred))
    return matches / len(y_true)
```
`zip` pairs up each true label with its matching prediction. We count how many pairs match, then divide by the total to get the fraction correct.

---

### `lib/scorers.py` — `macro_f1`

```python
classes = set(y_true)
```
Gets the unique labels that actually appear in the correct answers (we use `y_true` here, not the predictions, so an unused label doesn't skew the average).

```python
for cls in classes:
    tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
    fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
    fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
```
For each label, we count:
- **TP**: correctly predicted as this label
- **FP**: wrongly predicted as this label (it was actually something else)
- **FN**: should've been this label, but wasn't predicted as such

```python
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / denom if denom > 0 else 0.0
```
Standard precision/recall/F1 formulas. We check for zero before dividing, since a class the model never predicts would otherwise cause a divide-by-zero.

```python
return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
```
**Macro F1** is just the plain average of each label's F1 score. Every label counts equally, even rare ones — this matters if some labels are much more common than others.

---

### `tests/test_scorers.py`

```python
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
```
Adds the project's root folder to Python's search path, so the test file can find `lib/scorers.py` no matter where it's run from.

```python
def test_accuracy_all_correct():
    assert accuracy(['positive', 'negative', 'neutral'],
                    ['positive', 'negative', 'neutral']) == 1.0
```
The simplest test: if predictions exactly match the truth, accuracy should be 1.0. This catches the function breaking in a future edit.

```python
def test_accuracy_one_wrong():
    result = accuracy(y_true, y_pred)
    assert abs(result - 2 / 3) < 1e-9
```
Since computer math with fractions isn't perfectly exact, we check the result is *very close* to 2/3 rather than exactly equal.

```python
def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError, match='Length mismatch'):
        accuracy(['positive'], ['positive', 'negative'])
```
Checks that mismatched list lengths correctly trigger the error we expect, with the right message.

---

### Complete code — Section 4

```python
from lib.scorers import accuracy, macro_f1

# ── accuracy demo ─────────────────────────────────────────────────────────
y_true_a = ['positive', 'negative', 'negative', 'neutral']
y_pred_a = ['positive', 'positive', 'negative', 'neutral']
acc_demo = accuracy(y_true_a, y_pred_a)
print(f'accuracy demo : {acc_demo:.4f}   (3 correct / 4 total = 0.75)')

# ── macro_f1 demo ─────────────────────────────────────────────────────────
y_true_f = ['positive', 'negative', 'neutral']
y_pred_f = ['positive', 'positive', 'neutral']
f1_demo  = macro_f1(y_true_f, y_pred_f)
print(f'macro_f1 demo : {f1_demo:.4f}   (expected ≈ 0.5556)')
assert 0.50 < f1_demo < 0.70

# ── Run the unit tests ────────────────────────────────────────────────────
result = subprocess.run(
    ['python3', '-m', 'pytest', 'tests/', '-v'],
    capture_output=True, text=True,
)
print(result.stdout)
```

---

## Section 5 — Reading `/metrics` for Derived Signals (10 min)

### What the `/metrics` endpoint provides

The service tracks the last 100 requests' response times and reports statistics at `GET /metrics`. The key number is `p95_latency_ms`.

**Why p95 instead of the average?** The average can hide slow outliers — a service could average 20ms while 5% of users wait 2 full seconds. p95 tells you "95% of requests were faster than this," which is the standard way production systems measure "how slow does it get."

`lib/metrics_reader.py` is provided for you — you use it, but don't need to write it. That mirrors how it works in the real world: one team builds the monitoring tools, another team uses them.

---

### `lib/metrics_reader.py`

```python
def fetch_metrics(base_url: str = 'http://localhost:8000') -> dict:
    response = httpx.get(f'{base_url}/metrics', timeout=5.0)
    response.raise_for_status()
    return response.json()
```
Calls the `/metrics` endpoint. `.raise_for_status()` throws a clear error if the server responds with a failure code, instead of silently continuing with bad data.

```python
def extract_p95(metrics: dict) -> Optional[float]:
    return metrics.get('p95_latency_ms')
```
`.get()` returns `None` if the key doesn't exist, instead of crashing. That matters because the server reports `null` when it hasn't handled any requests yet.

---

### Using the helper in the notebook

```python
metrics_raw = httpx.get(f'{BASE_URL}/metrics').json()
print(json.dumps(metrics_raw, indent=2))
```
Look at the raw data first, before using the helper functions — it's easier to trust a helper once you've seen what it's working with.

```python
from lib.metrics_reader import fetch_metrics, extract_p95

metrics = fetch_metrics(BASE_URL)
p95     = extract_p95(metrics)
```
Now use the two helper functions: one fetches the data, the other pulls out just the number we want.

```python
if p95 is not None:
    print(f'p95 latency : {p95:.1f} ms')
else:
    print('p95 not yet available — no requests have been served')
```
Check for the "no data yet" case before trying to print a number, and explain it in plain language instead of just printing `None`.

---

### Complete code — Section 5

```python
from lib.metrics_reader import fetch_metrics, extract_p95

# Raw view
metrics_raw = httpx.get(f'{BASE_URL}/metrics').json()
print('Raw /metrics payload:')
print(json.dumps(metrics_raw, indent=2))

# Helper view
metrics = fetch_metrics(BASE_URL)
p95     = extract_p95(metrics)

if p95 is not None:
    print(f'\np95 latency : {p95:.1f} ms')
    print('Interpretation: 95% of requests completed faster than this.')
else:
    print('\np95 not yet available.')
```

---

## Section 6 — Writing the Report: Surface the Failures (15 min)

### The two rules of honest evaluation reporting

**Rule 1 — Always show which items failed, not just the overall score.**

"Accuracy: 86.67%" tells you 13 of 15 reviews were correct, but not *which* two were wrong. Without knowing that, you can't dig into why the model failed or check whether a fix actually worked.

**Rule 2 — Don't let failures blend into a big table.**

A table with 13 checkmarks and 2 X's is easy to skim past. The report should call out the failed items separately, showing the true label, the wrong guess, and the actual text — side by side.

Both rules are handled by `eval_sentiment.py`'s `write_report` function.

---

### Running the production harness

```python
run = subprocess.run(
    ['python', 'eval_sentiment.py'],
    capture_output=True,
    text=True,
)
print(run.stdout)
```
Runs the script and waits for it to finish (unlike `Popen`, which we used to start the server without waiting). `capture_output=True` grabs its printed output; `text=True` gives us that output as normal text instead of raw bytes.

---

### Reading the generated report

```python
report_text = pathlib.Path('report.md').read_text()
print(report_text)
```
Reads the report file that the script just wrote, so you can see it right in the notebook. Open it in a Markdown viewer for a nicer view.

---

### The full inline eval and failure spotlight

```python
all_results = []

for review in reviews:
    resp      = httpx.post(f'{BASE_URL}/predict', json={'text': review['text']}, timeout=5.0)
    payload   = resp.json()
    label     = review['label']
    predicted = payload['sentiment']

    all_results.append({
        'id': review['id'], 'text': review['text'],
        'label': label, 'predicted': predicted,
        'correct': (label == predicted),
    })
```
Same pattern as Section 2, now run on all 15 reviews instead of 3. Doing it here in the notebook lets us work with the results directly as Python data.

```python
y_true    = [r['label']     for r in all_results]
y_pred    = [r['predicted'] for r in all_results]
acc       = accuracy(y_true, y_pred)
f1        = macro_f1(y_true, y_pred)
n_correct = sum(r['correct'] for r in all_results)
```
Pulls out the two lists the scoring functions need, then calls them. Because `accuracy` and `macro_f1` are pure functions, they don't care where this data came from.

```python
failures = [r for r in all_results if not r['correct']]
```
Filters down to just the wrong predictions.

```python
for item in failures:
    print(f'Review {item["id"]}')
    print(f'  True label : {item["label"]}')
    print(f'  Predicted  : {item["predicted"]}')
    print(f'  Text       : {item["text"]}')
```
Prints each failure with everything needed to understand it: the ID, the correct label, the model's guess, and the original text.

---

### What the two failures reveal

**Review 11**: `"Item arrived damaged, very frustrating."` → predicted `positive`.

The model seems to treat "arrived" as a positive signal and mishandles the "arrived...damaged" combination, so it guesses positive with only 0.61 confidence. This is a common problem with keyword-based models: they lose context when checking words individually.

**Review 13**: `"Would not recommend to anyone."` → predicted `positive`.

The phrase "would not recommend" confuses the model. A model that properly understood negation ("not recommend" = negative) would catch this — but this one doesn't. This is a classic NLP challenge known as the "negation problem."

Both failures are negative reviews the model mistakenly called positive. 86.67% accuracy alone doesn't sound alarming — but knowing exactly *which* reviews failed and *why* is what makes the evaluation useful.

---

### Complete code — Section 6

```python
# ── Run the production harness ────────────────────────────────────────────
run = subprocess.run(['python3', 'eval_sentiment.py'], capture_output=True, text=True)
print(run.stdout)
if run.returncode != 0:
    print('Error:', run.stderr)

# ── Display the report ────────────────────────────────────────────────────
print(pathlib.Path('report.md').read_text())

# ── Reproduce inline + surface failures ───────────────────────────────────
from lib.scorers import accuracy, macro_f1

all_results = []
for review in reviews:
    resp      = httpx.post(f'{BASE_URL}/predict', json={'text': review['text']}, timeout=5.0)
    payload   = resp.json()
    label, predicted = review['label'], payload['sentiment']
    all_results.append({
        'id': review['id'], 'text': review['text'],
        'label': label, 'predicted': predicted, 'correct': (label == predicted),
    })

y_true = [r['label']     for r in all_results]
y_pred = [r['predicted'] for r in all_results]
acc    = accuracy(y_true, y_pred)
f1     = macro_f1(y_true, y_pred)
n_ok   = sum(r['correct'] for r in all_results)

print(f'Accuracy  : {acc:.2%}   ({n_ok}/{len(all_results)} correct)')
print(f'Macro F1  : {f1:.4f}')

failures = [r for r in all_results if not r['correct']]
print(f'\n=== {len(failures)} FAILURES ===\n')
for item in failures:
    print(f'Review {item["id"]}')
    print(f'  True label : {item["label"]}')
    print(f'  Predicted  : {item["predicted"]}')
    print(f'  Text       : {item["text"]}')
    print()

# ── Shut down the server ──────────────────────────────────────────────────
server.terminate()
time.sleep(0.5)
print(f'Server (PID {server.pid}) stopped.')
```

---

## File Reference

| File | Role |
|------|------|
| `Module_11_Lab.ipynb` | Interactive notebook — run cells top to bottom |
| `eval_sentiment.py` | Production harness — run as `python eval_sentiment.py` |
| `sentiment-api/app.py` | Pre-shipped FastAPI service (do not modify) |
| `data/sentiment_demo.json` | 15 labeled reviews (the ground truth) |
| `lib/scorers.py` | Pure scoring functions: `accuracy`, `macro_f1` |
| `lib/metrics_reader.py` | Pre-shipped metrics helper: `fetch_metrics`, `extract_p95` |
| `tests/test_scorers.py` | Unit tests for `lib/scorers.py` |
| `report.md` | Generated output — created when you run the harness |

---

## Key Principles Embedded in This Lab

| Principle | Where it lands |
|-----------|---------------|
| Harness = script, not notebook | `eval_sentiment.py` vs. `Module_11_Lab.ipynb` |
| Methodology paragraph in three places | Script docstring · spec · guide |
| Scorers as pure functions | `lib/scorers.py` — no I/O, testable with synthetics |
| `compute = lib/; harness = thin coordinator` | `eval_sentiment.py` calls `lib/`, does not embed math |
| Derived signals from `/metrics` | `lib/metrics_reader.py` — pre-shipped, consumed not written |
| Failures surfaced explicitly | `## Failures` section in `report.md` |
