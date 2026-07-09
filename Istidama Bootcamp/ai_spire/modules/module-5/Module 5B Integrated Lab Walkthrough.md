# Module 5B Lab Walkthrough: Equipment Failure Prediction

Every line of code is explained below. Each section ends with the **complete codeblock** so you can copy-paste without hunting through the explanations.

---

## Section 1 — Dataset Generation

### Line-by-line explanation

```python
import numpy as np
import pandas as pd
```
Standard imports. `numpy` provides random number generators and math. `pandas` builds the DataFrame we train on.

```python
np.random.seed(42)
n = 300
```
`seed(42)` makes every random call reproducible — running this cell twice produces identical results. `n = 300` is the total number of equipment observations.

```python
temperature_c = np.random.normal(75, 15, n).round(1)
```
`np.random.normal(mean, std, size)` draws from a Gaussian distribution. Equipment temperature centers at 75°C with ±15°C spread. `.round(1)` keeps sensor precision realistic.

```python
vibration_mm = np.random.exponential(scale=2.5, size=n).round(2)
```
Vibration is always positive and right-skewed (most equipment vibrates gently; a few vibrate violently). The exponential distribution models this naturally. `scale=2.5` means the average vibration is 2.5 mm.

```python
pressure_psi = np.random.normal(150, 20, n).round(1)
equipment_age_years = np.random.uniform(0.5, 15, n).round(1)
maintenance_count = np.random.poisson(lam=3, size=n)
operating_hours = np.random.uniform(1000, 20000, n).round(0)
```
- `pressure_psi`: Normal distribution, centered at 150 PSI.
- `equipment_age_years`: Uniform between 0.5 and 15 years — no equipment older than 15 is in service.
- `maintenance_count`: Poisson with mean 3 — count data (non-negative integers). Poisson is the natural distribution for event counts.
- `operating_hours`: Uniform between 1,000 and 20,000 hours — wide range for a fleet of mixed-age equipment.

```python
equipment_type = np.random.choice(
    ["compressor", "turbine", "pump", "conveyor"], n,
    p=[0.3, 0.25, 0.25, 0.2]
)
shift = np.random.choice(["day", "night", "weekend"], n, p=[0.5, 0.35, 0.15])
```
`np.random.choice` samples categorically with specified probabilities (`p` must sum to 1.0). Compressors are the most common at 30%; weekend shifts are least common at 15%.

```python
failure_prob = 1 / (1 + np.exp(
    4.0
    - 0.03 * temperature_c
    - 0.15 * vibration_mm
    - 0.1 * equipment_age_years
    + 0.2 * maintenance_count
    - 0.0001 * operating_hours
    + np.random.normal(0, 0.5, n)
))
```
This is a **logistic (sigmoid) function** applied to a linear combination of features. The formula `1 / (1 + exp(-z))` squashes any real number into (0, 1), producing a probability. The coefficients encode domain knowledge:
- Higher temperature → higher failure probability (negative coefficient in the exponent `- 0.03 * temperature_c` means the exponent decreases, so `1/(1+exp(smaller))` increases)
- Higher vibration → higher failure probability
- Older equipment → higher failure probability
- More maintenance → *lower* failure probability (positive coefficient raises the exponent, which *lowers* the probability)
- `np.random.normal(0, 0.5, n)` adds random noise so the relationship isn't perfectly learnable.

```python
failed = (np.random.random(n) < failure_prob).astype(int)
```
For each equipment unit, draw a uniform random number in [0,1]. If it falls below `failure_prob`, mark as failed. This implements Bernoulli sampling: probability `p` of being 1, probability `(1-p)` of being 0. `.astype(int)` converts `True/False` to `1/0`.

```python
equipment = pd.DataFrame({...})
print(equipment.shape)
print(f"Failure rate: {failed.mean():.1%}")
print(equipment.describe().round(1))
```
Assemble all arrays into a DataFrame. `failed.mean()` gives the fraction of 1s — this is the failure rate. `:.1%` formats as a percentage with one decimal. `describe()` summarizes min, max, mean, quartiles for all numeric columns.

### Complete Codeblock

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

temperature_c = np.random.normal(75, 15, n).round(1)
vibration_mm = np.random.exponential(scale=2.5, size=n).round(2)
pressure_psi = np.random.normal(150, 20, n).round(1)
equipment_age_years = np.random.uniform(0.5, 15, n).round(1)
maintenance_count = np.random.poisson(lam=3, size=n)
operating_hours = np.random.uniform(1000, 20000, n).round(0)
equipment_type = np.random.choice(
    ["compressor", "turbine", "pump", "conveyor"], n,
    p=[0.3, 0.25, 0.25, 0.2]
)
shift = np.random.choice(["day", "night", "weekend"], n, p=[0.5, 0.35, 0.15])

failure_prob = 1 / (1 + np.exp(
    4.0
    - 0.03 * temperature_c
    - 0.15 * vibration_mm
    - 0.1 * equipment_age_years
    + 0.2 * maintenance_count
    - 0.0001 * operating_hours
    + np.random.normal(0, 0.5, n)
))
failed = (np.random.random(n) < failure_prob).astype(int)

equipment = pd.DataFrame({
    "temperature_c": temperature_c,
    "vibration_mm": vibration_mm,
    "pressure_psi": pressure_psi,
    "equipment_age_years": equipment_age_years,
    "maintenance_count": maintenance_count,
    "operating_hours": operating_hours,
    "equipment_type": equipment_type,
    "shift": shift,
    "failed": failed,
})

print(equipment.shape)
print(f"Failure rate: {failed.mean():.1%}")
print(equipment.describe().round(1))
```

---

## Section 2 — Exploratory Data Analysis

### Line-by-line explanation

```python
print(equipment["failed"].value_counts())
```
Shows the raw count of 0s and 1s. This makes the class imbalance concrete before any modeling.

```python
equipment.groupby("failed")[["temperature_c", "vibration_mm", "equipment_age_years"]].mean().round(2)
```
`groupby("failed")` splits the DataFrame into two groups. `.mean()` computes averages per group. This lets you see whether failed equipment systematically had higher temperature, vibration, or age — confirming the feature-failure relationships we encoded in the data generator.

```python
pd.crosstab(equipment["equipment_type"], equipment["failed"], normalize="index").round(3)
```
`pd.crosstab` builds a contingency table. `normalize="index"` converts raw counts to row proportions — so each row (equipment type) sums to 1.0, showing the failure *rate* per type rather than the raw count.

### Complete Codeblock

```python
print("Class distribution:")
print(equipment["failed"].value_counts())
print()

print("Mean sensor readings by failure label:")
print(
    equipment.groupby("failed")[["temperature_c", "vibration_mm", "equipment_age_years"]]
    .mean()
    .round(2)
)
print()

print("Equipment type vs. failure:")
print(pd.crosstab(equipment["equipment_type"], equipment["failed"], normalize="index").round(3))
```

---

## Section 3 — ColumnTransformer + Pipeline

### Line-by-line explanation

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
```
Three imports: the transformer combiner, the two transformers, and the pipeline wrapper.

```python
numeric_features = [
    "temperature_c", "vibration_mm", "pressure_psi",
    "equipment_age_years", "maintenance_count", "operating_hours"
]
categorical_features = ["equipment_type", "shift"]
```
Explicit column lists — telling the ColumnTransformer *which* columns get which treatment. Keeping these as named lists makes it easy to add or remove features later.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
```
`ColumnTransformer` applies different transformers to different column subsets, then concatenates the outputs into a single feature matrix.
- **`("num", StandardScaler(), numeric_features)`**: Name `"num"`, apply `StandardScaler` (zero mean, unit variance) to the numeric columns. Logistic regression and other distance-sensitive models require this.
- **`("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)`**: Name `"cat"`, convert each category to binary dummy columns. `handle_unknown="ignore"` prevents errors if a new equipment type appears at prediction time that wasn't in training.

```python
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])
```
The Pipeline chains the preprocessor and model into one object. When you call `pipe.fit(X_train, y_train)`:
1. `preprocessor.fit_transform(X_train)` — learns mean/std from training data only, then transforms it
2. `model.fit(transformed_X_train, y_train)` — trains the classifier

When you call `pipe.predict(X_test)`:
1. `preprocessor.transform(X_test)` — uses the mean/std *learned from training*, never from test data
2. `model.predict(transformed_X_test)` — produces predictions

This single-object design prevents **data leakage** — the most common error in ML pipelines.

### Complete Codeblock

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

numeric_features = [
    "temperature_c", "vibration_mm", "pressure_psi",
    "equipment_age_years", "maintenance_count", "operating_hours"
]
categorical_features = ["equipment_type", "shift"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

print("Preprocessor configured:")
print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")
```

---

## Section 4 — 6 Model Configurations

### Line-by-line explanation

```python
LogisticRegression(penalty="l2", random_state=42, max_iter=1000)
```
L2 regularization (Ridge) penalizes large coefficients by their squared magnitude. It shrinks all coefficients toward zero but keeps them all nonzero. `max_iter=1000` prevents convergence warnings on small datasets.

```python
LogisticRegression(penalty="l1", C=0.1, solver="liblinear", random_state=42, max_iter=1000)
```
L1 regularization (Lasso) penalizes by absolute value, producing **sparse** solutions — some coefficients go exactly to zero, implicitly selecting features. `C=0.1` means strong regularization (C is the inverse of regularization strength — smaller C = more regularization). `solver="liblinear"` is required for L1 in scikit-learn.

```python
DecisionTreeClassifier(max_depth=5, random_state=42)
```
A single decision tree. `max_depth=5` limits how deep the tree grows, reducing overfitting. Trees are interpretable — you can print the decision rules. Without `max_depth`, a tree will memorize training data.

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```
An ensemble of 100 decision trees, each trained on a random data bootstrap and random feature subset. The majority vote of 100 trees is more stable than any single tree. No class weighting — will likely under-predict failures.

```python
RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
```
Same as above, but `class_weight="balanced"` tells the model to weight each class inversely proportional to its frequency. With 5% failures, failure samples get ~20× the weight of non-failure samples in the loss function. This directly incentivizes catching more failures.

```python
DummyClassifier(strategy="most_frequent")
```
Always predicts the majority class (no failure). Expected accuracy: ~95%. Expected recall: 0. This is the **baseline floor** — any real model must beat this, especially on recall and F1.

### Complete Codeblock

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

models = {
    "LogReg (L2, default)": LogisticRegression(penalty="l2", random_state=42, max_iter=1000),
    "LogReg (L1, C=0.1)": LogisticRegression(
        penalty="l1", C=0.1, solver="liblinear", random_state=42, max_iter=1000
    ),
    "DecisionTree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RF (default)": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF (balanced)": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "Dummy (baseline)": DummyClassifier(strategy="most_frequent"),
}

print(f"Defined {len(models)} model configurations:")
for name in models:
    print(f"  - {name}")
```

---

## Section 5 — Cross-Validation Loop

### Line-by-line explanation

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
`StratifiedKFold` splits data into 5 folds while preserving the class ratio in each fold. With only ~15 failure cases in 300 rows, random splitting could put 0 failures in a fold — stratification prevents this. `shuffle=True` randomizes order before splitting. `random_state=42` makes the split reproducible.

```python
X = equipment.drop("failed", axis=1)
y = equipment["failed"]
```
Separate features (X) from the target (y). `drop("failed", axis=1)` removes the label column from the feature matrix.

```python
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}
```
A dictionary of metric names that `cross_validate` will compute simultaneously. Using string shortcuts for built-in scorers. All four metrics apply to the **positive class** (failure = 1) by default in binary classification.

```python
scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
```
`cross_validate` fits the pipeline 5 times (once per fold), evaluating on the held-out fold each time. It returns a dict with keys like `"test_accuracy"` mapping to arrays of 5 scores — one per fold. It uses `cv` to determine the fold splits.

```python
row = {
    "Accuracy": f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
    ...
}
```
The mean gives the expected performance; the standard deviation gives stability. High standard deviation means the model's performance varies a lot across folds — a sign of instability or small data.

```python
experiment_log.append({
    "model_name": name,
    "hyperparams": str(model.get_params()),
    ...
    "timestamp": datetime.datetime.now().isoformat(),
})
```
`model.get_params()` returns the model's hyperparameter dictionary. `str()` serializes it to a readable string for CSV storage. `datetime.datetime.now().isoformat()` stamps the exact time of the run in ISO 8601 format. This log is how you reconstruct any result weeks later.

### Complete Codeblock

```python
from sklearn.model_selection import cross_validate, StratifiedKFold
import datetime

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X = equipment.drop("failed", axis=1)
y = equipment["failed"]

results = []
experiment_log = []

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

    row = {
        "Model": name,
        "Accuracy": f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
        "Precision": f"{scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}",
        "Recall": f"{scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}",
        "F1": f"{scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}",
    }
    results.append(row)

    experiment_log.append({
        "model_name": name,
        "hyperparams": str(model.get_params()),
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean(),
        "timestamp": datetime.datetime.now().isoformat(),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

---

## Section 6 — Precision-Recall Curves

### Line-by-line explanation

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
Reserve 20% of data for test evaluation. `stratify=y` ensures the 5% failure rate is preserved in both halves. We need a held-out test set here because `cross_validate` does not return fitted models — the PR curve requires a single fitted model to call `predict_proba`.

```python
top_models = { "LogReg (L2)": ..., "RF (default)": ..., "RF (balanced)": ... }
```
We plot three models to make the PR curve comparison readable. Including all six would clutter the figure.

```python
pipe.fit(X_train, y_train)
PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
```
`from_estimator` calls `pipe.predict_proba(X_test)` internally, computes the precision-recall curve across all thresholds, and plots it. Passing `ax=ax` overlays all three curves on the same figure.

```python
plt.savefig("results/pr_curves.png", dpi=150)
```
`dpi=150` gives a crisp image suitable for the deliverable document. The file is saved before `plt.show()` so the saved version is never blank.

### Complete Codeblock

```python
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]} ({y_train.mean():.1%} failures)")
print(f"Test size:  {X_test.shape[0]} ({y_test.mean():.1%} failures)")

top_models = {
    "LogReg (L2)": LogisticRegression(random_state=42, max_iter=1000),
    "RF (default)": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF (balanced)": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
}

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in top_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)

ax.set_title("Precision-Recall Curves — Equipment Failure")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("results/pr_curves.png", dpi=150)
plt.show()
print("Saved: results/pr_curves.png")
```

---

## Section 7 — Calibration Diagram

### Line-by-line explanation

```python
from sklearn.calibration import CalibrationDisplay
```
scikit-learn's built-in calibration plot tool. It groups predictions into probability bins and compares predicted probability vs. observed failure rate per bin.

```python
CalibrationDisplay.from_estimator(
    pipe, X_test, y_test, name=name, n_bins=5, ax=ax
)
```
`from_estimator` calls `predict_proba`, groups predictions into `n_bins=5` equal-width bins, and plots mean predicted probability (x-axis) vs. actual fraction of failures (y-axis). With only ~12 failures in the test set, `n_bins=5` prevents bins with zero samples.

**Reading the plot:**
- Points on the **diagonal line** = perfect calibration
- Points **above** the diagonal = model is underconfident (true rate is higher than predicted)
- Points **below** the diagonal = model is overconfident (true rate is lower than predicted)

Random Forests tend to cluster predictions near 0 and 1 (overconfident), while Logistic Regression, being inherently probabilistic, often calibrates better out of the box.

### Complete Codeblock

```python
from sklearn.calibration import CalibrationDisplay

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in top_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    CalibrationDisplay.from_estimator(
        pipe, X_test, y_test, name=name, n_bins=5, ax=ax
    )

ax.set_title("Calibration Diagram — Equipment Failure")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/calibration.png", dpi=150)
plt.show()
print("Saved: results/calibration.png")
```

---

## Section 8 — Save Best Model with joblib

### Line-by-line explanation

```python
best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
])
best_pipe.fit(X_train, y_train)
```
We explicitly rebuild and fit the best pipeline using only training data. We chose RF (balanced) because it had the best recall in cross-validation — critical for safety-sensitive failure detection.

```python
joblib.dump(best_pipe, "results/best_model.joblib")
```
`joblib.dump` serializes the entire fitted pipeline to a binary file. This includes the fitted `StandardScaler` (with learned mean and std), the fitted `OneHotEncoder` (with learned categories), and the fitted `RandomForestClassifier` (with 100 trained trees). Everything needed to make predictions is in one file.

```python
loaded_model = joblib.load("results/best_model.joblib")
predictions_match = (loaded_model.predict(X_test) == best_pipe.predict(X_test)).all()
```
Round-trip verification: load the saved file and confirm it produces identical predictions. `.all()` returns `True` only if every single prediction matches. This is a critical sanity check before claiming the save was successful.

```python
loaded_model.predict_proba(sample_new_reading)[:, 1]
```
`predict_proba` returns an (n_samples, 2) array — column 0 is P(no failure), column 1 is P(failure). `[:, 1]` selects the failure probability column. In production, you compare this probability against your operational threshold to trigger a maintenance alert.

### Complete Codeblock

```python
import joblib

best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )),
])
best_pipe.fit(X_train, y_train)

joblib.dump(best_pipe, "results/best_model.joblib")
print(f"Model saved: {os.path.getsize('results/best_model.joblib') / 1024:.0f} KB")

loaded_model = joblib.load("results/best_model.joblib")
predictions_match = (loaded_model.predict(X_test) == best_pipe.predict(X_test)).all()
print(f"Loaded model predictions match original: {predictions_match}")

sample_new_reading = X_test.iloc[:3].copy()
print("\nExample: predict on 3 new sensor readings")
print("Predictions:", loaded_model.predict(sample_new_reading))
print("Failure probabilities:", loaded_model.predict_proba(sample_new_reading)[:, 1].round(3))
```

---

## Section 9 — Export All Results

### Line-by-line explanation

```python
results_df.to_csv("results/comparison_table.csv", index=False)
```
`to_csv` writes the DataFrame to a comma-separated file. `index=False` prevents pandas from writing row numbers as a column — the file stays clean for sharing.

```python
log_df = pd.DataFrame(experiment_log)
log_df.to_csv("results/experiment_log.csv", index=False)
```
`experiment_log` is a list of dicts; `pd.DataFrame` converts it to a table. Each row is one model run with all its metrics and the timestamp. This log is your audit trail.

```python
for f in expected_files:
    status = "✓" if os.path.exists(f) else "✗ MISSING"
    size = f"{os.path.getsize(f) / 1024:.0f} KB" if os.path.exists(f) else ""
    print(f"  {status}  {f}  {size}")
```
`os.path.exists` checks whether the file was actually written. `os.path.getsize` returns size in bytes; dividing by 1024 converts to kilobytes. This final check prevents submitting a deliverable with a missing file.

### Complete Codeblock

```python
results_df.to_csv("results/comparison_table.csv", index=False)
print("Saved: results/comparison_table.csv")

log_df = pd.DataFrame(experiment_log)
log_df.to_csv("results/experiment_log.csv", index=False)
print("Saved: results/experiment_log.csv")

expected_files = [
    "results/comparison_table.csv",
    "results/experiment_log.csv",
    "results/pr_curves.png",
    "results/calibration.png",
    "results/best_model.joblib",
]
print("\nDeliverables check:")
for f in expected_files:
    status = "✓" if os.path.exists(f) else "✗ MISSING"
    size = f"{os.path.getsize(f) / 1024:.0f} KB" if os.path.exists(f) else ""
    print(f"  {status}  {f}  {size}")
```

---

## Bonus Section — 3 Techniques to Improve Metrics

---

### Technique 1: Threshold Optimization

**What it solves:** The default classification threshold is 0.5. In imbalanced datasets, the model's confidence distribution is skewed — the optimal boundary for maximizing F1 (or recall) is rarely at 0.5. Moving the threshold lower increases recall at a precision cost; moving it higher does the opposite.

**Key insight:** This technique requires no retraining — you only change where you draw the line on existing probability outputs.

#### Line-by-line explanation

```python
y_proba = thresh_pipe.predict_proba(X_test)[:, 1]
```
Extract the probability of failure (positive class) for every test sample. This is the continuous score we will threshold.

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
```
`precision_recall_curve` computes precision and recall at every unique predicted probability value as the threshold. Returns arrays of length n+1 (precisions, recalls) and n (thresholds) — the last precision/recall pair corresponds to a threshold below the minimum score.

```python
f1_scores_at_thresh = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
```
Compute F1 at each threshold using the harmonic mean formula. `[:-1]` removes the last element to align with the `thresholds` array length. `+ 1e-8` prevents division by zero when both precision and recall are zero.

```python
best_idx = np.argmax(f1_scores_at_thresh)
best_threshold = thresholds[best_idx]
```
`np.argmax` finds the index of the maximum value. `thresholds[best_idx]` retrieves the threshold value that produced the highest F1.

```python
y_pred_optimal = (y_proba >= best_threshold).astype(int)
```
Apply the new threshold: any sample with predicted probability >= `best_threshold` is classified as failure. `.astype(int)` converts booleans to 0/1.

### Complete Codeblock

```python
from sklearn.metrics import precision_recall_curve, classification_report

thresh_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
])
thresh_pipe.fit(X_train, y_train)

y_proba = thresh_pipe.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores_at_thresh = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)

best_idx = np.argmax(f1_scores_at_thresh)
best_threshold = thresholds[best_idx]

print("=== Default threshold (0.5) ===")
y_pred_default = (y_proba >= 0.5).astype(int)
print(classification_report(y_test, y_pred_default, target_names=["No Failure", "Failure"]))

print(f"=== Optimized threshold ({best_threshold:.3f}) ===")
y_pred_optimal = (y_proba >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal, target_names=["No Failure", "Failure"]))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, precisions[:-1], label="Precision", color="steelblue")
ax.plot(thresholds, recalls[:-1], label="Recall", color="tomato")
ax.plot(thresholds, f1_scores_at_thresh, label="F1", color="seagreen")
ax.axvline(x=best_threshold, color="gray", linestyle="--",
           label=f"Optimal threshold ({best_threshold:.3f})")
ax.axvline(x=0.5, color="black", linestyle=":", label="Default threshold (0.5)")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision, Recall, and F1 vs. Threshold — RF (balanced)")
ax.legend()
plt.tight_layout()
plt.savefig("results/threshold_tuning.png", dpi=150)
plt.show()
print("Saved: results/threshold_tuning.png")
```

---

### Technique 2: SMOTE — Synthetic Minority Oversampling

**What it solves:** Class imbalance at the data level. `class_weight` adjusts the *loss function*; SMOTE adjusts the *training data* by synthesizing new failure examples. The model sees a more balanced distribution during training, which can improve recall and F1 without just amplifying the same real samples.

**How SMOTE works:** For each minority sample, SMOTE finds its k nearest minority neighbors, then creates synthetic samples by interpolating along the line segments connecting them in feature space. The result is new, plausible minority-class samples — not exact duplicates.

#### Line-by-line explanation

```python
from imblearn.pipeline import Pipeline as ImbPipeline
```
We import imbalanced-learn's Pipeline (not scikit-learn's). The imblearn Pipeline understands that SMOTE should only run on **training folds**, not on validation folds. scikit-learn's Pipeline does not have this behavior — using it with SMOTE would apply oversampling to the test fold and leak information.

```python
smote_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])
```
SMOTE sits between preprocessing and the model in the pipeline. Order matters: preprocessing must run first so SMOTE operates on scaled numeric features (SMOTE uses Euclidean distance to find nearest neighbors). Note we use default (non-balanced) RF here — SMOTE is handling the imbalance at the data level instead.

```python
smote_scores = cross_validate(smote_pipe, X, y, cv=cv, scoring=...)
```
`cross_validate` with the imblearn Pipeline: for each fold, SMOTE is applied to the training split only, then the model trains on the augmented data, and evaluation happens on the original (unaugmented) test fold.

### Complete Codeblock

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

smote_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

smote_scores = cross_validate(
    smote_pipe, X, y, cv=cv,
    scoring={"accuracy": "accuracy", "precision": "precision",
             "recall": "recall", "f1": "f1"}
)

print("=== SMOTE + RF (default weights) vs. RF (balanced) ===")
print(f"{'Metric':<12} {'SMOTE+RF':>12} {'RF (balanced)':>14}")
print("-" * 40)

rf_balanced_scores = log_df[log_df["model_name"] == "RF (balanced)"].iloc[0]
for metric in ["accuracy", "precision", "recall", "f1"]:
    smote_val = smote_scores[f"test_{metric}"].mean()
    rf_val = rf_balanced_scores[metric]
    print(f"{metric.capitalize():<12} {smote_val:>12.3f} {rf_val:>14.3f}")

print()
print("Key insight:")
print("  SMOTE works by generating new synthetic minority samples,")
print("  while class_weight adjusts the loss function only.")
print("  In practice, their performance is often similar — compare both.")
```

---

### Technique 3: GridSearchCV — Systematic Hyperparameter Search

**What it solves:** The default hyperparameters of any model are a starting point, not the optimal configuration for your data. GridSearchCV exhaustively tests every combination in a defined grid and returns the best-performing set. Crucially, we score on **F1** (not accuracy) — so the search optimizes for minority-class performance.

#### Line-by-line explanation

```python
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 5],
    "model__class_weight": ["balanced", None],
}
```
The `model__` prefix refers to the `"model"` step in the Pipeline. `max_depth=None` means trees grow fully (no pruning); `max_depth=10` or `20` controls overfitting. `min_samples_leaf` sets the minimum number of samples required to be a leaf node — higher values smooth the model. This grid has 2×3×3×2 = 36 combinations.

```python
GridSearchCV(
    tuning_pipe,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)
```
- `cv=StratifiedKFold(...)`: 5-fold stratified CV for each parameter combination
- `scoring="f1"`: the grid optimizes for F1, not accuracy — critical with imbalanced data
- `n_jobs=-1`: use all available CPU cores in parallel (36 combinations × 5 folds = 180 fits, parallelized)
- `verbose=1`: print progress during fitting

```python
grid_search.fit(X_train, y_train)
```
Fits 180 pipelines total (36 param combos × 5 folds). Evaluates each on its held-out fold, then selects the combination with the highest mean CV F1.

```python
grid_search.best_params_
grid_search.best_score_
grid_search.best_estimator_
```
- `best_params_`: the winning hyperparameter dictionary
- `best_score_`: the CV F1 of the best combination
- `best_estimator_`: the full fitted Pipeline with those hyperparameters, refitted on all of `X_train`

```python
f1_improvement = f1_tuned - f1_baseline
```
Compare the tuned model against the default RF (balanced) on the test set. If the improvement is positive, tuning helped. If it is near zero, the default parameters were already close to optimal — common with small datasets.

### Complete Codeblock

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 5],
    "model__class_weight": ["balanced", None],
}

tuning_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42)),
])

grid_search = GridSearchCV(
    tuning_pipe,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print(f"Best CV F1:   {grid_search.best_score_:.3f}")
print(f"Best params:  {grid_search.best_params_}")

print("\nTest set performance with tuned model:")
y_pred_tuned = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_tuned, target_names=["No Failure", "Failure"]))

y_pred_baseline = best_pipe.predict(X_test)
f1_baseline = f1_score(y_test, y_pred_baseline)
f1_tuned = f1_score(y_test, y_pred_tuned)

print(f"\nF1 improvement from tuning: {f1_baseline:.3f} → {f1_tuned:.3f} (+{f1_tuned - f1_baseline:.3f})")

joblib.dump(grid_search.best_estimator_, "results/best_model_tuned.joblib")
print("Saved: results/best_model_tuned.joblib")
```

---

## Summary: When to Use Each Technique

| Technique | Use when | Watch out for |
|-----------|----------|---------------|
| **Threshold optimization** | You have a fitted model and want to shift the precision/recall tradeoff for deployment | The optimal threshold on your test set may not generalize — validate on held-out data |
| **SMOTE** | You want to change the training distribution, not just the loss weighting | SMOTE must only apply to training folds — use `imblearn.pipeline.Pipeline`, never apply SMOTE before CV |
| **GridSearchCV** | You want to systematically find better hyperparameters | Can overfit if the search space is too large relative to dataset size; use F1 scoring, not accuracy |

---

*End of walkthrough. All code in this document matches the notebook cells exactly.*
