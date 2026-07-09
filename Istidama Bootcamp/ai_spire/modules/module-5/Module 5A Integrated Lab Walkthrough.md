# Integration 5 — Code Walkthrough
## ML Evaluation Pipeline | Module 5 Integrated Lab

This document explains every code cell in `integration_5a_lab.ipynb` line by line, followed by the complete code block for that section.

---

## Part 0 — Imports

### Line-by-line explanation

```python
import numpy as np
import pandas as pd
```
`numpy` provides array math and random number generation. `pandas` provides the `DataFrame` structure used to hold and inspect the dataset.

```python
import matplotlib.pyplot as plt
```
`matplotlib.pyplot` is the plotting interface used for the Precision-Recall and Calibration visualizations. `plt.subplots` creates figure/axes objects; `plt.savefig` writes the output to disk.

```python
import joblib
```
`joblib` serializes Python objects — specifically fitted scikit-learn pipelines — to disk. `joblib.dump` saves, `joblib.load` restores. This is how you persist a trained model so it can be deployed without retraining.

```python
import os
import datetime
```
`os` is used for `os.makedirs` (creating the `results/` output directory) and `os.path.getsize` (confirming the saved model file exists and has content). `datetime` provides `datetime.datetime.now().isoformat()` to timestamp each experiment log entry.

```python
import warnings
warnings.filterwarnings('ignore')
```
Suppresses convergence and precision-score warnings that appear when a model never predicts the positive class. These are expected with severely imbalanced data and would obscure the output without providing actionable information at this stage.

```python
from sklearn.compose import ColumnTransformer
```
`ColumnTransformer` applies different preprocessing steps to different column subsets and concatenates the results into a single transformed feature matrix. It is the standard scikit-learn pattern for datasets with mixed numeric and categorical features.

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```
`StandardScaler` centers each numeric feature to zero mean and unit variance — essential because features like `temperature_c` (range ~30–120) and `maintenance_count` (range 0–10) are on very different scales. `OneHotEncoder` converts categorical string values (e.g., `"compressor"`) into binary indicator columns.

```python
from sklearn.pipeline import Pipeline
```
`Pipeline` chains a preprocessor and a model into a single object. When passed to `cross_validate`, preprocessing steps are fit only on training folds and applied to test folds, preventing data leakage.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
```
The four model families used in the evaluation. `LogisticRegression` is the linear baseline (with L2 and L1 variants). `RandomForestClassifier` is an ensemble of decision trees (with default and class-weighted variants). `DecisionTreeClassifier` is a single tree with depth constraint. `DummyClassifier` ignores features entirely and always predicts the majority class — it is the sanity-check baseline.

```python
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, train_test_split
)
```
`StratifiedKFold` creates k folds that preserve the class ratio in each fold — critical at a 5% failure rate where a random split could put nearly all failures into one fold. `cross_validate` runs the CV loop and returns per-fold scores for multiple metrics simultaneously. `train_test_split` creates the held-out set needed for PR curves and calibration diagrams (since `cross_validate` does not return fitted models).

```python
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
```
`PrecisionRecallDisplay.from_estimator` plots a PR curve for a fitted pipeline on held-out data. `CalibrationDisplay.from_estimator` plots a reliability diagram showing whether the model's predicted probabilities match observed failure rates.

### Complete code block

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, train_test_split
)
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
```

---

## Part 1 — Dataset Generation

### Line-by-line explanation

```python
np.random.seed(42)
n = 300
```
Seeds NumPy's global random generator so that every stochastic call below produces identical results across runs. `n = 300` sets the total number of equipment records. 300 rows at a ~5% failure rate yields roughly 15 positive-class samples — a realistic and deliberately challenging class imbalance.

```python
temperature_c = np.random.normal(75, 15, n).round(1)
```
Samples operating temperature from a normal distribution with mean 75°C and standard deviation 15°C. `.round(1)` rounds to one decimal place to simulate sensor precision. Higher temperatures contribute to failure probability via the logistic formula below.

```python
vibration_mm = np.random.exponential(scale=2.5, size=n).round(2)
```
Samples vibration level from an exponential distribution. The exponential is appropriate here because vibration is bounded below by zero and has a long right tail — most equipment vibrates at low levels, but a few have extreme readings. `scale=2.5` sets the mean of the distribution.

```python
pressure_psi = np.random.normal(150, 20, n).round(1)
```
Samples operating pressure from a normal distribution (mean 150 psi, std 20 psi). Pressure is included as a feature but is **not** used in the failure probability formula — it is a noise feature that tests whether the model can ignore irrelevant inputs.

```python
equipment_age_years = np.random.uniform(0.5, 15, n).round(1)
```
Samples equipment age uniformly between 6 months and 15 years. `np.random.uniform` is appropriate because the plant operates equipment across the full age range without a preferred distribution.

```python
maintenance_count = np.random.poisson(lam=3, size=n)
```
Samples the number of maintenance services from a Poisson distribution with mean 3. The Poisson is the natural distribution for count data with no upper bound. Higher maintenance counts *decrease* failure probability — equipment that is serviced more often is less likely to fail.

```python
operating_hours = np.random.uniform(1000, 20000, n).round(0)
```
Samples total operating hours uniformly between 1,000 and 20,000. Rounded to whole hours. High operating hours slightly increase failure probability (wear and fatigue).

```python
equipment_type = np.random.choice(
    ["compressor", "turbine", "pump", "conveyor"], n,
    p=[0.3, 0.25, 0.25, 0.2]
)
```
Assigns a categorical equipment type. The probability vector `p=[0.3, 0.25, 0.25, 0.2]` gives compressors the highest frequency (30%), reflecting their prevalence in the plant. This categorical feature will be one-hot encoded by the `ColumnTransformer`.

```python
shift = np.random.choice(["day", "night", "weekend"], n, p=[0.5, 0.35, 0.15])
```
Assigns the operating shift. Day shifts are most common (50%), weekend shifts are rare (15%). Shift is also a categorical feature and a potential proxy for supervision and staffing levels.

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
Computes the failure probability for each row using a **logistic (sigmoid) function**. The formula encodes domain knowledge:
- High `temperature_c` → higher failure probability (negative sign in exponent reversal: `-0.03 * temperature_c` lowers the exponent denominator, raising the probability)
- High `vibration_mm` → higher failure probability
- Older equipment (`equipment_age_years`) → higher failure probability
- More maintenance (`maintenance_count`) → lower failure probability (`+0.2` increases the exponent, lowering the probability)
- More operating hours → slightly higher failure probability
- `np.random.normal(0, 0.5, n)` adds irreducible noise — not every high-vibration unit fails

The intercept `4.0` is chosen so that the resulting failure rate is approximately 5%, matching the domain constraint.

```python
failed = (np.random.random(n) < failure_prob).astype(int)
```
Converts probabilities to binary labels using Bernoulli sampling: draw a uniform random number for each row; if it is less than that row's failure probability, assign `failed = 1`. `.astype(int)` converts the boolean array to 0/1 integers.

```python
equipment = pd.DataFrame({
    "temperature_c": temperature_c,
    ...
    "failed": failed,
})
```
Assembles all features and the target into a single DataFrame with descriptive column names. The target column `"failed"` is appended last.

```python
print(equipment.shape)
print(f"Failure rate: {failed.mean():.1%}")
print(equipment.describe().round(1))
```
Sanity checks: confirm 300 rows and 9 columns, the failure rate is near 5%, and summary statistics for each numeric feature look reasonable.

### Complete code block

```python
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

# Failure driven by temperature, vibration, age, low maintenance
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

## Part 2 — Data Inspection

### Line-by-line explanation

```python
print(equipment.shape)
```
Confirms 300 rows and 9 columns (8 features + 1 target). This is the first sanity check after generating data.

```python
print(f"Failure rate: {equipment['failed'].mean():.1%}")
```
Computes the mean of the binary `failed` column, which equals the proportion of positive-class samples. At ~5%, this confirms the dataset is severely imbalanced. The `.1%` format specifier renders the value as a percentage with one decimal place (e.g., `5.0%`).

```python
print(equipment[["temperature_c", "vibration_mm", "equipment_age_years"]].describe().round(1))
```
Prints summary statistics for the three features most directly linked to failure in the data generation formula. This helps orient learners to the feature ranges before building the preprocessor — for example, knowing that `temperature_c` ranges roughly 30–120°C clarifies why `StandardScaler` is needed to normalize it alongside `maintenance_count` (range 0–9).

### Complete code block

```python
print(equipment.shape)
print(f"Failure rate: {equipment['failed'].mean():.1%}")
print(equipment[["temperature_c", "vibration_mm", "equipment_age_years"]].describe().round(1))
```

---

## Part 3 — ColumnTransformer + Pipeline

### Line-by-line explanation

```python
numeric_features = ["temperature_c", "vibration_mm", "pressure_psi",
                    "equipment_age_years", "maintenance_count", "operating_hours"]
categorical_features = ["equipment_type", "shift"]
```
Explicit feature type lists are required by `ColumnTransformer`. Declaring them as named variables at the top of the preprocessing section also documents which features receive which treatment. Notice that `"failed"` is absent from both lists — it will be separated out when building `X` and `y` in Part 5.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
```
`ColumnTransformer` takes a list of `(name, transformer, columns)` tuples:

- `("num", StandardScaler(), numeric_features)`: applies `StandardScaler` to all six numeric columns, producing z-scores with mean 0 and std 1.
- `("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)`: encodes `equipment_type` (4 categories → 4 binary columns) and `shift` (3 categories → 3 binary columns), for 7 new columns total.

`handle_unknown="ignore"` tells the encoder to produce an all-zeros row if it encounters a category during test/prediction that it did not see during training, rather than raising an error. This is essential for robustness in cross-validation when rare categories (like `"weekend"`) may be absent from some training folds.

The default `remainder="drop"` discards any columns not listed — there are none here since we list all 8 features, but the behavior is worth knowing.

After transformation, the numeric block (6 columns) and categorical block (7 columns) are **horizontally concatenated** into a 13-column feature matrix.

Note: unlike the Week A walkthrough, there is no `SimpleImputer` step here because the generation script produces no missing values. In a real dataset, you would add an imputer before the scaler and encoder, exactly as shown in the Week A lab.

### Complete code block

```python
numeric_features = ["temperature_c", "vibration_mm", "pressure_psi",
                    "equipment_age_years", "maintenance_count", "operating_hours"]
categorical_features = ["equipment_type", "shift"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

print("Preprocessor created.")
print(f"Numeric features   : {numeric_features}")
print(f"Categorical features: {categorical_features}")
```

---

## Part 4 — Model Configurations

### Line-by-line explanation

```python
models = {
    "LogReg (L2, default)": LogisticRegression(random_state=42),
```
Standard logistic regression with L2 (Ridge) regularization and `C=1.0`. The default solver (`lbfgs`) supports L2. `random_state=42` seeds the solver's initialization. This is the linear baseline that will be compared to the L1 variant and the tree-based models.

```python
    "LogReg (L1, C=0.1)": LogisticRegression(
        penalty="l1", C=0.1, solver="liblinear", random_state=42
    ),
```
Logistic regression with L1 (Lasso) regularization and a strong regularization strength (`C=0.1` — smaller C means stronger regularization). L1 penalization drives less-informative feature coefficients exactly to zero, performing automatic feature selection. `solver="liblinear"` is required because `lbfgs` does not support L1 penalty. Comparing this to `LogReg (L2, default)` isolates the effect of regularization type and strength.

```python
    "RF (default)": RandomForestClassifier(n_estimators=100, random_state=42),
```
A random forest with 100 trees and default settings, including equal class weights. On a 5% failure rate dataset, the default RF may under-predict failures because the majority class dominates the loss. This is the baseline tree ensemble.

```python
    "RF (balanced)": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
```
Identical to `RF (default)` except `class_weight="balanced"` automatically adjusts each tree's sample weights so that the minority class (failures) receives weight inversely proportional to its frequency — approximately 19× upweight at a 5% failure rate. This is the primary tool for improving recall on imbalanced datasets without modifying the data itself. Comparing this to `RF (default)` isolates the effect of class weighting.

```python
    "DecisionTree (max_depth=5)": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
```
A single decision tree limited to 5 levels of depth. `max_depth=5` is a regularization hyperparameter that prevents the tree from memorizing the training data. Unlike the random forest, a single tree is fully interpretable — you can visualize it with `sklearn.tree.plot_tree`. It typically underperforms the forest but is useful as a reference point between linear and ensemble models.

```python
    "Dummy (baseline)": DummyClassifier(strategy="most_frequent"),
```
Predicts the majority class (`failed=0`) for every row, ignoring all features. At a 5% failure rate, this achieves ~95% accuracy, zero recall, zero precision, and zero F1. Its presence in the table proves that high accuracy is meaningless when the failure class is rare. Any useful model must beat the Dummy on recall and F1.

### Complete code block

```python
models = {
    "LogReg (L2, default)": LogisticRegression(random_state=42),
    "LogReg (L1, C=0.1)": LogisticRegression(
        penalty="l1", C=0.1, solver="liblinear", random_state=42
    ),
    "RF (default)": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF (balanced)": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "DecisionTree (max_depth=5)": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Dummy (baseline)": DummyClassifier(strategy="most_frequent"),
}

print(f"{len(models)} model configurations defined.")
for name in models:
    print(f"  - {name}")
```

---

## Part 5 — Cross-Validation Loop with Experiment Log

### Line-by-line explanation

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
Creates a 5-fold stratified cross-validator. `shuffle=True` randomly permutes the data before splitting, reducing the risk that rows are ordered by some external variable (e.g., equipment ID). `random_state=42` makes the shuffle reproducible. Stratification ensures each fold contains ~5% failures rather than concentrating all failures in one fold — without it, some folds might contain zero failures, making recall estimates undefined or misleading.

```python
X = equipment.drop("failed", axis=1)
y = equipment["failed"]
```
Separates the feature matrix `X` (8 columns) from the binary target `y`. `axis=1` drops by column name. `y` remains a pandas Series — `cross_validate` and `StratifiedKFold` accept both arrays and Series.

```python
results = []
experiment_log = []
```
Two empty lists that will be populated inside the loop. `results` will hold formatted display strings for the comparison table. `experiment_log` will hold raw numeric scores plus metadata for the CSV log.

```python
for name, model in models.items():
```
Iterates over each `(name, model)` pair in the models dictionary. Adding or removing a model only requires changing the dictionary — the loop code is unchanged.

```python
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
```
Wraps the `ColumnTransformer` and the current model into a single `Pipeline`. This is the **data leakage prevention step**: when `cross_validate` calls `.fit(X_train, y_train)` on each fold, the `preprocessor` fits its `StandardScaler` and `OneHotEncoder` using only the training rows. When it calls `.predict(X_test)`, the preprocessor transforms test rows using statistics from the training fold. This exactly mirrors production conditions, where held-out data is never seen during training.

```python
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
```
A dictionary of metric names mapped to scikit-learn scorer strings. Passing a dictionary to `cross_validate` computes all four metrics in a single pass over the data rather than running CV four separate times.

```python
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
```
Runs the 5-fold CV loop. Returns a dictionary where each key is a scorer name prefixed with `"test_"` (e.g., `"test_recall"`), mapped to a NumPy array of 5 fold scores. `return_train_score` defaults to `False`, omitting training scores.

```python
    row = {
        "Model": name,
        "Accuracy": f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
        ...
    }
    results.append(row)
```
For each metric, computes the **mean** and **standard deviation** across the 5 folds and formats them as `"mean ± std"` strings. The standard deviation indicates stability — a model with a high mean but also high std is less reliable than one with a slightly lower but consistent mean.

```python
    experiment_log.append({
        "model_name": name,
        "hyperparams": str(model.get_params()),
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean(),
        "timestamp": datetime.datetime.now().isoformat(),
    })
```
Appends a log entry with the model name, all hyperparameters (via `.get_params()`), mean metric scores, and an ISO-formatted timestamp. The experiment log is the difference between a reproducible experiment and a one-off run. In production, it allows you to trace any deployed model back to the exact configuration and timestamp that produced it — essential for auditing and incident investigation.

### Complete code block

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X = equipment.drop("failed", axis=1)
y = equipment["failed"]

results = []
experiment_log = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

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

    print(f"  Done: {name}")

print("\nAll models evaluated.")
```

---

## Part 6 — Results Table

### Line-by-line explanation

```python
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```
Converts the list of result dictionaries to a DataFrame. `.to_string(index=False)` prints the full table without the row index — useful in terminal output where HTML rendering is unavailable. In Jupyter, `results_df` on its own line renders as a formatted HTML table.

```python
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/comparison_table.csv", index=False)
print("\nSaved: results/comparison_table.csv")
```
`os.makedirs("results", exist_ok=True)` creates the `results/` directory if it does not already exist. `exist_ok=True` prevents an error if the directory is already there. `.to_csv(..., index=False)` writes the DataFrame to CSV without the row index column. This is one of the five required deliverable files.

### Complete code block

```python
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/comparison_table.csv", index=False)
print("\nSaved: results/comparison_table.csv")
```

---

## Part 7 — Train/Test Split for Visualizations

### Line-by-line explanation

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
Creates an 80/20 train/test split. This split is **separate from the CV loop** and is used only for the PR curve and calibration diagram.

Why is a separate split needed? `cross_validate` evaluates each fold and returns scores, but it does not return fitted model objects. The PR curve and calibration diagram both require a model fitted on training data and then evaluated on held-out data. The `train_test_split` provides that held-out set.

`stratify=y` ensures the 5% failure rate is preserved in both the train and test sets. Without stratification, a 60-row test set might contain zero failures by chance, making the visualizations meaningless.

`random_state=42` makes the split reproducible — every run produces the same 60 test rows.

```python
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Train failure rate: {y_train.mean():.1%} | Test failure rate: {y_test.mean():.1%}")
```
Confirms the split sizes (240 train, 60 test) and that the failure rate is approximately preserved in both splits.

### Complete code block

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Train failure rate: {y_train.mean():.1%} | Test failure rate: {y_test.mean():.1%}")
```

---

## Part 8 — Precision-Recall Curves

### Line-by-line explanation

```python
fig, ax = plt.subplots(figsize=(8, 6))
```
Creates a single figure with one axes object. `figsize=(8, 6)` sets the dimensions in inches. All three PR curves will be drawn on the same axes for direct comparison.

```python
top_models = {
    "LogReg (L2)": LogisticRegression(random_state=42),
    "RF (default)": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF (balanced)": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
}
```
Defines the three models to include in the PR curve visualization. These are fresh, unfitted instances — separate from the objects used in the CV loop. This is necessary because `cross_validate` does not preserve fitted models between folds.

The dummy classifier is excluded from PR curves because it produces no probability scores (it always predicts class 0), making a threshold-based curve impossible.

```python
for name, model in top_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
```
For each model: wraps it in a pipeline with the same preprocessor, fits the pipeline on the training split, then calls `PrecisionRecallDisplay.from_estimator`. This class method internally calls `.predict_proba` (for classifiers that support it) or `.decision_function`, sweeps through every possible decision threshold, and plots the corresponding (recall, precision) pairs as a curve. `name=name` adds the model label to the legend. `ax=ax` draws all curves onto the same shared axes.

```python
ax.set_title("Precision-Recall Curves -- Equipment Failure")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("results/pr_curves.png", dpi=150)
plt.show()
```
Adds a title and legend, adjusts layout to prevent label clipping, saves the figure at 150 DPI (adequate for screen and standard print), and displays the plot inline in Jupyter. `results/pr_curves.png` is the second required deliverable file.

**Reading PR curves:** The curve closer to the top-right corner achieves high precision at high recall — the ideal. The area under the PR curve (PR-AUC) is reported automatically in the legend by `PrecisionRecallDisplay`. For the equipment failure domain, the key question is: at what threshold can you achieve 80%+ recall while keeping precision high enough that the maintenance team is not overwhelmed by false alarms?

### Complete code block

```python
fig, ax = plt.subplots(figsize=(8, 6))

top_models = {
    "LogReg (L2)": LogisticRegression(random_state=42),
    "RF (default)": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF (balanced)": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
}

for name, model in top_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)

ax.set_title("Precision-Recall Curves -- Equipment Failure")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("results/pr_curves.png", dpi=150)
plt.show()
print("Saved: results/pr_curves.png")
```

---

## Part 9 — Calibration Diagram

### Line-by-line explanation

```python
fig, ax = plt.subplots(figsize=(8, 6))
```
Creates a fresh figure for the calibration diagram — separate from the PR curves figure.

```python
for name, model in top_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    CalibrationDisplay.from_estimator(pipe, X_test, y_test, name=name, n_bins=5, ax=ax)
```
Iterates over the same three `top_models`. Note that the models are re-fitted here — the fits from the PR curves loop are not reused because they were local variables inside that loop. Each pipeline is fitted fresh on `X_train, y_train`.

`CalibrationDisplay.from_estimator` calls `.predict_proba` on `X_test`, groups predicted probabilities into `n_bins=5` bins, and for each bin plots the mean predicted probability against the fraction of actual positives. A perfectly calibrated model traces the diagonal (`y=x`).

`n_bins=5` is intentionally low because the test set has only ~3 failures. More bins would produce empty or misleading bins.

```python
ax.set_title("Calibration Diagram -- Equipment Failure")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/calibration.png", dpi=150)
plt.show()
```
Adds title and legend (`loc="lower right"` avoids the diagonal line), saves to `results/calibration.png` (the third required deliverable file), and displays inline.

**Why calibration matters:** If you plan to set an alert threshold at "50% predicted probability of failure," you need that 50% to mean something. If the model is overconfident (predicted 50% but only 20% of those cases actually fail), your threshold does not mean what you think. The calibration plot catches this. In production, a poorly calibrated model erodes operator trust — if maintenance crews find that most "high probability" flags are false alarms, they stop acting on them.

**Logistic regression** is typically well-calibrated because its output is a proper probability derived from a sigmoid function. **Random forests** tend to be overconfident at extreme probabilities (near 0 and 1) because averaging predictions across many trees compresses the distribution toward 0.5 — this often shows as underconfidence near the extremes on the calibration plot.

### Complete code block

```python
fig, ax = plt.subplots(figsize=(8, 6))

for name, model in top_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    CalibrationDisplay.from_estimator(pipe, X_test, y_test, name=name, n_bins=5, ax=ax)

ax.set_title("Calibration Diagram -- Equipment Failure")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/calibration.png", dpi=150)
plt.show()
print("Saved: results/calibration.png")
```

---

## Part 10 — Save Best Model

### Line-by-line explanation

```python
best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )),
])
best_pipe.fit(X_train, y_train)
```
Constructs and fits the best pipeline — `RF (balanced)` — on the training split. This is a final, fresh fit on `X_train`. The `preprocessor` inside the pipeline will fit its `StandardScaler` and `OneHotEncoder` on these 240 training rows; the fitted parameters become part of the saved pipeline object. In production, you would typically refit on the full dataset (`X, y`) before deploying, so the model benefits from all available data. Here we fit on the train split for consistency with the visualization cells.

```python
joblib.dump(best_pipe, "results/best_model.joblib")
print(f"Model saved: {os.path.getsize('results/best_model.joblib') / 1024:.0f} KB")
```
`joblib.dump` serializes the entire fitted pipeline object — preprocessor state (scaler means, encoder vocabulary) and model weights — to a single `.joblib` file. `os.path.getsize` reads the file size in bytes; dividing by 1024 converts to kilobytes. A typical fitted random forest on this dataset is a few hundred KB.

The `.joblib` format is preferred over Python's built-in `pickle` for scikit-learn objects because `joblib` handles large NumPy arrays more efficiently (memory-mapped storage) and is the format officially recommended by the scikit-learn documentation.

```python
loaded = joblib.load("results/best_model.joblib")
print(f"Loaded model predictions match: {(loaded.predict(X_test) == best_pipe.predict(X_test)).all()}")
```
Loads the saved pipeline from disk into a new Python object `loaded`. Calls `.predict(X_test)` on both `loaded` and `best_pipe` and checks that every prediction is identical with `.all()`. If this prints `True`, the serialization round-trip is verified — the loaded model is functionally identical to the fitted model. This verification step should always be included when saving a model for deployment.

### Complete code block

```python
# Save the balanced RF as the best model
best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )),
])
best_pipe.fit(X_train, y_train)

joblib.dump(best_pipe, "results/best_model.joblib")
print(f"Model saved: {os.path.getsize('results/best_model.joblib') / 1024:.0f} KB")

# Verify it loads and produces identical predictions
loaded = joblib.load("results/best_model.joblib")
print(f"Loaded model predictions match: {(loaded.predict(X_test) == best_pipe.predict(X_test)).all()}")
```

---

## Part 11 — Save Experiment Log

### Line-by-line explanation

```python
experiment_log_df = pd.DataFrame(experiment_log)
```
Converts the list of log entry dictionaries (populated during the CV loop in Part 5) into a DataFrame. Each row is one model run; columns are `model_name`, `hyperparams`, `accuracy`, `precision`, `recall`, `f1`, and `timestamp`.

```python
experiment_log_df.to_csv("results/experiment_log.csv", index=False)
print("Saved: results/experiment_log.csv")
```
Saves the log to `results/experiment_log.csv`, the fifth required deliverable file. `index=False` omits the auto-generated row numbers.

```python
print(experiment_log_df[["model_name", "accuracy", "precision", "recall", "f1"]].round(3).to_string(index=False))
```
Prints a compact summary of the log, selecting only the key columns and rounding metric values to 3 decimal places. This allows a quick visual comparison of raw numeric scores without the verbose hyperparameter strings.

**Why the experiment log matters:** Without logging, reproducing a result requires re-running everything from scratch and hoping that randomness was controlled identically. With the log, you can look at the CSV and know exactly what was run, when, with what hyperparameters, and what results were obtained. This is standard practice in any production ML workflow and connects directly to the deployment and monitoring modules later in the bootcamp.

### Complete code block

```python
experiment_log_df = pd.DataFrame(experiment_log)
experiment_log_df.to_csv("results/experiment_log.csv", index=False)
print("Saved: results/experiment_log.csv")
print(experiment_log_df[["model_name", "accuracy", "precision", "recall", "f1"]].round(3).to_string(index=False))
```

---

## Part 12 — Decision Memo

### What this section asks you to do

The decision memo is the **highest-weight section** of the rubric (25%). It requires you to move beyond reading numbers off a table and instead reason about why certain metrics matter more in this specific domain.

A memo that says "I recommend the model with the highest accuracy" demonstrates no understanding of the problem. A memo that names a model, cites recall and the PR curve, explains the asymmetric cost structure of manufacturing failures, and identifies a concrete limitation demonstrates the kind of reasoning that makes an engineer effective.

### Required reasoning framework

| Question | What to address |
|---|---|
| Which metric is the priority? | Recall — missing a failure is a safety hazard, not just a business cost |
| Which model maximizes recall? | Identify from the results table (likely `RF (balanced)`) |
| How does it compare to the baseline? | Compare to `Dummy (baseline)` — which has ~0 recall at ~95% accuracy |
| What is the precision-recall tradeoff? | Higher recall means more false alarms; quantify the maintenance burden |
| Is the model calibrated? | Reference the calibration diagram; discuss whether the threshold can be trusted |
| What could go wrong in production? | Concept drift, sensor failures, failure mode specificity |

### Memo structure (required three paragraphs)

**Paragraph 1 — Recommendation:** Name the model. Cite its recall and compare it to at least one other model. State why recall is the priority metric — not accuracy, not precision. Reference the 5% failure rate and the consequence of a missed failure.

**Paragraph 2 — Tradeoff analysis:** Explain the precision-recall tradeoff in manufacturing terms. A false negative means an undetected failure — equipment damage, production downtime, and potential worker injury. A false positive triggers an inspection that costs maintenance hours. At a 5% failure rate, the balanced RF flags some fraction of equipment for inspection — estimate whether that workload is manageable. Reference the calibration diagram: if the model's predicted probabilities are trustworthy, the threshold can be set with confidence; if they are not, note that.

**Paragraph 3 — Limitations:** Identify at least one limitation. Options include: (a) concept drift — sensor distributions change as equipment ages beyond the training data range; (b) the model predicts failure occurrence but not failure mode, so the maintenance team still needs diagnostics; (c) 300 rows is small for tree-based models. Propose one mitigation (e.g., quarterly retraining, adding failure-mode labels, expanding the dataset with historical records).

### Common mistakes to avoid

- Recommending the model with the highest accuracy without mentioning class imbalance
- Ignoring `Dummy (baseline)` — the rubric explicitly checks for this comparison
- Writing a generic paragraph that could apply to any dataset rather than this specific manufacturing context
- Skipping the calibration diagram reference — it is a required deliverable and must appear in the memo

---

## Summary: How the Parts Connect

```
Dataset (Part 1)
    ↓
Data inspection (Part 2)  ← confirm ~5% failure rate
    ↓
ColumnTransformer (Part 3)  ← numeric scaling + categorical encoding
    ↓
Model configurations (Part 4)  ← 6 models including Dummy baseline
    ↓
Pipeline wrapping (Part 5)  ← preprocessor + model = no data leakage
    ↓
cross_validate (Part 5)  ← StratifiedKFold, 4 metrics, experiment log
    ↓
Results table + comparison_table.csv (Part 6)
    ↓
train_test_split (Part 7)  ← needed because cross_validate doesn't return fitted models
    ↓
PR curves + pr_curves.png (Part 8)  ← full threshold tradeoff
    ↓
Calibration diagram + calibration.png (Part 9)  ← probability trustworthiness
    ↓
Save best model + best_model.joblib (Part 10)  ← bridge to deployment
    ↓
Save experiment log + experiment_log.csv (Part 11)  ← reproducibility
    ↓
Decision memo (Part 12)  ← business reasoning, not just "highest metric wins"
```

The `Pipeline` + `ColumnTransformer` + `StratifiedKFold` pattern you built in this lab is the foundation that will be extended in Module 10 (deployment with FastAPI) and Module 11 (monitoring and concept drift). `joblib.dump` is the first step in that chain.
