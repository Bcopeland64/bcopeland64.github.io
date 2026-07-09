# Loan Default Prediction — Line-by-Line Walkthrough
**Module 5B | Jordanian Microfinance Institution Demo**

---

## Table of Contents
1. [Section 1 — Data Generation](#section-1--data-generation)
2. [Section 2 — Exploration & Class Imbalance](#section-2--exploration--class-imbalance)
3. [Section 3 — Train / Test Split](#section-3--train--test-split)
4. [Section 4 — Decision Tree: Baseline & Visualization](#section-4--decision-tree-baseline--visualization)
5. [Section 5 — Overfitting: Capped vs. Uncapped Tree](#section-5--overfitting-capped-vs-uncapped-tree)
6. [Section 6 — Random Forest & Feature Importances](#section-6--random-forest--feature-importances)
7. [Section 7 — Class Weighting: Default vs. Balanced](#section-7--class-weighting-default-vs-balanced)
8. [Section 8 — Precision-Recall Curves & PR-AUC](#section-8--precision-recall-curves--pr-auc)
9. [Section 9 — Advanced Technique 1: SMOTE](#section-9--advanced-technique-1-smote)
10. [Section 10 — Advanced Technique 2: Feature Engineering](#section-10--advanced-technique-2-domain-driven-feature-engineering)
11. [Section 11 — Advanced Technique 3: Optimal Threshold Selection](#section-11--advanced-technique-3-optimal-threshold-selection)
12. [Section 12 — Final Model Comparison](#section-12--final-model-comparison)

---

## Section 1 — Data Generation

This section creates 200,000 synthetic loan applicants with realistic statistical distributions. The target variable (`defaulted`) is generated from a logistic function so the features have genuine predictive power.

---

### Line-by-Line

```python
import numpy as np
```
Imports NumPy, the foundation for all numerical operations. Provides array math, random number generation, and statistical functions used throughout the notebook.

```python
import pandas as pd
```
Imports pandas for tabular data handling. The final dataset will live in a `DataFrame`, which gives us column-named access, easy filtering, and `.describe()` summaries.

```python
np.random.seed(42)
```
Seeds the random number generator with 42. Every subsequent call to `np.random.*` will produce the same sequence on every run, making the notebook fully reproducible.

```python
n = 200_000
```
Sets the dataset size to 200,000 applicants. Python allows underscores in integer literals as thousand-separators — this is purely cosmetic and equal to `200000`.

```python
monthly_income = np.random.lognormal(mean=7.0, sigma=0.5, size=n).round(0)
```
Draws monthly income from a log-normal distribution. Log-normal is appropriate because income is always positive and positively skewed — most people earn moderate amounts, a few earn very high amounts. `mean=7.0` and `sigma=0.5` in log-space produce incomes centered around ~JD 1,100/month. `.round(0)` removes decimals.

```python
employment_years = np.random.exponential(scale=5, size=n).round(1)
```
Draws years of employment from an exponential distribution. Exponential is appropriate because job tenure naturally clusters near zero (many new hires) and thins out at higher values. `scale=5` means the average tenure is 5 years.

```python
loan_amount = np.random.uniform(500, 15000, n).round(0)
```
Draws loan sizes uniformly between JD 500 and JD 15,000. Uniform means every loan size in that range is equally likely — a reasonable simplification for a microfinance portfolio.

```python
credit_score = np.random.normal(650, 80, n).clip(300, 850).round(0)
```
Draws credit scores from a normal distribution centered at 650 with standard deviation 80. `.clip(300, 850)` enforces the real-world credit score bounds so no score falls outside the valid range.

```python
prior_defaults = np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05])
```
Assigns prior default count using a discrete probability distribution. 70% of applicants have no prior defaults, 15% have one, 10% have two, and 5% have three. `p` must sum to 1.0.

```python
has_collateral = np.random.choice([0, 1], n, p=[0.4, 0.6])
```
Creates a binary collateral indicator. 60% of applicants offer collateral (1), 40% do not (0). Having collateral reduces default risk in the logistic formula below.

```python
region = np.random.choice(
    ["Amman", "Irbid", "Zarqa", "Aqaba", "Mafraq"], n,
    p=[0.35, 0.2, 0.2, 0.15, 0.1]
)
```
Assigns each applicant to one of five Jordanian governorates with probabilities reflecting approximate population distribution. Amman is the largest city (35%).

```python
default_prob = 1 / (1 + np.exp(
    2.0
    - 0.0003 * monthly_income
    - 0.005  * credit_score
    + 0.8    * prior_defaults
    + 0.0002 * loan_amount
    - 0.05   * employment_years
    - 0.3    * has_collateral
    + np.random.normal(0, 0.3, n)
))
```
Computes a per-applicant default probability using the logistic (sigmoid) function: `1 / (1 + exp(-z))`. The expression inside `exp()` is the linear score `z`. Each coefficient encodes domain logic:
- `+2.0` — base intercept (sets the overall default rate)
- `-0.0003 * monthly_income` — higher income → lower default risk
- `-0.005 * credit_score` — higher credit score → lower default risk
- `+0.8 * prior_defaults` — prior defaults strongly increase risk
- `+0.0002 * loan_amount` — larger loans → slightly higher risk
- `-0.05 * employment_years` — longer employment → lower risk
- `-0.3 * has_collateral` — collateral reduces risk
- `np.random.normal(0, 0.3, n)` — individual-level noise that makes the relationship probabilistic, not deterministic

```python
defaulted = (np.random.random(n) < default_prob).astype(int)
```
Converts each probability into a binary outcome by drawing a uniform random number. If the random draw is less than `default_prob`, the applicant defaults (1); otherwise they do not (0). `.astype(int)` converts the boolean array to integers.

```python
loans = pd.DataFrame({
    "monthly_income":   monthly_income,
    "employment_years": employment_years,
    "loan_amount":      loan_amount,
    "credit_score":     credit_score,
    "prior_defaults":   prior_defaults,
    "has_collateral":   has_collateral,
    "region":           region,
    "defaulted":        defaulted,
})
```
Assembles all seven feature arrays and the target into a single pandas DataFrame with named columns. This is the canonical format expected by scikit-learn.

```python
print(loans.shape)
print(f"Default rate: {defaulted.mean():.1%}")
print(loans.describe().round(1))
```
- `.shape` confirms (200000, 8) — 200,000 rows and 8 columns.
- `.mean()` on the binary target gives the proportion of 1s, formatted as a percentage with one decimal.
- `.describe()` shows min, max, mean, standard deviation, and quartiles for all numeric columns.

---

### Complete Code Block — Section 1

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200_000

monthly_income     = np.random.lognormal(mean=7.0, sigma=0.5, size=n).round(0)
employment_years   = np.random.exponential(scale=5, size=n).round(1)
loan_amount        = np.random.uniform(500, 15000, n).round(0)
credit_score       = np.random.normal(650, 80, n).clip(300, 850).round(0)
prior_defaults     = np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05])
has_collateral     = np.random.choice([0, 1], n, p=[0.4, 0.6])
region             = np.random.choice(
    ["Amman", "Irbid", "Zarqa", "Aqaba", "Mafraq"], n,
    p=[0.35, 0.2, 0.2, 0.15, 0.1]
)

default_prob = 1 / (1 + np.exp(
    2.0
    - 0.0003 * monthly_income
    - 0.005  * credit_score
    + 0.8    * prior_defaults
    + 0.0002 * loan_amount
    - 0.05   * employment_years
    - 0.3    * has_collateral
    + np.random.normal(0, 0.3, n)
))
defaulted = (np.random.random(n) < default_prob).astype(int)

loans = pd.DataFrame({
    "monthly_income":   monthly_income,
    "employment_years": employment_years,
    "loan_amount":      loan_amount,
    "credit_score":     credit_score,
    "prior_defaults":   prior_defaults,
    "has_collateral":   has_collateral,
    "region":           region,
    "defaulted":        defaulted,
})

print(loans.shape)
print(f"Default rate: {defaulted.mean():.1%}")
print(loans.describe().round(1))
```

---

## Section 2 — Exploration & Class Imbalance

Before building any model, we inspect the dataset to understand the class distribution and how features differ between defaulters and non-defaulters.

---

### Line-by-Line

```python
import matplotlib.pyplot as plt
import seaborn as sns
```
Imports the two plotting libraries. `matplotlib.pyplot` is the low-level engine; `seaborn` builds on it with higher-level statistical chart functions. Both are standard in data science notebooks.

```python
print(f"Default rate:  {loans['defaulted'].mean():.1%}")
print(f"Defaults:      {loans['defaulted'].sum():,} out of {len(loans):,}")
```
Computes and prints two key facts: the proportion of defaults (`.mean()` on 0/1 column) and the absolute count (`.sum()`). The `:,` format specifier adds thousand-separators for readability.

```python
print(loans['defaulted'].value_counts())
```
Shows the exact count of each class (0 and 1) sorted by frequency. Immediately reveals the imbalance: ~184,000 non-defaults vs. ~16,000 defaults.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```
Creates a figure with two side-by-side subplots, each referenced through the `axes` array. `figsize=(14, 5)` sets the total canvas width to 14 inches and height to 5 inches.

```python
counts = loans['defaulted'].value_counts()
axes[0].bar(
    ["No Default (0)", "Default (1)"],
    counts.values,
    color=["steelblue", "tomato"],
    edgecolor="black"
)
```
Draws a bar chart of class counts. `counts.values` provides the heights. Two distinct colors make the imbalance visually obvious.

```python
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10)
```
Adds a title, y-axis label, and value annotations sitting just above each bar (`v + 200` for vertical offset). `ha="center"` horizontally centers each label.

```python
loans[loans["defaulted"] == 0]["credit_score"].plot(
    kind="hist", bins=40, alpha=0.6, color="steelblue",
    label="No Default", ax=axes[1]
)
loans[loans["defaulted"] == 1]["credit_score"].plot(
    kind="hist", bins=40, alpha=0.6, color="tomato",
    label="Default", ax=axes[1]
)
```
Plots overlapping histograms of `credit_score` split by class. `alpha=0.6` makes both histograms semi-transparent so overlapping regions are visible. 40 bins gives enough granularity without noise.

```python
axes[1].set_title("Credit Score by Default Status")
axes[1].set_xlabel("Credit Score")
axes[1].legend()
```
Adds chart labels and a legend identifying the two distributions.

```python
plt.tight_layout()
plt.show()
```
`tight_layout()` automatically adjusts subplot spacing to prevent label overlap. `show()` renders the figure inline in the notebook.

---

### Complete Code Block — Section 2

```python
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Default rate:  {loans['defaulted'].mean():.1%}")
print(f"Defaults:      {loans['defaulted'].sum():,} out of {len(loans):,}")
print()
print("Class distribution:")
print(loans['defaulted'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = loans['defaulted'].value_counts()
axes[0].bar(
    ["No Default (0)", "Default (1)"],
    counts.values,
    color=["steelblue", "tomato"],
    edgecolor="black"
)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10)

loans[loans["defaulted"] == 0]["credit_score"].plot(
    kind="hist", bins=40, alpha=0.6, color="steelblue",
    label="No Default", ax=axes[1]
)
loans[loans["defaulted"] == 1]["credit_score"].plot(
    kind="hist", bins=40, alpha=0.6, color="tomato",
    label="Default", ax=axes[1]
)
axes[1].set_title("Credit Score by Default Status")
axes[1].set_xlabel("Credit Score")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Section 3 — Train / Test Split

Splits the data into a training set (80%) and a held-out test set (20%) before any model is built.

---

### Line-by-Line

```python
from sklearn.model_selection import train_test_split
```
Imports scikit-learn's utility for splitting data. It handles random shuffling and stratification automatically.

```python
X = loans[[
    "monthly_income", "employment_years", "loan_amount",
    "credit_score", "prior_defaults", "has_collateral"
]]
```
Creates the feature matrix `X` by selecting only the six numeric predictor columns. The `region` column is excluded here because it requires categorical encoding (a `ColumnTransformer` step introduced in Week A). The double bracket syntax `[[...]]` returns a DataFrame rather than a Series.

```python
y = loans["defaulted"]
```
Creates the target vector `y`. Single-bracket syntax returns a Series — the format scikit-learn expects for labels.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```
Splits into 80% train / 20% test. Key arguments:
- `test_size=0.2` — reserve 20% for evaluation
- `random_state=42` — reproducible shuffle
- `stratify=y` — preserves the ~8% default rate in *both* splits; without this, a random shuffle could accidentally put most defaults in one split

```python
print(f"Training set:  {X_train.shape[0]:,} samples")
print(f"Test set:      {X_test.shape[0]:,} samples")
print(f"Train default rate: {y_train.mean():.1%}")
print(f"Test  default rate: {y_test.mean():.1%}")
```
Confirms the split sizes (160,000 / 40,000) and verifies that the default rate is approximately equal in both sets — confirmation that `stratify=y` worked.

---

### Complete Code Block — Section 3

```python
from sklearn.model_selection import train_test_split

X = loans[[
    "monthly_income", "employment_years", "loan_amount",
    "credit_score", "prior_defaults", "has_collateral"
]]
y = loans["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set:  {X_train.shape[0]:,} samples")
print(f"Test set:      {X_test.shape[0]:,} samples")
print(f"Train default rate: {y_train.mean():.1%}")
print(f"Test  default rate: {y_test.mean():.1%}")
```

---

## Section 4 — Decision Tree: Baseline & Visualization

Trains a single decision tree with a depth cap and visualizes the learned splitting rules.

---

### Line-by-Line

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
```
Imports the decision tree classifier, the tree-drawing utility, and the classification report function. `classification_report` prints precision, recall, F1, and support for every class.

```python
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
```
Instantiates a decision tree capped at 5 levels deep. `max_depth=5` is a regularization hyperparameter that prevents the tree from memorizing the training data. `random_state=42` makes tie-breaking deterministic.

```python
tree.fit(X_train, y_train)
```
Trains the tree by recursively finding the best feature and threshold to split on at each node. "Best" means the split that maximizes the reduction in Gini impurity. The tree learns nothing from the test set during this call.

```python
y_pred_tree = tree.predict(X_test)
```
Applies the learned tree to the unseen test set. Each test sample walks down the tree from root to leaf, and the leaf's majority class becomes the prediction.

```python
print("Decision Tree (max_depth=5) — Classification Report")
print(classification_report(y_test, y_pred_tree, target_names=["no_default", "default"]))
```
Prints per-class metrics. The key numbers to watch are:
- **Recall for class 1 (default)** — proportion of actual defaults the tree caught
- **Precision for class 1** — of predicted defaults, how many were real

With an imbalanced dataset and no class weighting, recall for defaults is typically low here — the tree favours the majority class.

```python
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["no_default", "default"],
    filled=True,
    max_depth=3,
    ax=ax,
    fontsize=8
)
plt.title("Decision Tree — Loan Default (max_depth=5, showing 3 levels)", fontsize=14)
plt.tight_layout()
plt.show()
```
Draws the first 3 levels of the 5-level tree. Each node shows:
- The splitting condition (e.g., `credit_score <= 612.5`)
- Gini impurity (lower = purer)
- Sample count and class breakdown
- Node color (orange tint = majority default, blue tint = majority no-default)

`filled=True` colors each node by the majority class. Showing only `max_depth=3` keeps the diagram readable.

---

### Complete Code Block — Section 4

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("Decision Tree (max_depth=5) — Classification Report")
print(classification_report(y_test, y_pred_tree, target_names=["no_default", "default"]))

fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["no_default", "default"],
    filled=True,
    max_depth=3,
    ax=ax,
    fontsize=8
)
plt.title("Decision Tree — Loan Default (max_depth=5, showing 3 levels)", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Section 5 — Overfitting: Capped vs. Uncapped Tree

Demonstrates overfitting by training a tree with no depth limit and comparing its train/test accuracy to the capped tree.

---

### Line-by-Line

```python
tree_deep = DecisionTreeClassifier(random_state=42)
```
Creates a second tree with no `max_depth` argument. Without a constraint, the tree will grow until every leaf is perfectly pure — it memorizes the training data.

```python
tree_deep.fit(X_train, y_train)
```
Fits the unconstrained tree. With 200,000 training samples and 6 features, this tree will grow very deep, creating thousands of leaves that each contain only one or two identical training examples.

```python
print(f"Deep tree  — Train accuracy: {tree_deep.score(X_train, y_train):.4f}")
print(f"Deep tree  — Test  accuracy: {tree_deep.score(X_test,  y_test):.4f}")
```
`.score()` returns accuracy (fraction of correct predictions). The deep tree will print 1.0000 on training and a noticeably lower number on test — the signature of overfitting.

```python
print(f"Capped tree — Train accuracy: {tree.score(X_train, y_train):.4f}")
print(f"Capped tree — Test  accuracy: {tree.score(X_test,  y_test):.4f}")
```
The capped tree has lower training accuracy but comparable or better test accuracy — it generalizes instead of memorizing.

```python
print(f"Deep tree max_depth reached: {tree_deep.get_depth()}")
```
`.get_depth()` returns how many levels the unconstrained tree actually grew to. On 200,000 samples this is typically 40-50 levels, vs. our capped tree's 5.

---

### Complete Code Block — Section 5

```python
tree_deep = DecisionTreeClassifier(random_state=42)
tree_deep.fit(X_train, y_train)

print(f"Deep tree  — Train accuracy: {tree_deep.score(X_train, y_train):.4f}")
print(f"Deep tree  — Test  accuracy: {tree_deep.score(X_test,  y_test):.4f}")
print()
print(f"Capped tree — Train accuracy: {tree.score(X_train, y_train):.4f}")
print(f"Capped tree — Test  accuracy: {tree.score(X_test,  y_test):.4f}")
print()
print(f"Deep tree max_depth reached: {tree_deep.get_depth()}")
```

---

## Section 6 — Random Forest & Feature Importances

Trains an ensemble of 100 decision trees and extracts which features drove the most splits.

---

### Line-by-Line

```python
from sklearn.ensemble import RandomForestClassifier
```
Imports the Random Forest classifier from scikit-learn's ensemble module.

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```
Instantiates a random forest.
- `n_estimators=100` — build 100 trees
- `random_state=42` — reproducible
- `n_jobs=-1` — use all available CPU cores in parallel (critical for 200,000 samples)

Each tree is trained on a bootstrap sample (random subset with replacement) of the training data, and each split considers only a random subset of features. These two randomizations de-correlate the trees so their errors cancel when aggregated.

```python
rf.fit(X_train, y_train)
```
Trains all 100 trees in parallel (given `n_jobs=-1`). Each tree independently learns on its bootstrap sample.

```python
y_pred_rf = rf.predict(X_test)
```
Each of the 100 trees votes for a class; the forest predicts whichever class gets the most votes. This majority vote is more robust than any single tree.

```python
print("Random Forest (default) — Classification Report")
print(classification_report(y_test, y_pred_rf, target_names=["no_default", "default"]))
```
Prints class-level metrics. The forest typically shows better recall for the majority class than a single tree, but still underperforms on the minority class without weighting.

```python
importances = rf.feature_importances_
```
`feature_importances_` is a post-training attribute containing the mean decrease in Gini impurity contributed by each feature across all 100 trees, normalized to sum to 1.0.

```python
importance_df = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False).reset_index(drop=True)
```
Builds a clean DataFrame pairing feature names with their importance scores, sorted highest to lowest. `reset_index(drop=True)` re-numbers the rows after sorting.

```python
print(importance_df.to_string(index=False))
```
`.to_string(index=False)` prints the DataFrame without the row number index — cleaner for reading.

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="steelblue",
    edgecolor="black"
)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Random Forest Feature Importances — Loan Default")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```
Horizontal bar chart with features on the y-axis. `invert_yaxis()` puts the most important feature at the top. `credit_score` and `prior_defaults` should dominate; `has_collateral` and `employment_years` will rank lower.

---

### Complete Code Block — Section 6

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest (default) — Classification Report")
print(classification_report(y_test, y_pred_rf, target_names=["no_default", "default"]))

importances = rf.feature_importances_
importance_df = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="steelblue",
    edgecolor="black"
)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Random Forest Feature Importances — Loan Default")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

---

## Section 7 — Class Weighting: Default vs. Balanced

Retrains the random forest with `class_weight="balanced"` to make the model more sensitive to the minority class.

---

### Line-by-Line

```python
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```
`class_weight="balanced"` instructs scikit-learn to automatically compute class weights as `n_samples / (n_classes * np.bincount(y))`. For an 8% default rate, the minority class (1) receives a weight of approximately 92/8 ≈ 11.5×, meaning each default sample counts 11.5 times as much as a non-default sample in the Gini calculation.

```python
rf_balanced.fit(X_train, y_train)
```
Trains the balanced forest. The trees now split more aggressively to separate the minority class because misclassifying a defaulter is weighted much more heavily in the impurity calculation.

```python
y_pred_balanced = rf_balanced.predict(X_test)
```
Generates predictions from the balanced model. With `class_weight="balanced"`, the effective decision threshold shifts downward — the model predicts "default" at lower probability thresholds, which increases recall but reduces precision.

```python
print("Default RF:")
print(classification_report(y_test, y_pred_rf, target_names=["no_default", "default"], digits=3))
print("\nBalanced RF:")
print(classification_report(y_test, y_pred_balanced, target_names=["no_default", "default"], digits=3))
```
Side-by-side comparison. The balanced RF will show notably higher recall for class 1 (defaults caught) but lower precision (more false alarms). The overall accuracy drops slightly because the model now makes more errors on the majority class. This trade-off is intentional and often correct for risk applications.

---

### Complete Code Block — Section 7

```python
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

print("Default RF:")
print(classification_report(y_test, y_pred_rf, target_names=["no_default", "default"], digits=3))
print("\nBalanced RF:")
print(classification_report(y_test, y_pred_balanced, target_names=["no_default", "default"], digits=3))
```

---

## Section 8 — Precision-Recall Curves & PR-AUC

Plots the full precision-recall tradeoff curve for both RF variants and computes PR-AUC.

---

### Line-by-Line

```python
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
```
`PrecisionRecallDisplay` builds and plots the PR curve in one call. `average_precision_score` computes the area under the PR curve (PR-AUC), which summarizes the entire curve in a single number.

```python
fig, ax = plt.subplots(figsize=(9, 6))
```
Creates the figure canvas that both PR curves will share.

```python
PrecisionRecallDisplay.from_estimator(
    rf, X_test, y_test, name="RF (default)", ax=ax
)
```
Calls `.predict_proba()` on `rf` internally, computes precision and recall at every possible probability threshold, and plots the resulting curve. The `name` argument sets the legend label.

```python
PrecisionRecallDisplay.from_estimator(
    rf_balanced, X_test, y_test, name="RF (balanced)", ax=ax
)
```
Overlays the balanced RF's curve on the same axes. A curve further toward the top-right corner is better.

```python
ax.set_title("Precision-Recall Curves — Loan Default")
ax.legend()
plt.tight_layout()
plt.show()
```
Adds the chart title and legend identifying which curve belongs to which model.

```python
pr_auc_default  = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
pr_auc_balanced = average_precision_score(y_test, rf_balanced.predict_proba(X_test)[:, 1])
```
`.predict_proba(X_test)` returns an (n, 2) matrix of class probabilities. `[:, 1]` selects the second column — the probability of class 1 (default). `average_precision_score` integrates the PR curve to produce a single score; 1.0 is perfect, baseline is the actual default rate (~0.08).

```python
print(f"PR-AUC (default):  {pr_auc_default:.4f}")
print(f"PR-AUC (balanced): {pr_auc_balanced:.4f}")
```
Prints the two summary scores. The balanced RF typically has a slightly lower or similar PR-AUC because the curve shape shifts — it has higher recall at the cost of lower precision, but the integrated area may be similar.

---

### Complete Code Block — Section 8

```python
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

fig, ax = plt.subplots(figsize=(9, 6))

PrecisionRecallDisplay.from_estimator(
    rf,          X_test, y_test, name="RF (default)",  ax=ax
)
PrecisionRecallDisplay.from_estimator(
    rf_balanced, X_test, y_test, name="RF (balanced)", ax=ax
)

ax.set_title("Precision-Recall Curves — Loan Default")
ax.legend()
plt.tight_layout()
plt.show()

pr_auc_default  = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
pr_auc_balanced = average_precision_score(y_test, rf_balanced.predict_proba(X_test)[:, 1])
print(f"PR-AUC (default):  {pr_auc_default:.4f}")
print(f"PR-AUC (balanced): {pr_auc_balanced:.4f}")
```

---

## Section 9 — Advanced Technique 1: SMOTE

**SMOTE (Synthetic Minority Over-Sampling Technique)** creates artificial minority-class examples by interpolating between real ones, balancing the training set at the data level rather than the loss level.

---

### Line-by-Line

```python
from imblearn.over_sampling import SMOTE
```
Imports SMOTE from the `imbalanced-learn` library (`imblearn`). This library extends scikit-learn with algorithms designed specifically for imbalanced classification.

```python
smote = SMOTE(random_state=42, n_jobs=-1)
```
Instantiates the SMOTE resampler. SMOTE works by:
1. Selecting a minority-class sample
2. Finding its k nearest minority-class neighbours (default k=5) in feature space
3. Drawing a random point on the line segment connecting the sample to one of its neighbours
4. That synthetic point becomes a new training example

`n_jobs=-1` parallelizes the nearest-neighbour search across all cores.

```python
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```
`fit_resample()` analyses `y_train` to determine how many synthetic samples are needed to balance the classes (up-samples the minority class to match the majority), then generates those samples using the algorithm above. The test set is **never** touched — applying SMOTE to the test set would leak information and invalidate evaluation.

```python
print(f"Before SMOTE — training samples: {len(X_train):,}")
print(f"Before SMOTE — default rate:     {y_train.mean():.1%}")
print()
print(f"After  SMOTE — training samples: {len(X_train_sm):,}")
print(f"After  SMOTE — default rate:     {y_train_sm.mean():.1%}")
```
Confirms the resampling: training size approximately doubles (from ~160,000 to ~320,000) and the default rate becomes 50% — perfectly balanced.

```python
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_sm, y_train_sm)
```
Trains a standard random forest (no `class_weight`) on the SMOTE-balanced training set. Because the training data is now 50/50, the forest sees equal numbers of each class and learns a balanced decision boundary without needing to re-weight the loss.

```python
y_pred_smote = rf_smote.predict(X_test)
```
Generates predictions on the original (imbalanced) test set. It is crucial that the test set reflects the real distribution — if it were also balanced, the evaluation metrics would not generalize to deployment.

```python
print("RF + SMOTE — Classification Report")
print(classification_report(y_test, y_pred_smote, target_names=["no_default", "default"], digits=3))
```
Prints the SMOTE model's metrics. Compare recall for class 1 against the balanced RF from Section 7. SMOTE sometimes achieves higher recall because it expands the minority class's representation in feature space rather than just reweighting.

```python
pr_auc_smote = average_precision_score(y_test, rf_smote.predict_proba(X_test)[:, 1])
print(f"PR-AUC (SMOTE): {pr_auc_smote:.4f}")
```
PR-AUC for the SMOTE model. A higher number than the balanced RF indicates better overall precision-recall tradeoff.

```python
fig, ax = plt.subplots(figsize=(9, 6))

PrecisionRecallDisplay.from_estimator(rf,          X_test, y_test, name="RF (default)",  ax=ax)
PrecisionRecallDisplay.from_estimator(rf_balanced, X_test, y_test, name="RF (balanced)", ax=ax)
PrecisionRecallDisplay.from_estimator(rf_smote,    X_test, y_test, name="RF + SMOTE",    ax=ax)

ax.set_title("Precision-Recall: Default vs. Balanced vs. SMOTE")
ax.legend()
plt.tight_layout()
plt.show()
```
Three-way PR curve comparison. If the SMOTE curve sits above both others, it dominates across all threshold choices.

---

### Complete Code Block — Section 9

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, n_jobs=-1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE — training samples: {len(X_train):,}")
print(f"Before SMOTE — default rate:     {y_train.mean():.1%}")
print()
print(f"After  SMOTE — training samples: {len(X_train_sm):,}")
print(f"After  SMOTE — default rate:     {y_train_sm.mean():.1%}")

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_sm, y_train_sm)
y_pred_smote = rf_smote.predict(X_test)

print("RF + SMOTE — Classification Report")
print(classification_report(y_test, y_pred_smote, target_names=["no_default", "default"], digits=3))

pr_auc_smote = average_precision_score(y_test, rf_smote.predict_proba(X_test)[:, 1])
print(f"PR-AUC (SMOTE): {pr_auc_smote:.4f}")

fig, ax = plt.subplots(figsize=(9, 6))
PrecisionRecallDisplay.from_estimator(rf,          X_test, y_test, name="RF (default)",  ax=ax)
PrecisionRecallDisplay.from_estimator(rf_balanced, X_test, y_test, name="RF (balanced)", ax=ax)
PrecisionRecallDisplay.from_estimator(rf_smote,    X_test, y_test, name="RF + SMOTE",    ax=ax)
ax.set_title("Precision-Recall: Default vs. Balanced vs. SMOTE")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Section 10 — Advanced Technique 2: Domain-Driven Feature Engineering

Creates three new features that encode domain knowledge, giving the model compressed signals it cannot derive from raw columns alone.

---

### Line-by-Line

```python
def add_engineered_features(df):
```
Defines a reusable function so the exact same transformations can be applied consistently to both the training and test sets. Applying different transformations to train vs. test is a common source of data leakage.

```python
    out = df.copy()
```
Creates a copy of the input DataFrame to avoid mutating the original. `.copy()` is important — without it, modifying `out` would silently modify the original `df` due to pandas' reference semantics.

```python
    out["debt_to_income_ratio"] = out["loan_amount"] / (out["monthly_income"] + 1)
```
Computes the debt-to-income (DTI) ratio: loan amount divided by monthly income. `+ 1` prevents division by zero if any income is 0. A high DTI means the applicant is borrowing a large multiple of their monthly earnings — a classic risk indicator in lending that a tree cannot discover by splitting on `loan_amount` and `monthly_income` independently without this ratio.

```python
    out["credit_risk_score"] = out["credit_score"] - (out["prior_defaults"] * 50)
```
Creates a composite risk score by penalizing credit score for prior defaults. Each prior default subtracts 50 points from the credit score. This encodes domain knowledge that prior defaults matter more than raw credit score alone, and compresses two correlated features into one signal.

```python
    out["income_stability"] = out["employment_years"] / (out["monthly_income"] / 1000 + 0.01)
```
Measures employment tenure relative to income level. Someone earning a high income with short tenure is less stable than someone earning the same income with long tenure. `/ 1000` scales income to years units for comparability. `+ 0.01` prevents zero division.

```python
    return out
```
Returns the expanded DataFrame with three additional columns appended.

```python
X_train_fe = add_engineered_features(X_train)
X_test_fe  = add_engineered_features(X_test)
```
Applies the same function to both sets. The function uses only column arithmetic — no statistics derived from the training set — so applying it to the test set introduces no leakage.

```python
print("New feature columns added:")
print(X_train_fe.columns.tolist())
```
Verifies the 9-column feature matrix (6 original + 3 engineered).

```python
print(X_train_fe[["debt_to_income_ratio", "credit_risk_score", "income_stability"]].describe().round(3))
```
Checks the scale and distribution of the three new features. This sanity check catches accidental infinities or extreme values.

```python
rf_fe = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_fe.fit(X_train_fe, y_train)
```
Trains on the 9-feature matrix with balanced weighting. The combination of feature engineering and class weighting stacks two improvement strategies.

```python
y_pred_fe = rf_fe.predict(X_test_fe)
```
Must use `X_test_fe` (the engineered test set) — not `X_test` — because the model was trained on 9 columns.

```python
pr_auc_fe = average_precision_score(y_test, rf_fe.predict_proba(X_test_fe)[:, 1])
print(f"PR-AUC (feature engineering): {pr_auc_fe:.4f}")
```
PR-AUC for the feature-engineered model.

```python
colors = ["tomato" if "ratio" in f or "risk" in f or "stability" in f else "steelblue"
          for f in fe_importance_df["Feature"]]
```
A list comprehension that assigns red color to the three new features and blue to the originals. This makes it easy to see in the bar chart whether the engineered features rank highly — if they do, the model found them informative.

---

### Complete Code Block — Section 10

```python
def add_engineered_features(df):
    out = df.copy()
    out["debt_to_income_ratio"] = out["loan_amount"] / (out["monthly_income"] + 1)
    out["credit_risk_score"]    = out["credit_score"] - (out["prior_defaults"] * 50)
    out["income_stability"]     = out["employment_years"] / (out["monthly_income"] / 1000 + 0.01)
    return out

X_train_fe = add_engineered_features(X_train)
X_test_fe  = add_engineered_features(X_test)

print("New feature columns added:")
print(X_train_fe.columns.tolist())
print()
print(X_train_fe[["debt_to_income_ratio", "credit_risk_score", "income_stability"]].describe().round(3))

rf_fe = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_fe.fit(X_train_fe, y_train)
y_pred_fe = rf_fe.predict(X_test_fe)

print("RF + Feature Engineering — Classification Report")
print(classification_report(y_test, y_pred_fe, target_names=["no_default", "default"], digits=3))

pr_auc_fe = average_precision_score(y_test, rf_fe.predict_proba(X_test_fe)[:, 1])
print(f"PR-AUC (feature engineering): {pr_auc_fe:.4f}")

fe_importances = rf_fe.feature_importances_
fe_importance_df = pd.DataFrame({
    "Feature":    X_train_fe.columns,
    "Importance": fe_importances
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(fe_importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["tomato" if "ratio" in f or "risk" in f or "stability" in f else "steelblue"
          for f in fe_importance_df["Feature"]]
ax.barh(fe_importance_df["Feature"], fe_importance_df["Importance"],
        color=colors, edgecolor="black")
ax.set_xlabel("Importance")
ax.set_title("Feature Importances With Engineered Features (red = new)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

---

## Section 11 — Advanced Technique 3: Optimal Threshold Selection

Finds the decision threshold that maximizes the F2 score, weighting recall twice as heavily as precision to reflect the asymmetric cost of loan defaults.

---

### Line-by-Line

```python
from sklearn.metrics import precision_recall_curve, fbeta_score, classification_report
```
Imports three tools:
- `precision_recall_curve` — computes precision and recall at every possible threshold
- `fbeta_score` — F-score with a configurable beta weight (used for sanity checking)
- `classification_report` — for comparing threshold variants

```python
proba_balanced = rf_balanced.predict_proba(X_test)[:, 1]
```
Extracts the default probability scores from the balanced RF. These raw probabilities are what we will apply different thresholds to. The model itself is not retrained — only the post-processing rule changes.

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, proba_balanced)
```
Computes precision and recall at every distinct probability value that appears in `proba_balanced`. `thresholds` has length n-1 (one fewer than `precisions` and `recalls` because the final point is the precision=1, recall=0 endpoint with no associated threshold).

```python
beta = 2
f2_scores = (
    (1 + beta**2) * precisions[:-1] * recalls[:-1]
    / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-9)
)
```
Applies the F-beta formula at every threshold:
`F_beta = (1 + beta²) × precision × recall / (beta² × precision + recall)`

With `beta=2`, recall is weighted 4× as much as precision in the denominator (beta squared). `precisions[:-1]` drops the final element to align with `thresholds`. `+ 1e-9` prevents division by zero at degenerate thresholds.

```python
best_idx       = f2_scores.argmax()
best_threshold = thresholds[best_idx]
best_f2        = f2_scores[best_idx]
```
`.argmax()` returns the index of the highest F2 score. We then look up the corresponding threshold and score values.

```python
print(f"Best threshold (F2-maximising): {best_threshold:.4f}")
print(f"F2 score at best threshold:     {best_f2:.4f}")
```
Displays the optimal threshold. For a model with balanced class weighting, this is typically below 0.5 (e.g., 0.20-0.35) — the model should predict "default" whenever it assigns ≥ that probability.

```python
y_pred_optimal = (proba_balanced >= best_threshold).astype(int)
```
Applies the optimal threshold manually. `proba_balanced >= best_threshold` produces a boolean array; `.astype(int)` converts it to 0/1. This replaces what `.predict()` would do with its fixed 0.5 threshold.

```python
print(f"--- Balanced RF at default threshold (0.50) ---")
print(classification_report(y_test, y_pred_balanced, target_names=["no_default", "default"], digits=3))

print(f"--- Balanced RF at optimal threshold ({best_threshold:.4f}) ---")
print(classification_report(y_test, y_pred_optimal, target_names=["no_default", "default"], digits=3))
```
Direct comparison between the two threshold strategies using the same underlying model. Recall for class 1 will be meaningfully higher with the optimal threshold; precision will be lower; F2 will be higher.

```python
axes[0].plot(thresholds, f2_scores, color="steelblue", linewidth=1.5)
axes[0].axvline(best_threshold, color="tomato", linestyle="--",
                label=f"Optimal threshold = {best_threshold:.3f}")
```
Plots F2 score as a function of threshold. The vertical red dashed line marks the optimal point. The curve shows a clear peak — thresholds above or below it both hurt F2.

```python
axes[1].plot(thresholds, precisions[:-1], label="Precision", color="steelblue", linewidth=1.5)
axes[1].plot(thresholds, recalls[:-1],    label="Recall",    color="tomato",    linewidth=1.5)
axes[1].axvline(best_threshold, color="black", linestyle="--",
                label=f"Optimal = {best_threshold:.3f}")
```
Plots both curves against threshold on the same axes. The vertical line shows the exact trade-off point chosen: at this threshold, precision and recall are at specific values. Raising the threshold increases precision but drops recall; lowering it does the opposite.

---

### Complete Code Block — Section 11

```python
from sklearn.metrics import precision_recall_curve, fbeta_score, classification_report

proba_balanced = rf_balanced.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, proba_balanced)

beta = 2
f2_scores = (
    (1 + beta**2) * precisions[:-1] * recalls[:-1]
    / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-9)
)

best_idx       = f2_scores.argmax()
best_threshold = thresholds[best_idx]
best_f2        = f2_scores[best_idx]

print(f"Best threshold (F2-maximising): {best_threshold:.4f}")
print(f"F2 score at best threshold:     {best_f2:.4f}")

y_pred_optimal = (proba_balanced >= best_threshold).astype(int)

print(f"--- Balanced RF at default threshold (0.50) ---")
print(classification_report(y_test, y_pred_balanced, target_names=["no_default", "default"], digits=3))

print(f"--- Balanced RF at optimal threshold ({best_threshold:.4f}) ---")
print(classification_report(y_test, y_pred_optimal, target_names=["no_default", "default"], digits=3))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(thresholds, f2_scores, color="steelblue", linewidth=1.5)
axes[0].axvline(best_threshold, color="tomato", linestyle="--",
                label=f"Optimal threshold = {best_threshold:.3f}")
axes[0].set_xlabel("Decision Threshold")
axes[0].set_ylabel("F2 Score")
axes[0].set_title("F2 Score vs. Decision Threshold")
axes[0].legend()

axes[1].plot(thresholds, precisions[:-1], label="Precision", color="steelblue", linewidth=1.5)
axes[1].plot(thresholds, recalls[:-1],    label="Recall",    color="tomato",    linewidth=1.5)
axes[1].axvline(best_threshold, color="black", linestyle="--",
                label=f"Optimal = {best_threshold:.3f}")
axes[1].set_xlabel("Decision Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title("Precision & Recall vs. Decision Threshold")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Section 12 — Final Model Comparison

Aggregates all model variants into a single summary table and visualization.

---

### Line-by-Line

```python
from sklearn.metrics import recall_score, precision_score, f1_score
```
Imports individual metric functions (rather than the full classification report) so we can compute them programmatically and store results in a DataFrame.

```python
models = {
    "Decision Tree (max_depth=5)": y_pred_tree,
    "RF default":                  y_pred_rf,
    "RF balanced":                 y_pred_balanced,
    "RF + SMOTE":                  y_pred_smote,
    "RF + Feature Eng.":           y_pred_fe,
    "RF balanced + Optimal Threshold": y_pred_optimal,
}
```
Dictionary mapping each model's human-readable name to its test set predictions. This structure makes the loop below clean and extensible.

```python
rows = []
for name, preds in models.items():
    rows.append({
        "Model": name,
        "Accuracy":  round((preds == y_test).mean(), 4),
        "Precision (default)": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall (default)": round(recall_score(y_test, preds), 4),
        "F1 (default)": round(f1_score(y_test, preds), 4),
    })
```
Iterates over all six model variants and computes four metrics for each:
- **Accuracy** — proportion of all correct predictions (misleading alone for imbalanced data)
- **Precision** — of predicted defaults, how many were real (`zero_division=0` handles the edge case where no defaults are predicted)
- **Recall** — of real defaults, how many were caught
- **F1** — harmonic mean of precision and recall for the default class

```python
comparison_df = pd.DataFrame(rows)
print(comparison_df.to_string(index=False))
```
Assembles all rows into a DataFrame and prints a clean table. Scanning the Recall column from top to bottom shows the progression: each improvement technique should push recall higher for the default class.

```python
x       = range(len(comparison_df))
width   = 0.22
metrics = ["Precision (default)", "Recall (default)", "F1 (default)"]
colors  = ["steelblue", "tomato", "seagreen"]
```
Sets up the grouped bar chart geometry. `width=0.22` makes three bars fit side-by-side without overlap within each model's slot.

```python
for i, (metric, color) in enumerate(zip(metrics, colors)):
    offsets = [xi + i * width for xi in x]
    ax.bar(offsets, comparison_df[metric], width=width, label=metric,
           color=color, alpha=0.85, edgecolor="black")
```
Draws one bar group per metric, offsetting each group by `i * width` to avoid overlap. `zip(metrics, colors)` pairs each metric name with its designated color. `enumerate` tracks `i` for position offset.

```python
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(comparison_df["Model"], rotation=25, ha="right", fontsize=9)
```
Sets tick positions to the center of each bar group and rotates the labels 25° to prevent overlap.

---

### Complete Code Block — Section 12

```python
from sklearn.metrics import recall_score, precision_score, f1_score

models = {
    "Decision Tree (max_depth=5)":     y_pred_tree,
    "RF default":                       y_pred_rf,
    "RF balanced":                      y_pred_balanced,
    "RF + SMOTE":                       y_pred_smote,
    "RF + Feature Eng.":                y_pred_fe,
    "RF balanced + Optimal Threshold":  y_pred_optimal,
}

rows = []
for name, preds in models.items():
    rows.append({
        "Model": name,
        "Accuracy":  round((preds == y_test).mean(), 4),
        "Precision (default)": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall (default)": round(recall_score(y_test, preds), 4),
        "F1 (default)": round(f1_score(y_test, preds), 4),
    })

comparison_df = pd.DataFrame(rows)
print(comparison_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 6))

x       = range(len(comparison_df))
width   = 0.22
metrics = ["Precision (default)", "Recall (default)", "F1 (default)"]
colors  = ["steelblue", "tomato", "seagreen"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    offsets = [xi + i * width for xi in x]
    ax.bar(offsets, comparison_df[metric], width=width, label=metric,
           color=color, alpha=0.85, edgecolor="black")

ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(comparison_df["Model"], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Default Class Metrics")
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()
```

---

## Summary of Three Advanced Accuracy Techniques

| Technique | Where it acts | Key idea | Best for |
|---|---|---|---|
| **SMOTE** | Training data | Synthesises new minority samples by interpolating between real ones in feature space | When the minority class is severely underrepresented and needs a denser learned boundary |
| **Feature Engineering** | Feature space | Compresses domain knowledge (DTI ratio, composite risk score, income stability) into explicit columns the model can split on directly | When domain expertise can produce signals that raw features encode only implicitly |
| **Threshold Optimisation** | Post-training output | Replaces the default 0.5 cutoff with the threshold that maximises F2 (recall-weighted F-score) | When the deployment cost of false negatives and false positives are asymmetric — e.g., loan defaults cost more than false alarms |
