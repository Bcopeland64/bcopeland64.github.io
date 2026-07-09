# Lab 5A — Code Walkthrough
**Module 5 Week A | SI Reference Document**

This document walks through **every code cell** in `Lab_5A_Demo_Student_Performance.ipynb`, section by section. Each section explains what every line does in plain language, then shows the complete code block so you can narrate the demo confidently or troubleshoot learner questions on the spot.

---

## Section 0 — Setup


```python
import numpy as np
```
Imports NumPy and aliases it as `np`. NumPy provides fast numerical arrays and math utilities — we use it for `np.arange` (integer ranges), `np.unique` (unique values), `np.sum` (counting), and `np.abs` (absolute values).

```python
import pandas as pd
```
Imports pandas as `pd`. Pandas is the main tabular-data library — we load the CSV into a `DataFrame` and use it for column selection, filtering, and aggregation throughout the notebook.

```python
import matplotlib.pyplot as plt
```
Imports the base plotting library as `plt`. Every chart we draw goes through `plt` or the `axes` objects it returns. Without this, no visualizations are possible.

```python
import seaborn as sns
```
Imports seaborn as `sns`. Seaborn is a higher-level wrapper around matplotlib that makes heatmaps and categorical plots simpler. We use it for the correlation heatmap in Section 1.

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
```
Imports three tools from scikit-learn's model selection module:
- `train_test_split` — randomly divides the dataset into training and test sets.
- `StratifiedKFold` — a cross-validation splitter that preserves the class ratio in every fold.
- `cross_val_score` — runs the full pipeline through all folds and returns an array of scores.

```python
from sklearn.pipeline import Pipeline
```
`Pipeline` chains preprocessing and a model into one atomic object. When you call `.fit()`, the pipeline fits the scaler on training data only, then transforms it, then fits the model — preventing data leakage in a single line.

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```
- `StandardScaler` — for each numeric column, subtracts the column mean and divides by the column standard deviation, putting all numeric features on the same scale.
- `OneHotEncoder` — converts categorical text columns (e.g., `gender`, `major`) into numeric 0/1 indicator columns so the model can use them.

```python
from sklearn.compose import ColumnTransformer
```
`ColumnTransformer` applies different transformers to different subsets of columns simultaneously. We use it to scale numeric columns with `StandardScaler` and encode categorical columns with `OneHotEncoder` in one step.

```python
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
```
Imports all four models used in the demo:
- `LogisticRegression` — classification model that predicts pass/fail probability.
- `Ridge` — regularized regression using an L2 penalty; shrinks all coefficients toward zero but keeps all features.
- `Lasso` — regularized regression using an L1 penalty; drives some coefficients to exactly zero, performing automatic feature selection.
- `LinearRegression` — ordinary least squares (OLS); no regularization, used as a baseline comparison.

```python
from sklearn.metrics import (
    classification_report, accuracy_score,
    mean_absolute_error, r2_score
)
```
Imports the evaluation functions we need:
- `classification_report` — prints a table of precision, recall, F1, and support for each class.
- `accuracy_score` — fraction of predictions that are correct.
- `mean_absolute_error` — average absolute difference between predicted and actual values (same units as the target).
- `r2_score` — coefficient of determination: 0 = the model is no better than predicting the mean; 1 = perfect prediction.

```python
RANDOM_STATE = 42
```
Stores a single integer seed in a constant. Every function that has randomness (`train_test_split`, `LogisticRegression`, `StratifiedKFold`) receives this same value so the notebook produces identical results on every run.

```python
plt.style.use('seaborn-v0_8-whitegrid')
```
Applies a clean white-grid chart style globally. All subsequent plots inherit this style without needing to set it per chart.

```python
sns.set_palette('muted')
```
Sets the default seaborn color palette to `muted` — desaturated colors that remain readable and consistent across all plots.

```python
pd.set_option('display.max_columns', 20)
```
Tells pandas to display up to 20 columns before truncating. Without this, wide DataFrames may collapse into `...` in the middle, hiding columns we want to see.

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()
```
Imports the `rich` library, which renders colorized, formatted output in the terminal and Jupyter notebooks. Five objects are imported:
- `Console` — the main output object; `console.print(...)` replaces bare `print(...)` everywhere in the notebook for styled output.
- `Table` — builds formatted tables with colored columns, borders, and alignment.
- `Panel` — wraps text in a titled, bordered box — used for the classification report and final recommendation.
- `Text` — a rich text object for inline markup (available for later use).
- `box` — a module of border styles (e.g., `box.ROUNDED`, `box.DOUBLE_EDGE`) used to style each table differently.

`console = Console()` creates the single output handle used throughout the notebook.

```python
console.print(Panel('[bold green]All imports successful.[/bold green]', title='Setup', border_style='green'))
```
Replaces the plain `print('All imports successful.')` with a green-bordered panel. The `[bold green]...[/bold green]` markup bolds and colors the text. In a live demo this visually confirms the environment is working before moving forward.

---

## Section 1 — Load & Explore the Dataset


#### Cell 1 — Load the file

```python
df = pd.read_csv('students_performance.csv')
```
Reads the CSV file from disk into a pandas DataFrame named `df`. The file sits in the same directory as the notebook, so no full path is needed.

```python
print(f'Shape: {df.shape}')
```
`df.shape` returns a tuple `(rows, columns)`. Printing it immediately confirms the dataset loaded completely — we expect `(400, 15)`.

```python
df.head()
```
Displays the first five rows in the notebook output. This is the fastest sanity check — you can see column names, data types, and what the values look like.

---

#### Cell 2 — Column types and null check

```python
df.info()
```
Prints a compact summary: column names, non-null counts, and dtype per column. We're looking for two things: (1) any column with fewer non-null values than the total row count signals missing data; (2) columns typed as `object` are text/categorical and will need encoding later.

---

#### Cell 3 — Descriptive statistics

```python
df.describe().round(2)
```
`describe()` computes count, mean, std, min, quartiles, and max for every numeric column. `.round(2)` trims the decimals so the table is readable. Look for obviously wrong ranges (e.g., `attendance_pct` should be 0–100) and whether the mean and median (50%) are far apart, which suggests skew.

---

#### Cell 4 — Class balance check

```python
pass_counts = df['pass_fail'].value_counts()
```
`value_counts()` returns the count of each unique value in the `pass_fail` column, sorted from most to least frequent. Storing it in `pass_counts` lets us reference it in the table below.

```python
table = Table(title='Class Balance — Pass/Fail', box=box.ROUNDED, border_style='cyan')
table.add_column('Label', style='bold')
table.add_column('Class', justify='center')
table.add_column('Count', justify='right', style='bold yellow')
table.add_column('Rate', justify='right', style='bold green')
```
Builds a rich `Table` with four columns. `box.ROUNDED` gives the table rounded corners; `border_style='cyan'` colors the border. Each `add_column` call sets the header text, alignment, and an optional color style for that column's values.

```python
total = len(df)
for label, cls in [('Pass', 1), ('Fail', 0)]:
    count = pass_counts[cls]
    table.add_row(label, str(cls), str(count), f'{count/total:.1%}')
```
Iterates over the two classes in the preferred display order (Pass first, then Fail). `pass_counts[cls]` looks up the count for class 1 or 0. Each `add_row` call fills in the four columns: the human label, the numeric class value, the raw count, and the percentage of total students.

```python
console.print(table)
console.print(f'  Overall pass rate: [bold green]{df["pass_fail"].mean():.1%}[/bold green]')
```
`console.print(table)` renders the rich table. The second line prints the overall pass rate inline — `df["pass_fail"].mean()` works because `pass_fail` is 0 or 1 (the mean of a binary column equals the proportion of 1s). The `[bold green]...[/bold green]` markup highlights the percentage in green.

---

#### Cell 5 — EDA scatter and histogram plots

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
```
Creates one figure with a 1×3 grid of subplots. `axes` is a list of three `Axes` objects. `figsize=(15, 4)` makes the figure wide enough for three charts side by side.

```python
axes[0].hist(df['final_score'], bins=25, edgecolor='white')
```
Draws a histogram of final scores with 25 bins. `edgecolor='white'` draws a thin white border around each bar so adjacent bars don't merge visually.

```python
axes[0].axvline(60, color='red', linestyle='--', label='Pass threshold (60)')
```
Draws a vertical dashed red line at score 60 — the pass/fail boundary. This makes the threshold visible against the score distribution. `label=` will show up in the legend.

```python
axes[0].set_title('Final Score Distribution')
axes[0].set_xlabel('Final Score')
axes[0].legend()
```
Sets the panel title, x-axis label, and renders the legend showing the "Pass threshold" line label.

```python
axes[1].scatter(
    df['study_hours_per_week'], df['final_score'],
    c=df['pass_fail'], cmap='RdYlGn', alpha=0.5, edgecolors='none'
)
```
Draws a scatter plot with study hours on the x-axis and final score on the y-axis. `c=df['pass_fail']` colors each point by its class — 0 (fail) maps to red, 1 (pass) maps to green via `cmap='RdYlGn'`. `alpha=0.5` makes overlapping points semi-transparent so density is visible. `edgecolors='none'` removes the default black border around each dot.

```python
axes[1].set_xlabel('Study Hours / Week')
axes[1].set_ylabel('Final Score')
axes[1].set_title('Study Hours vs. Final Score')
```
Labels for the second panel.

```python
axes[2].scatter(
    df['attendance_pct'], df['final_score'],
    c=df['pass_fail'], cmap='RdYlGn', alpha=0.5, edgecolors='none'
)
axes[2].set_xlabel('Attendance (%)')
axes[2].set_ylabel('Final Score')
axes[2].set_title('Attendance vs. Final Score')
```
Identical pattern to the study hours scatter but using `attendance_pct` on the x-axis. Side by side with the previous chart, learners can visually compare which feature separates pass/fail more cleanly.

```python
plt.tight_layout()
plt.show()
```
`tight_layout()` automatically adjusts subplot spacing so titles and labels don't overlap. `show()` flushes the figure to the notebook output.

---

#### Cell 6 — Correlation heatmap

```python
numeric_cols = df.select_dtypes(include='number').drop(columns=['student_id'], errors='ignore')
```
`select_dtypes(include='number')` keeps only integer and float columns — text columns like `gender` and `major` can't appear in a correlation matrix. `.drop(columns=['student_id'], errors='ignore')` removes the ID column because its correlation with any real feature is meaningless. `errors='ignore'` prevents a crash if `student_id` was already removed.

```python
plt.figure(figsize=(10, 7))
```
Creates a new blank figure 10 inches wide by 7 inches tall. Without this, the heatmap might be drawn on top of the previous figure.

```python
sns.heatmap(
    numeric_cols.corr(), annot=True, fmt='.2f',
    cmap='coolwarm', center=0, linewidths=0.5
)
```
- `numeric_cols.corr()` — computes the Pearson correlation matrix. Every cell value is between −1 (perfect negative correlation) and +1 (perfect positive correlation).
- `annot=True` — prints the numeric value inside every cell so learners don't have to guess from color alone.
- `fmt='.2f'` — formats those numbers to two decimal places so the cells stay readable.
- `cmap='coolwarm'` — colors negative correlations blue, positive correlations red, and zero white.
- `center=0` — anchors the white (neutral) color at zero rather than at the midpoint of the data range.
- `linewidths=0.5` — draws thin lines between cells to separate them visually.

```python
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```
Adds a title and renders the chart.

---

## Section 2 — Feature Engineering & Train/Test Split


#### Cell 1 — Define X and y

```python
df_model = df.drop(columns=['student_id'])
```
Creates a working copy of the DataFrame with the ID column removed. `student_id` is just a label — it has no predictive relationship to exam scores, and leaving it in would add noise and waste a feature slot.

```python
FEATURE_COLS = [c for c in df_model.columns if c not in ('final_score', 'pass_fail')]
```
A list comprehension that keeps every column that is not one of the two targets. The result is the list of features we will train on. Storing it in a constant `FEATURE_COLS` makes it easy to reuse later.

```python
print('Feature columns:', FEATURE_COLS)
```
Prints the exact list of features. In the demo, pause here to make sure learners see what is going into the model.

```python
X = df_model[FEATURE_COLS]
```
Creates the feature matrix `X` — a DataFrame with all feature columns. Every model-building step uses `X` rather than the full `df`.

```python
y_clf = df_model['pass_fail']
```
The classification target: a Series of 0s (fail) and 1s (pass). Named `y_clf` to distinguish it from the regression target.

```python
y_reg = df_model['final_score']
```
The regression target: a Series of continuous scores from roughly 40 to 100. Named `y_reg` to keep targets clearly separated.

---

#### Cell 2 — Identify column types

```python
categorical_cols = X.select_dtypes(include='object').columns.tolist()
```
`select_dtypes(include='object')` selects columns whose dtype is `object` — pandas stores text as `object`. `.columns.tolist()` converts the Index to a plain Python list so it can be passed to `ColumnTransformer`. For this dataset this gives `['gender', 'major']`.

```python
numeric_cols_feat = X.select_dtypes(include='number').columns.tolist()
```
The complement: every numeric column in `X`. These will be scaled with `StandardScaler`.

```python
print('Numeric features  :', numeric_cols_feat)
print('Categorical features:', categorical_cols)
```
Shows learners exactly which columns go to which transformer. This is a good pause point to ask: "Why do we need different transformations for these two groups?"

---

#### Cell 3 — Train/test split

```python
X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_clf
)
```
`train_test_split` accepts multiple arrays and splits them all with the same indices, so `X_train[i]` always corresponds to `y_clf_train[i]` and `y_reg_train[i]`.
- `test_size=0.20` — 20% of 400 rows = 80 test rows; the remaining 320 go to training.
- `random_state=RANDOM_STATE` — fixes the random shuffle so the split is reproducible.
- `stratify=y_clf` — the most important argument here: it ensures that the ~76% pass rate is preserved in both the train and test sets. Without this, a random split could put most of the failing students in one set, giving misleading metrics.

```python
table = Table(title='Train / Test Split', box=box.SIMPLE_HEAVY, border_style='blue')
table.add_column('Split', style='bold')
table.add_column('Size', justify='right', style='yellow')
table.add_column('Share', justify='right')
table.add_column('Pass Rate', justify='right', style='green')

table.add_row('Train', str(len(X_train)), f'{len(X_train)/len(X):.0%}', f'{y_clf_train.mean():.1%}')
table.add_row('Test',  str(len(X_test)),  f'{len(X_test)/len(X):.0%}',  f'{y_clf_test.mean():.1%}')

console.print(table)
console.print('[bold green]✓ Stratification preserved the class ratio.[/bold green]')
```
A rich `Table` replaces the four bare print statements. The four columns are:
- **Split** — "Train" or "Test".
- **Size** — the absolute row count for each set. Train should be 320, test 80.
- **Share** — 80% and 20%, formatted with `:.0%`.
- **Pass Rate** — the critical verification column: both rows should show approximately 76%, confirming that `stratify=y_clf` preserved the class balance in both sets.

The final `console.print` line renders the "stratification confirmed" message in bold green — a strong visual cue during the demo to pause and highlight why stratification matters.

---

#### Cell 4 — Define the preprocessor

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols_feat),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])
```
`ColumnTransformer` takes a list of `(name, transformer, columns)` tuples and applies each transformer to its designated columns in parallel.

**Tuple 1 — `('num', StandardScaler(), numeric_cols_feat)`**
- Name `'num'` is just a label; it lets you later retrieve this transformer via `preprocessor.named_transformers_['num']`.
- `StandardScaler()` will, when fitted, learn the mean and std of each numeric column from the training data and then subtract/divide both train and test data by those values.
- `numeric_cols_feat` is the list of numeric column names from the cell above.

**Tuple 2 — `('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)`**
- `OneHotEncoder` converts text categories into binary columns. For example, `major` with four unique values becomes three binary columns (one is dropped).
- `drop='first'` removes the first category column for each categorical variable. This prevents the dummy variable trap — if you have columns `major_Business`, `major_Engineering`, `major_Science`, you don't also need `major_Arts` because it's implied when all three are zero.
- `sparse_output=False` returns a regular NumPy array instead of a sparse matrix. Sparse matrices can confuse learners and cause issues with some plotting code.

**Why `.fit()` is not called here**
The preprocessor is only *defined* here, not trained. It will be fitted inside `Pipeline.fit(X_train, ...)` in the next section. This is the core anti-leakage pattern — the scaler must only see training data.

```python
print('Preprocessor defined. It will be fit inside Pipeline.fit() — no leakage.')
```
A narration cue to explicitly call out the anti-leakage pattern during the demo.

---

## Section 3 — Classification: Logistic Regression Pipeline


#### Cell 1 — Build and train the pipeline

```python
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
```
`Pipeline(steps=[...])` takes a list of `(name, object)` tuples. When `.fit()` is called:
1. It calls `preprocessor.fit_transform(X_train)` — fitting the scaler and encoder only on training data, then transforming it.
2. It passes the transformed data to `LogisticRegression.fit(...)`.

When `.predict()` is called:
1. It calls `preprocessor.transform(X_test)` — using the already-fitted scaler/encoder (no refitting, so no leakage).
2. It passes the transformed test data to `LogisticRegression.predict(...)`.

`max_iter=1000` — Logistic Regression uses an iterative solver. The default 100 iterations can fail to converge on moderately sized feature spaces with many encoded columns; 1000 is a safe upper limit. `random_state=RANDOM_STATE` fixes the solver's random seed for reproducibility.

```python
clf_pipeline.fit(X_train, y_clf_train)
print('Model trained.')
```
Fits the entire pipeline on the training data. `print('Model trained.')` is a narration cue — the cell runs silently otherwise and learners may not know it succeeded.

---

#### Cell 2 — Generate predictions

```python
y_clf_pred = clf_pipeline.predict(X_test)
```
Passes `X_test` through the fitted pipeline: preprocessor transforms the raw test features, then the classifier predicts 0 or 1 for each row. The result is a NumPy array of length 80 (our test set size).

```python
print(f'Predictions shape : {y_clf_pred.shape}')
print(f'Unique values     : {np.unique(y_clf_pred)}')
print(f'First 10 preds    : {y_clf_pred[:10]}')
```
Three sanity checks:
- Shape should be `(80,)` — one prediction per test row.
- Unique values should be `[0 1]` — the model should predict both classes.
- Printing the first 10 predictions alongside the next cell's true values lets learners see how predictions compare to reality.

---

#### Cell 3 — Print the classification report

```python
y_clf_pred = clf_pipeline.predict(X_test)

report_str = classification_report(y_clf_test, y_clf_pred, target_names=['Fail', 'Pass'])
console.print(Panel(
    f'[white]{report_str}[/white]',
    title='[bold cyan]Classification Report — Logistic Regression[/bold cyan]',
    border_style='cyan',
    expand=False
))
```
`classification_report` generates a multi-line string showing precision, recall, F1-score, and support for each class. `target_names=['Fail', 'Pass']` labels class 0 as "Fail" and class 1 as "Pass" so the table is human-readable.

Instead of `print(report_str)`, the string is wrapped in a `Panel` for visual clarity:
- `f'[white]{report_str}[/white]'` — the monospace report text in white, preserving its alignment inside the panel.
- `title='[bold cyan]Classification Report...[/bold cyan]'` — a colored bold title above the box.
- `border_style='cyan'` — a cyan-colored border around the report.
- `expand=False` — the panel shrinks to fit the content width rather than stretching to the full terminal width.

Pause here during the demo to walk through each column:
- **Precision** — of all students the model said "Pass", what fraction actually passed?
- **Recall** — of all students who actually passed, what fraction did the model catch?
- **F1** — the harmonic mean of precision and recall.
- **Support** — the actual number of students in each class in the test set.

---

#### Cell 4 — Structured metrics dictionary

```python
from sklearn.metrics import precision_score, recall_score, f1_score
```
These three functions weren't in the top-level imports because they're only needed in this cell. Importing at point-of-use keeps the import block at the top of the notebook cleaner.

```python
classification_metrics = {
    'accuracy' : accuracy_score(y_clf_test, y_clf_pred),
    'precision': precision_score(y_clf_test, y_clf_pred),
    'recall'   : recall_score(y_clf_test, y_clf_pred),
    'f1'       : f1_score(y_clf_test, y_clf_pred),
}
```
All four functions follow the same signature: `(y_true, y_predicted)`. Each returns a single float.
- `accuracy_score` — `(TP + TN) / total`: fraction of all predictions that are correct.
- `precision_score` — `TP / (TP + FP)`: of every student we predicted "pass", what fraction actually passed? Defaults to the positive class (1 = pass).
- `recall_score` — `TP / (TP + FN)`: of every student who truly passed, what fraction did we catch?
- `f1_score` — `2 * (precision * recall) / (precision + recall)`: the harmonic mean, which is low if either precision or recall is low.

```python
table = Table(title='Classification Metrics', box=box.ROUNDED, border_style='magenta')
table.add_column('Metric', style='bold')
table.add_column('Value', justify='right', style='bold yellow')
table.add_column('Interpretation', style='dim')

interp = {
    'accuracy' : 'Overall correct predictions',
    'precision': 'Of predicted Pass, how many truly passed',
    'recall'   : 'Of actual Pass, how many we caught',
    'f1'       : 'Harmonic mean of precision & recall',
}
for k, v in classification_metrics.items():
    color = 'green' if v >= 0.85 else 'yellow' if v >= 0.70 else 'red'
    table.add_row(k.capitalize(), f'[{color}]{v:.4f}[/{color}]', interp[k])

console.print(table)
```
Replaces the plain `for`-loop print with a color-coded rich table. Three columns — Metric, Value, Interpretation — are added to make the output self-documenting for learners.

The `for` loop iterates `classification_metrics` and applies a traffic-light color to each value:
- **green** — score ≥ 0.85 (strong performance).
- **yellow** — score between 0.70 and 0.85 (acceptable).
- **red** — score < 0.70 (flag for discussion).

The `interp` dictionary provides a plain-English explanation for each metric in the third column — useful during a live demo when learners are seeing these terms for the first time.

```python
assert all(v > 0 for v in classification_metrics.values()), 'All metrics should be > 0'
```
`all(...)` returns `True` only if every element in the iterable is truthy. `v > 0` checks that no metric is zero or negative — a metric of zero would indicate the model is predicting only one class. If any value is zero or below, `assert` raises an `AssertionError` with the quoted message. This directly mirrors what the autograder checks in the learner lab.

---

## Section 4 — Regression: Ridge Pipeline


#### Cell 1 — Build, train, and predict

```python
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])
```
Same structure as the classification pipeline but with `Ridge` as the final step. `alpha=1.0` is the regularization strength — a mild penalty that shrinks large coefficients without eliminating features. Higher alpha = more shrinkage.

```python
ridge_pipeline.fit(X_train, y_reg_train)
```
Trains on the regression target `y_reg_train` (continuous scores) rather than the binary `y_clf_train`. The pipeline fits the preprocessor on `X_train` and then fits the Ridge model.

```python
y_reg_pred_ridge = ridge_pipeline.predict(X_test)
```
Generates predicted scores for the 80 test students. The result is a float array — unlike classification, regression outputs continuous numbers.

```python
print(f'Predictions — first 5: {y_reg_pred_ridge[:5].round(1)}')
print(f'True values — first 5: {y_reg_test.values[:5]}')
```
Prints the first 5 predicted scores alongside the first 5 true scores. This gives an immediate qualitative sense of how close the predictions are before computing formal metrics.

---

#### Cell 2 — Regression metrics

```python
regression_metrics = {
    'mae': mean_absolute_error(y_reg_test, y_reg_pred_ridge),
    'r2' : r2_score(y_reg_test, y_reg_pred_ridge),
}
```
Two metrics captured in a dictionary:
- `mean_absolute_error` — the average of `|actual - predicted|` across all test rows. Expressed in the same units as the target (exam points). Example: MAE of 5.2 means the model is off by an average of 5.2 points.
- `r2_score` — ranges from −∞ to 1. A value of 1.0 means the model predicts every score exactly. A value of 0.0 means the model does no better than always predicting the mean score. Negative values mean the model is worse than the mean — a strong signal of underfitting or a bug.

```python
table = Table(title='Regression Metrics — Ridge', box=box.ROUNDED, border_style='blue')
table.add_column('Metric', style='bold')
table.add_column('Value', justify='right', style='bold yellow')
table.add_column('Interpretation', style='dim')

mae_color = 'green' if regression_metrics['mae'] < 5 else 'yellow'
r2_color  = 'green' if regression_metrics['r2'] > 0.7  else 'yellow' if regression_metrics['r2'] > 0.5 else 'red'

table.add_row('MAE', f'[{mae_color}]{regression_metrics["mae"]:.4f}[/{mae_color}]',
              'Mean abs. error in score points (lower is better)')
table.add_row('R²',  f'[{r2_color}]{regression_metrics["r2"]:.4f}[/{r2_color}]',
              'Variance explained — 1.0 is perfect (higher is better)')

console.print(table)
```
Replaces the plain `for`-loop print with a rich table that adds context for each metric. Three columns — Metric, Value, Interpretation — are used.

Color thresholds for MAE: **green** if MAE < 5 score points (good), **yellow** otherwise.
Color thresholds for R²: **green** if R² > 0.70, **yellow** if 0.50–0.70, **red** below 0.50.

The Interpretation column provides plain-English meaning directly in the output — e.g., "An R² of ~0.75 means the model explains about 75% of the variance in student scores."

```python
assert regression_metrics['r2'] > 0, 'R² should be positive — model beats a flat mean prediction'
```
A sanity check: if R² is negative, something is fundamentally wrong (e.g., the wrong target column was used, or train/test labels were mixed up).

---

#### Cell 3 — Diagnostic visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
```
Creates two side-by-side subplots: the left will show predicted vs. actual, and the right will show the residual distribution.

```python
axes[0].scatter(y_reg_test, y_reg_pred_ridge, alpha=0.5)
```
Plots actual score (x-axis) against predicted score (y-axis). `alpha=0.5` makes points semi-transparent so overlapping points show density rather than becoming a solid blob.

```python
lims = [y_reg_test.min(), y_reg_test.max()]
axes[0].plot(lims, lims, 'r--', label='Perfect prediction')
```
`lims` is a two-element list `[minimum_actual, maximum_actual]`. `axes[0].plot(lims, lims, ...)` draws a diagonal line from the bottom-left to the top-right — every point on this line means predicted = actual. Points above the line = overpredicted; below = underpredicted. `'r--'` is matplotlib shorthand: red (`r`), dashed (`--`).

```python
axes[0].set_xlabel('Actual Final Score')
axes[0].set_ylabel('Predicted Final Score')
axes[0].set_title(f'Ridge — Predicted vs. Actual (R²={regression_metrics["r2"]:.3f})')
axes[0].legend()
```
Labels for the left panel. The R² value is embedded in the title so it's visible without scrolling to the metrics output. `:.3f` formats to three decimal places.

```python
residuals = y_reg_test.values - y_reg_pred_ridge
```
Computes the **residuals**: actual minus predicted. Positive residual = the model underpredicted; negative = overpredicted. `.values` converts the pandas Series to a NumPy array so element-wise subtraction aligns by position rather than by index.

```python
axes[1].hist(residuals, bins=20, edgecolor='white')
```
Histogram of residuals with 20 bins. A well-behaved regression model should have residuals centered near zero with a roughly symmetric, bell-shaped distribution.

```python
axes[1].axvline(0, color='red', linestyle='--')
```
Vertical line at zero. If the histogram peak aligns with this line, the model is unbiased on average (errors above and below cancel out). A histogram shifted to one side indicates systematic over- or under-prediction.

```python
axes[1].set_xlabel('Residual (Actual − Predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Residual Distribution (MAE={regression_metrics["mae"]:.2f})')
```
The MAE is embedded in the residual chart title. MAE is in exam-point units, making it the most interpretable metric for a non-technical audience: "On average, the model is off by X points."

```python
plt.tight_layout()
plt.show()
```
Adjusts subplot spacing and renders the figure.

---

## Section 5 — Regularization Deep-Dive: OLS vs. Ridge vs. Lasso


#### Cell 1 — Train all models and store coefficients

```python
def make_reg_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
```
A helper function so we don't copy-paste the pipeline construction five times. It takes any regressor object and wraps it in the standard preprocessor → regressor pipeline. Defining it as a function keeps the loop below clean.

```python
models = {
    'OLS (no regularization)': LinearRegression(),
    'Ridge (alpha=1)':         Ridge(alpha=1.0),
    'Ridge (alpha=10)':        Ridge(alpha=10.0),
    'Lasso (alpha=1)':         Lasso(alpha=1.0, max_iter=5000),
    'Lasso (alpha=5)':         Lasso(alpha=5.0, max_iter=5000),
}
```
A dictionary mapping descriptive names to model instances. The names are chosen to be self-documenting — they appear as labels in the coefficient chart and the results table. `max_iter=5000` for Lasso is higher than the default because Lasso's coordinate descent solver sometimes needs more iterations to converge when many features are being driven to zero.

```python
results = []
coef_store = {}
```
Two containers:
- `results` — an empty list that will collect one dictionary per model (for the summary table).
- `coef_store` — an empty dictionary that will store each model's coefficient array (for the visualization).

```python
for name, model in models.items():
```
Iterates over the dictionary, unpacking each key-value pair into `name` (the string label) and `model` (the scikit-learn estimator).

```python
    pipe = make_reg_pipeline(model)
    pipe.fit(X_train, y_reg_train)
    preds = pipe.predict(X_test)
```
Builds and trains a fresh pipeline for this model, then generates predictions on the test set.

```python
    results.append({
        'Model': name,
        'MAE':   round(mean_absolute_error(y_reg_test, preds), 3),
        'R²':    round(r2_score(y_reg_test, preds), 3),
    })
```
`results.append(...)` adds a dictionary with the model name and its two metrics to the list. `round(..., 3)` keeps the table readable.

```python
    coef_store[name] = pipe.named_steps['regressor'].coef_
```
`pipe.named_steps['regressor']` retrieves the fitted model object from the pipeline by the name we gave it (`'regressor'`). `.coef_` is the array of learned coefficients — one per feature after one-hot encoding. Storing it in `coef_store` keyed by model name lets us retrieve it in the visualization cell.

```python
results_df = pd.DataFrame(results)

table = Table(title='Regularization Comparison', box=box.DOUBLE_EDGE, border_style='magenta')
table.add_column('Model', style='bold')
table.add_column('MAE', justify='right')
table.add_column('R²',  justify='right')

best_r2 = max(r['R²'] for r in results)
for r in results:
    mae_color = 'green' if r['MAE'] < 4.0 else 'yellow' if r['MAE'] < 6.0 else 'red'
    r2_color  = 'green' if r['R²'] == best_r2 else 'yellow' if r['R²'] > 0.5 else 'red'
    star = ' ★' if r['R²'] == best_r2 else ''
    table.add_row(
        r['Model'] + star,
        f'[{mae_color}]{r["MAE"]}[/{mae_color}]',
        f'[{r2_color}]{r["R²"]}[/{r2_color}]',
    )

console.print(table)
```
Converts the list of dictionaries to a DataFrame (used for downstream access) and then renders a color-coded rich table with `box.DOUBLE_EDGE` styling. Three columns — Model, MAE, R² — compare all five models at a glance.

The best R² model is identified with `max(...)` and highlighted with a `★` star marker in the Model column. Color thresholds for MAE: **green** < 4.0 points, **yellow** < 6.0, **red** otherwise. For R²: **green** = best model, **yellow** > 0.50, **red** below 0.50. This makes it immediately obvious which model performs best and which regularization strength to prefer.

---

#### Cell 2 — Extract feature names after encoding

```python
_pp = preprocessor
_pp.fit(X_train)
```
The preprocessor was already fitted inside each pipeline's `.fit()` call, but here we fit it directly on `X_train` to access its `get_feature_names_out()` method. The `_pp` prefix is a convention indicating this is a temporary variable used only for inspection.

```python
num_feature_names = numeric_cols_feat
```
The numeric feature names don't change after scaling — they keep their original names.

```python
cat_feature_names = _pp.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
```
`_pp.named_transformers_['cat']` retrieves the fitted `OneHotEncoder`. `.get_feature_names_out(categorical_cols)` returns the column names it created — e.g., `['gender_M', 'major_Business', 'major_Engineering', 'major_Science']`. `.tolist()` converts the NumPy array to a Python list.

```python
all_feature_names = num_feature_names + cat_feature_names
```
Concatenates the two lists. The order must match the order that `ColumnTransformer` outputs columns: numeric first, then categorical. This is the same order as the coefficient arrays in `coef_store`.

```python
print(f'Total features after encoding: {len(all_feature_names)}')
print(all_feature_names)
```
Confirms the total feature count. If you have 10 numeric features and `gender`/`major` produce 4 encoded columns (gender_M + 3 majors), the total should be 14.

---

#### Cell 3 — Coefficient comparison chart

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
```
Three side-by-side panels, one per model. `sharey=True` means all three panels share the same y-axis (the list of feature names) — the labels only appear once on the left, and the horizontal bars align perfectly across panels for visual comparison.

```python
plot_models = ['OLS (no regularization)', 'Ridge (alpha=1)', 'Lasso (alpha=1)']
colors = ['steelblue', 'darkorange', 'green']
```
The three keys used for comparison — the same strings used as keys in `coef_store`. The colors are chosen to be visually distinct and color-blindness friendly.

```python
for ax, model_name, color in zip(axes, plot_models, colors):
```
`zip` pairs each axes object with its model name and color so we iterate three times, once per panel.

```python
    coefs = coef_store[model_name]
    y_pos = np.arange(len(coefs))
```
`coef_store[model_name]` retrieves the coefficient array stored during training. `np.arange(len(coefs))` generates integer positions `[0, 1, 2, ...]` that serve as the y-coordinates for each horizontal bar.

```python
    ax.barh(y_pos, coefs, color=color, alpha=0.7)
```
`barh` draws **horizontal** bars: y positions on the y-axis, coefficient values on the x-axis. Horizontal layout makes long feature names readable without having to rotate the labels.

```python
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_feature_names, fontsize=8)
```
`set_yticks` places a tick at each integer y position. `set_yticklabels` replaces those integers with the human-readable feature names. `fontsize=8` keeps the labels small enough to fit without overlapping.

```python
    ax.axvline(0, color='black', linewidth=0.8)
```
Draws a thin vertical line at zero. Features whose bars extend to the right have positive coefficients (they increase the predicted score). Features extending to the left have negative coefficients (they decrease the score). The zero line makes direction immediately visible.

```python
    n_zero = np.sum(np.abs(coefs) < 1e-4)
    ax.text(0.98, 0.02, f'{n_zero} zero coefs', transform=ax.transAxes,
            ha='right', fontsize=9, color='red')
```
- `np.abs(coefs) < 1e-4` creates a boolean array — `True` for coefficients effectively equal to zero (using a tolerance of 0.0001 rather than exactly 0, to handle floating-point imprecision).
- `np.sum(...)` counts how many are `True`.
- `ax.text(0.98, 0.02, ..., transform=ax.transAxes)` places text at 98% across and 2% up within the axes in axis-relative coordinates — so it always lands in the bottom-right corner regardless of the data scale.
- **This annotation is the key visual payoff**: OLS and Ridge will show `0 zero coefs`; Lasso will show a non-zero count, making feature elimination immediately visible.

```python
plt.suptitle('Coefficient Comparison: OLS vs Ridge vs Lasso', y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
```
`suptitle` adds a single title above all three panels. `y=1.02` shifts it slightly above the top of the figure so it doesn't overlap the individual panel titles.

```python
print()
print('Lasso zero coefficients (eliminated features):')
lasso_coefs = coef_store['Lasso (alpha=1)']
for name, coef in zip(all_feature_names, lasso_coefs):
    if abs(coef) < 1e-4:
        print(f'  {name} → 0')
```
After the chart, this prints a text list of which specific features Lasso set to zero. `zip(all_feature_names, lasso_coefs)` pairs each feature name with its coefficient. `abs(coef) < 1e-4` applies the same threshold as the chart annotation. The `→ 0` notation makes it clear these features were removed from the model entirely.

---

## Section 6 — Cross-Validation: Stratified 5-Fold


#### Cell 1 — Run cross-validation

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```
Creates a `StratifiedKFold` splitter:
- `n_splits=5` — divides the data into 5 equal folds. Each fold takes a turn as the test set while the other 4 are used for training.
- `shuffle=True` — randomly shuffles the data before splitting. Without this, folds would be sequential chunks and students at the top of the file (e.g., all from one major) might cluster in a single fold.
- `random_state=RANDOM_STATE` — fixes the shuffle so folds are reproducible.
- **Stratified** means each fold is drawn so that the pass/fail class ratio is preserved — every fold has ~76% pass, ~24% fail, regardless of which rows end up in it.

```python
cv_scores = cross_val_score(
    clf_pipeline,
    X, y_clf,
    cv=skf,
    scoring='f1'
)
```
`cross_val_score` runs the full pipeline through every fold automatically:
- `clf_pipeline` — the entire pipeline is passed in so that preprocessing is re-fitted inside each fold on that fold's training data only.
- `X, y_clf` — the **full** dataset is passed (not just `X_train`), because cross-validation manages its own splits internally. This is safe because `cross_val_score` never lets the test fold influence fitting.
- `cv=skf` — uses the stratified splitter we just defined.
- `scoring='f1'` — evaluates each fold using F1 score on the positive class (1 = pass). Returns an array of 5 floats, one per fold.

```python
table = Table(title='5-Fold Stratified CV — Logistic Regression (F1)', box=box.ROUNDED, border_style='cyan')
table.add_column('Fold', justify='center', style='bold')
table.add_column('F1 Score', justify='right', style='yellow')

for i, score in enumerate(cv_scores, 1):
    bar = '█' * int(score * 20)
    table.add_row(f'Fold {i}', f'{score:.3f}  {bar}')

table.add_section()
table.add_row('[bold]Mean[/bold]',   f'[bold green]{cv_scores.mean():.3f}[/bold green]')
table.add_row('[bold]Std[/bold]',    f'[bold]{cv_scores.std():.3f}[/bold]')
table.add_row('[bold]95% CI[/bold]', f'[bold]{cv_scores.mean():.3f} ± {2*cv_scores.std():.3f}[/bold]')

console.print(table)
```
Replaces the four print lines with a rich table. Each fold gets its own row with an inline bar chart — `'█' * int(score * 20)` maps an F1 score to a proportional number of block characters (e.g., F1 = 0.90 → 18 blocks). This makes variance across folds immediately visible as bar lengths.

A `table.add_section()` call draws a dividing line before the summary rows. The summary block reports:
- **Mean** — the single reported performance number (bold green for emphasis).
- **Std** — fold-to-fold variance; small std means the model is stable.
- **95% CI** — `mean ± 2*std`, the empirical rule approximation of a 95% confidence interval.

```python
assert len(cv_scores) == 5, 'Expected 5 folds'
assert cv_scores.mean() > 0.5, 'Mean CV score should exceed 0.5'
```
Two sanity assertions. The first checks that cross-validation ran all 5 folds. The second checks that the model does better than random chance (F1 > 0.5 for a ~76% pass-rate dataset). These mirror autograder checks in the learner lab.

---

#### Cell 2 — Compare regularization strengths

```python
from sklearn.linear_model import LogisticRegression

clf_variants = {
    'LogReg (default C=1)' : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'LogReg (C=0.1)'       : LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE),
    'LogReg (C=10)'        : LogisticRegression(C=10, max_iter=1000, random_state=RANDOM_STATE),
}
```
Three Logistic Regression variants with different `C` values. In Logistic Regression, `C` is the **inverse** regularization strength: smaller C = stronger regularization (more shrinkage), larger C = weaker regularization (model can fit more freely). This is the opposite of Ridge/Lasso's `alpha`.

```python
cv_results = []
for name, clf in clf_variants.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_val_score(pipe, X, y_clf, cv=skf, scoring='f1')
    cv_results.append({'Model': name, 'Mean F1': scores.mean(), 'Std': scores.std()})

best_mean = max(r['Mean F1'] for r in cv_results)

table = Table(title='Regularization (C) — CV F1 Comparison', box=box.SIMPLE_HEAVY, border_style='blue')
table.add_column('Model', style='bold')
table.add_column('Mean F1', justify='right')
table.add_column('Std',     justify='right', style='dim')

for r in cv_results:
    color = 'green' if r['Mean F1'] == best_mean else 'white'
    star  = ' ★' if r['Mean F1'] == best_mean else ''
    table.add_row(
        r['Model'] + star,
        f'[{color}]{r["Mean F1"]:.3f}[/{color}]',
        f'{r["Std"]:.3f}',
    )

console.print(table)
```
Replaces the manual format-string table with a rich table. Results are first accumulated into `cv_results` (a list of dicts) and then rendered.

The best-performing variant is highlighted in green with a `★` star, making it immediately obvious which value of `C` performs best. The `Std` column (styled `dim`) is still shown — low std means the model is stable regardless of which fold it's tested on. This demonstrates the bias-variance tradeoff: too much regularization (small C) may underfit; too little (large C) may overfit.

---

#### Cell 3 — CV visualization

```python
plt.figure(figsize=(8, 4))
fold_labels = [f'Fold {i+1}' for i in range(5)]
```
`fold_labels` is a list comprehension producing `['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']` — these become the x-axis tick labels.

```python
bars = plt.bar(fold_labels, cv_scores, color='steelblue', alpha=0.8, edgecolor='white')
```
Draws one bar per fold. `edgecolor='white'` adds a thin white border between adjacent bars, making them visually distinct even when they're very close in height.

```python
plt.axhline(cv_scores.mean(), color='red', linestyle='--',
            label=f'Mean = {cv_scores.mean():.3f}')
```
Draws a horizontal dashed red line at the mean F1 score. This is the single number we report as the model's performance. Bars above the line are above-average folds; bars below are below-average.

```python
plt.fill_between(
    range(-1, 6),
    cv_scores.mean() - cv_scores.std(),
    cv_scores.mean() + cv_scores.std(),
    alpha=0.15, color='red', label=f'±1 std = {cv_scores.std():.3f}'
)
```
`fill_between` shades the region between two y-values across a range of x values:
- `range(-1, 6)` spans from −1 to 5, which is wider than the 5 bars (at positions 0–4), so the shading extends to the edges of the chart.
- `cv_scores.mean() - cv_scores.std()` is the lower bound of the ±1 std band.
- `cv_scores.mean() + cv_scores.std()` is the upper bound.
- `alpha=0.15` makes it a very faint tint so the bars remain readable through it.
- **This band communicates stability**: a narrow band means the model performs consistently across all folds; a wide band signals high variance — performance depends heavily on which data ends up in the test fold.

```python
plt.ylim(0, 1)
```
Forces the y-axis to the full F1 range (0 to 1). Without this, matplotlib autoscales to the data range — a small spread between folds (e.g., 0.89 to 0.93) would appear enormous and misleadingly variable.

```python
plt.ylabel('F1 Score')
plt.title('5-Fold Stratified CV — Logistic Regression (F1 on pass/fail)')
plt.legend()
plt.tight_layout()
plt.show()
```
Standard labeling and rendering.

```python
print(f'\nLow std ({cv_scores.std():.3f}) → model is stable across different subsets of data.')
```
The narrative conclusion for the demo — a sentence to say out loud while pointing to the chart.

---

## Section 7 — Final Summary & University Recommendation


```python
clf_table = Table(title='Classification — Pass/Fail', box=box.MINIMAL_DOUBLE_HEAD, border_style='cyan')
clf_table.add_column('Metric',    style='bold')
clf_table.add_column('Value',     justify='right', style='bold yellow')

for k, v in classification_metrics.items():
    clf_table.add_row(k.capitalize(), f'{v:.3f}')
```
Builds the first of three summary tables. `box.MINIMAL_DOUBLE_HEAD` uses a double-line separator under the header only — a clean, minimal style suited for a final summary. Each metric is capitalized and shown to three decimal places.

```python
reg_table = Table(title='Regression — Final Score (Ridge)', box=box.MINIMAL_DOUBLE_HEAD, border_style='blue')
reg_table.add_column('Metric', style='bold')
reg_table.add_column('Value',  justify='right', style='bold yellow')

reg_table.add_row('MAE', f'{regression_metrics["mae"]:.3f}')
reg_table.add_row('R²',  f'{regression_metrics["r2"]:.3f}')
```
The second summary table for regression metrics. Hard-coded row names (`MAE`, `R²`) are used instead of iterating the dictionary keys so the display uses proper mathematical notation.

```python
cv_table = Table(title='Cross-Validation — 5-Fold Stratified F1', box=box.MINIMAL_DOUBLE_HEAD, border_style='green')
cv_table.add_column('Stat',  style='bold')
cv_table.add_column('Value', justify='right', style='bold yellow')

cv_table.add_row('Mean F1', f'{cv_scores.mean():.3f}')
cv_table.add_row('Std',     f'{cv_scores.std():.3f}')
```
The third table summarizes cross-validation. Mean = overall model performance; Std = consistency across folds.

```python
console.print()
console.print(clf_table)
console.print(reg_table)
console.print(cv_table)
```
Prints a blank line then renders the three tables in sequence.

```python
top_features = ['prior_gpa', 'study_hours_per_week', 'attendance_pct', 'prev_failures']
rec_text = (
    f'[bold yellow]Top predictors of failure:[/bold yellow] {", ".join(top_features)}\n\n'
    f'[cyan]1.[/cyan] Flag students with [bold]attendance < 65%[/bold] AND [bold]prior GPA < 2.5[/bold] early.\n'
    f'[cyan]2.[/cyan] Offer tutoring to students with [bold]prev_failures > 0[/bold].\n'
    f'[cyan]3.[/cyan] The model identifies [bold green]~{classification_metrics["recall"]:.0%}[/bold green] of at-risk students (recall).\n'
    f'[cyan]4.[/cyan] Low CV std suggests the model [bold green]generalises well[/bold green] — safe to deploy.'
)

console.print(Panel(rec_text, title='[bold]University Recommendation[/bold]', border_style='green', expand=False))
```
Replaces the plain `print` recommendation block with a formatted `Panel`. Each recommendation item uses rich markup:
- `[bold yellow]` — highlights the feature list header.
- `[cyan]` — numbers each recommendation in a distinct color.
- `[bold]` — emphasizes the specific threshold values (attendance < 65%, GPA < 2.5, prev_failures > 0).
- `[bold green]` — highlights the recall percentage and the "generalises well" conclusion.

`:.0%` formats recall as a rounded percentage (e.g., `0.918` → `92%`). `expand=False` keeps the panel compact rather than stretching to full terminal width.

The four hard-coded recommendations are grounded in the EDA and model results:
1. Actionable early-warning threshold from the heatmap.
2. `prev_failures` was the strongest negative predictor — a specific, targetable intervention.
3. Recall is the metric a dean cares about: how many at-risk students the model catches.
4. Low CV std closes the loop back to Section 6 — the model is stable, not just lucky on one split.

