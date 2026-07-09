# Lab 4 — Descriptive Analytics: Walkthrough

This document walks through every cell of `eda_analysis.ipynb` step by step, explaining what each block does and why, followed by the full code.

---

## Overview

The lab analyses a synthetic student performance dataset from Hashemite Technical University (2,000 students, 10 columns). It is split into four tasks:

| Task | Description | Output |
|------|-------------|--------|
| 1 | Data inspection & cleaning | `output/data_profile.txt` |
| 2 | Distribution analysis | `output/*.png` |
| 3 | Correlation analysis | `output/correlation_heatmap.png`, scatter plots |
| 4 | Hypothesis testing | printed results + `output/hypothesis_results.txt` |

---

## Cell 1 — Imports & Setup

Before any analysis can begin, we need to load the libraries we will use throughout the notebook and create the `output/` folder where all artefacts (charts, text files) will be saved.

- `warnings.filterwarnings("ignore")` suppresses minor deprecation warnings so the output stays clean.
- `numpy` and `pandas` handle numerical computation and tabular data.
- `matplotlib` and `seaborn` produce the visualisations.
- `scipy.stats` provides the statistical tests used in Task 4.
- `%matplotlib inline` tells Jupyter to render charts inside the notebook.
- `sns.set_theme` applies a consistent visual style, and `FIGSIZE` sets a shared default figure size so all plots look uniform.

```python
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

%matplotlib inline

os.makedirs("output", exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (9, 5)
```

---

## Task 1 — Data Inspection & Cleaning

---

### Cell 2 — Load the Dataset

We read the CSV file into a pandas DataFrame and immediately print its shape to confirm we loaded all rows and columns successfully. The result shows **2,000 rows × 10 columns**, which matches expectations.

```python
df = pd.read_csv("data/student_performance.csv")

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
```

**Output:**
```
Shape: 2000 rows × 10 columns
```

---

### Cell 3 — Column Data Types

Checking data types is the first sanity check after loading. It tells us whether pandas has inferred each column correctly — for instance, whether `gpa` is a float, `has_internship` is a bool, or a numeric column has accidentally been read as a string.

```python
# Column data types
df.dtypes
```

**Output:**
```
student_id              int64
department                str
semester                  str
course_load             int64
study_hours_weekly    float64
gpa                   float64
attendance_pct        float64
has_internship           bool
commute_minutes       float64
scholarship               str
dtype: object
```

All types look correct: IDs and counts are integers, continuous measurements are floats, the internship flag is boolean, and the categorical text columns are strings.

---

### Cell 4 — Identify Missing Values

We count null values in every column and express them as a percentage of total rows. Only columns that actually have missing values are displayed, keeping the output focused.

`study_hours_weekly` is missing for ~4% of students, and `commute_minutes` for ~10%. Both will be imputed in the next step.

```python
# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "pct%": missing_pct})
missing_df[missing_df["count"] > 0]
```

**Output:**
```
                    count   pct%
study_hours_weekly     81   4.05
commute_minutes       208  10.40
```

---

### Cell 5 — Summary Statistics

`df.describe()` gives us the count, mean, standard deviation, min, max, and quartiles for every numeric column. This is a quick way to spot outliers (e.g., a student studying 45 h/wk) and understand the central tendency and spread of each variable before we plot anything.

```python
# Summary statistics
df.describe().round(3)
```

**Output:**
```
       student_id  course_load  study_hours_weekly       gpa  attendance_pct  commute_minutes
count    2000.000     2000.000            1919.000  2000.000        2000.000         1792.000
mean    11000.500        5.011              15.976     2.719          70.784           46.309
std       577.495        1.385               6.786     0.683          12.694           24.624
min     10001.000        3.000               4.000     0.200          40.000            5.000
25%     10500.750        4.000              11.100     2.260          62.300           25.000
50%     11000.500        5.000              14.600     2.780          71.400           46.000
75%     11500.250        6.000              19.500     3.230          79.500           67.000
max     12000.000        7.000              45.000     4.000         100.000           90.000
```

---

### Cell 6 — Inspect Categorical Columns

For the three categorical columns, we print their unique values. This confirms there are no unexpected typos or extra categories (e.g., `"Buisness"` instead of `"Business"`) that would need to be cleaned.

```python
# Unique values — categorical columns
for col in ["department", "semester", "scholarship"]:
    vals = sorted([v for v in df[col].unique() if pd.notna(v)])
    print(f"{col}: {vals}")
```

**Output:**
```
department: ['Business', 'Civil Engineering', 'Computer Science', 'Electrical Engineering', 'Mechanical Engineering']
semester: ['Fall 2024', 'Fall 2025', 'Spring 2025']
scholarship: ['Full', 'No Scholarship', 'Partial']
```

---

### Cell 7 — Impute Missing Values with the Median

Rather than dropping rows (which wastes data) or using the mean (which is sensitive to outliers), we fill missing values with the **column median**. The median is robust to skew, making it ideal for continuous measurements like commute time or study hours.

After imputation, we verify that no nulls remain with `df.isnull().sum().sum()`.

```python
# Impute missing values with column median (robust to outliers / skew)
commute_median = df["commute_minutes"].median()
study_median   = df["study_hours_weekly"].median()

df["commute_minutes"]    = df["commute_minutes"].fillna(commute_median)
df["study_hours_weekly"] = df["study_hours_weekly"].fillna(study_median)

print(f"commute_minutes    fill value : {commute_median}")
print(f"study_hours_weekly fill value : {study_median}")
print(f"Missing values remaining      : {df.isnull().sum().sum()}")
```

**Output:**
```
commute_minutes    fill value : 46.0
study_hours_weekly fill value : 14.6
Missing values remaining      : 0
```

---

### Cell 8 — Save the Data Profile to Disk

We compile a human-readable summary of everything discovered in Task 1 — shape, types, missing counts, summary statistics, and imputation fill values — and write it to `output/data_profile.txt`. This creates a reproducible audit trail of the cleaning steps.

```python
# Persist data profile to disk
profile_lines = [
    f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
    "",
    "Column Data Types:",
    df.dtypes.to_string(),
    "",
    "Missing Values (before imputation):",
    missing_df[missing_df["count"] > 0].to_string(),
    "",
    "Summary Statistics:",
    df.describe().round(3).to_string(),
    "",
    f"commute_minutes    median fill : {commute_median}",
    f"study_hours_weekly median fill : {study_median}",
]

with open("output/data_profile.txt", "w") as f:
    f.write("\n".join(profile_lines))

print("✓ Saved output/data_profile.txt")
```

**Output:**
```
✓ Saved output/data_profile.txt
```

---

## Task 2 — Distribution Analysis

The goal here is to understand the **shape** of each important variable individually before looking at relationships between them.

---

### Cell 9 — GPA Distribution (Histogram + KDE)

We plot a histogram of GPA with a kernel density estimate (KDE) curve overlaid. The KDE smooths the histogram into a continuous probability curve, making it easier to see whether the distribution is roughly normal, skewed, or bimodal.

The chart is saved to `output/dist_gpa.png` for the report.

```python
# GPA distribution
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.histplot(df["gpa"], bins=30, kde=True, ax=ax, color="steelblue")
ax.set_title("GPA Distribution — Most Students Cluster Between 2.5 and 3.5")
ax.set_xlabel("Semester GPA")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig("output/dist_gpa.png", dpi=120)
plt.show()
```

---

### Cell 10 — Study Hours Distribution

The same histogram + KDE pattern is applied to `study_hours_weekly`. The title already hints at the finding: the distribution is **right-skewed**, meaning most students study around 15 h/wk but a long tail of heavy studiers pulls the mean above the median.

```python
# Weekly study hours distribution
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.histplot(df["study_hours_weekly"], bins=30, kde=True, ax=ax, color="mediumseagreen")
ax.set_title("Weekly Study Hours — Right-Skewed; Median ≈ 15 h/wk")
ax.set_xlabel("Study Hours per Week")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig("output/dist_study_hours.png", dpi=120)
plt.show()
```

---

### Cell 11 — Attendance Distribution

Attendance percentage follows a roughly normal distribution centred around 75%, as shown by the symmetric bell shape. This is useful context for the correlation analysis — a near-normal predictor behaves well in linear models.

```python
# Attendance distribution
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.histplot(df["attendance_pct"], bins=30, kde=True, ax=ax, color="darkorange")
ax.set_title("Attendance Percentage — Roughly Normal, Centred ~75%")
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig("output/dist_attendance.png", dpi=120)
plt.show()
```

---

### Cell 12 — Student Count by Department (Bar Chart)

For categorical variables, a bar chart is more appropriate than a histogram. `value_counts()` counts students per department, and a horizontal bar chart makes the department labels easy to read without rotation.

```python
# Department counts (categorical bar chart)
fig, ax = plt.subplots(figsize=FIGSIZE)
dept_counts = df["department"].value_counts()
sns.barplot(x=dept_counts.values, y=dept_counts.index, ax=ax, palette="pastel")
ax.set_title("Student Count by Department")
ax.set_xlabel("Number of Students")
ax.set_ylabel("Department")
fig.tight_layout()
fig.savefig("output/dist_department.png", dpi=120)
plt.show()
```

---

### Cell 13 — GPA by Department (Box Plot)

A box plot shows median, interquartile range, and outliers for GPA split by department. Departments are ordered by median GPA (highest first) so differences are immediately visible. The finding — Computer Science leads — is embedded in the chart title.

```python
# GPA by department (box plot)
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby("department")["gpa"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="department", y="gpa", order=order, ax=ax, palette="Set2")
ax.set_title("GPA Distribution by Department — Computer Science Leads")
ax.set_xlabel("Department")
ax.set_ylabel("Semester GPA")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig("output/boxplot_gpa_by_dept.png", dpi=120)
plt.show()
```

---

## Task 3 — Correlation Analysis

Correlation analysis measures the **linear relationship** between pairs of numeric variables. Values near +1 indicate a strong positive relationship, near −1 a strong negative one, and near 0 little or no linear relationship.

---

### Cell 14 — Correlation Matrix

We select the five meaningful numeric columns (excluding the ID), compute pairwise Pearson correlations, and display the rounded matrix. The two standout values are:

- `gpa` ↔ `attendance_pct` : **r = 0.789** (strong positive)
- `gpa` ↔ `study_hours_weekly` : **r = 0.271** (moderate positive)

```python
numeric_cols = ["course_load", "study_hours_weekly", "gpa",
                "attendance_pct", "commute_minutes"]
corr_matrix  = df[numeric_cols].corr()
corr_matrix.round(3)
```

**Output:**
```
                    course_load  study_hours_weekly    gpa  attendance_pct  commute_minutes
course_load               1.000              -0.009 -0.062          -0.047           -0.018
study_hours_weekly       -0.009               1.000  0.271           0.231           -0.039
gpa                      -0.062               0.271  1.000           0.789           -0.017
attendance_pct           -0.047               0.231  0.789           1.000           -0.007
commute_minutes          -0.018              -0.039 -0.017          -0.007            1.000
```

---

### Cell 15 — Correlation Heatmap

A colour-coded heatmap makes the correlation matrix much faster to read than a table of numbers. We apply a **triangular mask** (`np.triu`) to hide the upper half, since the matrix is symmetric — this removes the redundant mirror image and focuses attention on the lower triangle.

`coolwarm` is used as the colour palette, centred at 0, so strong positive correlations appear red and strong negative ones blue.

```python
# Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", mask=mask,
            cmap="coolwarm", center=0, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Correlation Heatmap — Numeric Variables")
fig.tight_layout()
fig.savefig("output/correlation_heatmap.png", dpi=120)
plt.show()
```

---

### Cell 16 — Top-2 Correlated Pairs Scatter Plots

To visualise the strongest relationships, we programmatically identify the top two correlated pairs from the masked, flattened correlation matrix and produce a scatter plot for each. `alpha=0.35` makes overlapping points semi-transparent so density is visible even where many students cluster.

The Pearson *r* value is shown in each chart title so the reader can immediately link the visual pattern to its numeric strength.

```python
# Top-2 correlated pairs — scatter plots
corr_flat = (corr_matrix.where(~mask)
             .stack()
             .abs()
             .sort_values(ascending=False))
top_pairs = corr_flat.head(2).index.tolist()

for i, (c1, c2) in enumerate(top_pairs, 1):
    r = corr_matrix.loc[c1, c2]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(data=df, x=c1, y=c2, alpha=0.35, ax=ax, color="steelblue")
    ax.set_title(f"Scatter: {c1} vs {c2}  (r = {r:.2f})")
    ax.set_xlabel(c1.replace("_", " ").title())
    ax.set_ylabel(c2.replace("_", " ").title())
    fig.tight_layout()
    fname = f"output/scatter_{c1}_vs_{c2}.png"
    fig.savefig(fname, dpi=120)
    plt.show()
    print(f"✓ {fname}")

print(f"\nTop correlated pair 1 : {top_pairs[0]}")
print(f"Top correlated pair 2 : {top_pairs[1]}")
```

The strongest pair is `attendance_pct` ↔ `gpa` (r = 0.79), and the second is `study_hours_weekly` ↔ `gpa` (r = 0.27).

---

## Task 4 — Hypothesis Testing

Hypothesis testing gives us a formal, probabilistic framework to decide whether an observed difference or association is likely real or could have occurred by chance. We use **α = 0.05** as our significance threshold throughout.

---

### Hypothesis 1 — Internship vs GPA (Independent-Samples t-test)

**Research question:** Do students who have internships earn a higher GPA than those who do not?

- **H₀:** Mean GPA (internship) = Mean GPA (no internship)
- **H₁:** Mean GPA (internship) > Mean GPA (no internship) *(one-tailed)*

---

### Cell 17 — Run the t-test

We split students into two groups by `has_internship`, then run Welch's t-test (`equal_var=False`), which does not assume the two groups have equal variance — safer when group sizes differ significantly (512 vs 1,488). Because our hypothesis is directional (internship students earn *higher* GPA), we halve the two-tailed p-value to get the one-tailed p-value.

```python
alpha = 0.05

gpa_intern    = df.loc[df["has_internship"] == True,  "gpa"]
gpa_no_intern = df.loc[df["has_internship"] == False, "gpa"]

t_stat, p_val_two = stats.ttest_ind(gpa_intern, gpa_no_intern, equal_var=False)
p_val_one = p_val_two / 2  # one-tailed

print(f"Group sizes   : internship={len(gpa_intern)}, no internship={len(gpa_no_intern)}")
print(f"Mean GPA      : internship={gpa_intern.mean():.3f}, no internship={gpa_no_intern.mean():.3f}")
print(f"t-statistic   : {t_stat:.4f}")
print(f"p-value (one-tailed): {p_val_one:.4f}")

if p_val_one < alpha and t_stat > 0:
    print(f"\nResult: REJECT H0 (p={p_val_one:.4f} < {alpha})")
    print("Interpretation: Statistically significant evidence that students with")
    print("internships achieve a higher average GPA than those without.")
else:
    print(f"\nResult: FAIL TO REJECT H0 (p={p_val_one:.4f} >= {alpha})")
    print("Interpretation: No significant GPA difference was found between groups.")
```

**Output:**
```
Group sizes   : internship=512, no internship=1488
Mean GPA      : internship=2.821, no internship=2.684
t-statistic   : 3.8916
p-value (one-tailed): 0.0001

Result: REJECT H0 (p=0.0001 < 0.05)
Interpretation: Statistically significant evidence that students with
internships achieve a higher average GPA than those without.
```

With p = 0.0001, which is far below our 0.05 threshold, and a positive t-statistic confirming the direction, we reject H₀. Internship students average a GPA of **2.821** vs **2.684** for non-internship students — a statistically significant difference.

---

### Hypothesis 2 — Scholarship vs Department (Chi-Square Test)

**Research question:** Is scholarship status related to which department a student belongs to, or are they independent?

- **H₀:** Scholarship status and department are independent
- **H₁:** Scholarship status and department are associated

---

### Cell 18 — Build the Contingency Table

A **contingency table** (cross-tabulation) counts how many students fall into each combination of scholarship type and department. This is the input required for the chi-square test.

```python
contingency = pd.crosstab(df["scholarship"], df["department"])
print("Contingency table:")
contingency
```

**Output:**
```
department      Business  Civil Engineering  Computer Science  Electrical Engineering  Mechanical Engineering
scholarship
Full                  51                 60               127                      64                      57
No Scholarship       201                178               207                     199                     213
Partial              127                 90               179                     121                     126
```

Notice that Computer Science has notably more `Full` scholarship holders (127) compared to other departments, which already hints at an association.

---

### Cell 19 — Run the Chi-Square Test

`scipy.stats.chi2_contingency` computes the chi-square statistic by comparing the observed counts in the contingency table to the counts we would expect if scholarship and department were truly independent. A large chi-square value means the observed counts deviate substantially from what independence would predict.

```python
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

print(f"Chi-square statistic : {chi2:.4f}")
print(f"Degrees of freedom   : {dof}")
print(f"p-value              : {p_chi2:.4f}")

if p_chi2 < alpha:
    print(f"\nResult: REJECT H0 (p={p_chi2:.4f} < {alpha})")
    print("Interpretation: Scholarship distribution differs significantly across")
    print("departments — scholarship status and department are NOT independent.")
else:
    print(f"\nResult: FAIL TO REJECT H0 (p={p_chi2:.4f} >= {alpha})")
    print("Interpretation: No significant association between scholarship status and department.")
```

**Output:**
```
Chi-square statistic : 37.2704
Degrees of freedom   : 8
p-value              : 0.0000

Result: REJECT H0 (p=0.0000 < 0.05)
Interpretation: Scholarship distribution differs significantly across
departments — scholarship status and department are NOT independent.
```

χ² = 37.27 with 8 degrees of freedom yields an essentially zero p-value. We reject H₀: scholarship type is **not** distributed uniformly across departments.

---

### Cell 20 — Save Hypothesis Results to Disk

The key numeric results from both tests are written to `output/hypothesis_results.txt` so they can be referenced in a report without re-running the notebook.

```python
# Save hypothesis results
hyp_lines = [
    "── Hypothesis 1 (t-test: internship vs GPA) ────────────────",
    f"  Group sizes   : internship={len(gpa_intern)}, no internship={len(gpa_no_intern)}",
    f"  Mean GPA      : internship={gpa_intern.mean():.3f}, no internship={gpa_no_intern.mean():.3f}",
    f"  t-statistic   : {t_stat:.4f}",
    f"  p-value (one-tailed): {p_val_one:.4f}",
    "",
    "── Hypothesis 2 (chi-square: scholarship vs department) ─────",
    f"  Chi-square statistic : {chi2:.4f}",
    f"  Degrees of freedom   : {dof}",
    f"  p-value              : {p_chi2:.4f}",
]

with open("output/hypothesis_results.txt", "w") as f:
    f.write("\n".join(hyp_lines))

print("✓ Saved output/hypothesis_results.txt")
```

**Output:**
```
✓ Saved output/hypothesis_results.txt
```

---

## Summary Cell — Final Report Numbers

The last cell pulls together the headline numbers from all four tasks into a single printed block, making it easy to copy the values straight into a report or presentation.

```python
_intern_mean   = gpa_intern.mean()
_nointern_mean = gpa_no_intern.mean()
_top_corr      = corr_matrix.loc[top_pairs[0][0], top_pairs[0][1]]

print("[SUMMARY FOR REPORT]")
print(f"  GPA w/ internship   : {_intern_mean:.3f}")
print(f"  GPA w/o internship  : {_nointern_mean:.3f}")
print(f"  t-stat / p (1-tail) : {t_stat:.3f} / {p_val_one:.4f}")
print(f"  chi2 / p            : {chi2:.3f} / {p_chi2:.4f}")
print(f"  Top corr pair       : {top_pairs[0]}  r={_top_corr:.2f}")
print("\n✓ All tasks complete. Check the output/ directory.")
```

**Output:**
```
[SUMMARY FOR REPORT]
  GPA w/ internship   : 2.821
  GPA w/o internship  : 2.684
  t-stat / p (1-tail) : 3.892 / 0.0001
  chi2 / p            : 37.270 / 0.0000
  Top corr pair       : ('attendance_pct', 'gpa')  r=0.79

✓ All tasks complete. Check the output/ directory.
```

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Strongest predictor of GPA** | Attendance percentage (r = 0.79) |
| **Second predictor** | Weekly study hours (r = 0.27) |
| **Commute time** | No meaningful correlation with GPA or study hours |
| **Internship effect** | Internship students GPA 2.821 vs 2.684 — significant (p = 0.0001) |
| **Scholarship distribution** | Differs significantly across departments (χ² = 37.27, p ≈ 0) |
| **Department with highest median GPA** | Computer Science |
