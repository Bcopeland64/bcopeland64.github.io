# SI Demo — Amman Rides: Instructor Walkthrough

> **INSTRUCTOR USE ONLY**  
> This notebook (`si_demo_amman_rides.ipynb`) is a **live demonstration tool** for the SI session — it is **not** a student assignment. Do not distribute it to learners before or during the lab. The domain (Amman Rides, a fictional ride-sharing service) is intentionally different from the student lab (Hashemite Technical University student performance data) so that learners cannot copy code directly. They must understand the concept and adapt it themselves.

---

## Purpose of This Demo

The demo mirrors the **exact same four-step analytical workflow** students follow in Lab 4:

| Lab 4 Step | SI Demo Equivalent |
|------------|-------------------|
| Data inspection & cleaning | Data inspection (no missing values in demo) |
| Distribution analysis | Fare histogram + Distance box plot |
| Correlation analysis | Numeric correlation heatmap + scatter |
| Hypothesis testing | Surge vs non-surge fare t-test |

The key instructional value is **narrating the decision-making process aloud** while running each cell. Suggested narration prompts are embedded in the notebook as callout blocks and are reproduced in this walkthrough.

---

## Dataset — `trips.csv`

1,500 rows · 9 columns

| Column | Type | Description |
|--------|------|-------------|
| `trip_id` | int | Unique trip identifier |
| `date` | str | Date of the trip |
| `time_of_day` | str | Morning / Afternoon / Evening / Night |
| `pickup_neighborhood` | str | Amman neighbourhood name |
| `distance_km` | float | Trip distance in kilometres |
| `fare_jod` | float | Fare charged in Jordanian Dinars |
| `driver_rating` | float | Passenger rating of driver (1–5) |
| `payment_type` | str | Cash / Card / Wallet |
| `is_surge` | bool | Whether surge pricing was active |

---

## Cell 1 — Imports & Setup

Every analysis starts with loading the necessary libraries. This cell is identical in structure to the student lab. Point this out explicitly so learners recognise the pattern.

- `warnings.filterwarnings("ignore")` suppresses minor deprecation noise.
- `%matplotlib inline` renders plots directly in the notebook output.
- `os.makedirs("output", exist_ok=True)` creates the output folder without raising an error if it already exists.
- `sns.set_theme` and `FIGSIZE` establish a consistent visual style across all plots.

```python
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch

%matplotlib inline

os.makedirs("output", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (9, 5)
```

---

## Step 1 — Data Inspection

**Teaching goal:** Show learners the mandatory first steps before any analysis — know your shape, types, and missingness.

---

### Cell 2 — Load the Dataset and Check Shape

Reading the CSV and immediately printing the shape is the first confirmation that the file loaded correctly and that no rows were silently dropped.

```python
df = pd.read_csv("trips.csv")

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
```

**Expected output:**
```
Shape: 1500 rows x 9 columns
```

---

### Cell 3 — Column Data Types

`df.dtypes` reveals whether pandas inferred each column correctly. Point out that `is_surge` should be a boolean — if it were read as a string `"True"/"False"`, the hypothesis test later would silently break.

```python
# Column data types
df.dtypes
```

**Expected output:**
```
trip_id                int64
date                  object
time_of_day           object
pickup_neighborhood   object
distance_km          float64
fare_jod             float64
driver_rating        float64
payment_type          object
is_surge               bool
dtype: object
```

---

### Cell 4 — Summary Statistics

`df.describe()` gives the count, mean, std, min, quartiles, and max for every numeric column in one shot. Use this to set expectations before plotting — for example, note that `fare_jod` ranges from a few tenths of a JOD to double digits, hinting at right skew.

```python
# Summary statistics for all numeric columns
df.describe().round(3)
```

---

### Cell 5 — Missing Value Check

This dataset has **no missing values**, which is intentional — it lets you focus the SI session on analysis rather than cleaning. Tell learners: *"In your lab, you found missing values in `study_hours_weekly` and `commute_minutes`. Here we get lucky — but always check anyway."*

```python
# Missing value check
missing = df.isnull().sum()
print("Missing values per column:")
print(missing)
print(f"\nTotal missing cells: {missing.sum()}")
# No missing values in this demo dataset — good starting point.
```

**Expected output:**
```
trip_id                0
date                   0
time_of_day            0
pickup_neighborhood    0
distance_km            0
fare_jod               0
driver_rating          0
payment_type           0
is_surge               0
dtype: int64

Total missing cells: 0
```

---

### Cell 6 — Inspect Categorical Columns

Printing unique values for categorical columns is a quick way to catch typos or unexpected categories (e.g., `"Nigt"` instead of `"Night"`).

```python
# Unique values — categorical columns
for col in ["time_of_day", "pickup_neighborhood", "payment_type"]:
    vals = sorted([v for v in df[col].unique() if pd.notna(v)])
    print(f"{col}: {vals}")
```

**Expected output:**
```
time_of_day: ['Afternoon', 'Evening', 'Morning', 'Night']
pickup_neighborhood: ['7th Circle', 'Abdali', 'Jabal Amman', ...]
payment_type: ['Card', 'Cash', 'Wallet']
```

---

## Step 2 — Distribution Analysis

**Teaching goal:** Emphasise that the *shape* of a distribution determines which summary statistic is most appropriate.

**Narration prompt:**
> *"Before computing mean or median, look at the shape of the distribution. A right-skewed distribution means the mean is pulled up by a few large fares. In that case, the median is more representative of the 'typical' fare."*

---

### Cell 7 — Fare Distribution (Histogram + KDE + Annotated Mean/Median)

This is the centrepiece of Step 2. The histogram reveals right skew. The two vertical dashed lines — one for mean, one for median — make the gap concrete and visual.

**Key teaching point:** When mean > median, the distribution is right-skewed. The median is the better representative of a "typical" observation.

```python
# Fare distribution — right-skewed, annotated with mean and median lines
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.histplot(df["fare_jod"], bins=40, kde=True, ax=ax, color="steelblue")

mean_fare   = df["fare_jod"].mean()
median_fare = df["fare_jod"].median()

ax.axvline(mean_fare,   color="red",    linestyle="--", label=f"Mean = {mean_fare:.2f} JOD")
ax.axvline(median_fare, color="orange", linestyle="--", label=f"Median = {median_fare:.2f} JOD")
ax.set_title("Fare Distribution is Right-Skewed — Median Is the Better 'Typical' Fare")
ax.set_xlabel("Fare (JOD)")
ax.set_ylabel("Trip Count")
ax.legend()
fig.tight_layout()
fig.savefig("output/dist_fare.png", dpi=120)
plt.show()
```

Saved to `output/dist_fare.png`.

---

### Cell 8 — Trip Distance by Time of Day (Box Plot)

A box plot is ideal for comparing a numeric variable across categories. By setting `order=time_order` we control the x-axis sequence so it reads chronologically (Morning → Night) rather than alphabetically.

**Key teaching point:** Night trips show a higher median and a wider IQR — more variable distances. This could be an interesting follow-up question for students.

```python
# Trip distance by time of day — box plot
fig, ax = plt.subplots(figsize=FIGSIZE)
time_order = ["Morning", "Afternoon", "Evening", "Night"]
sns.boxplot(data=df, x="time_of_day", y="distance_km",
            order=time_order, ax=ax, palette="Set2")
ax.set_title("Trip Distance by Time of Day — Night Trips Tend to Be Longer")
ax.set_xlabel("Time of Day")
ax.set_ylabel("Distance (km)")
fig.tight_layout()
fig.savefig("output/boxplot_distance_by_time.png", dpi=120)
plt.show()
```

Saved to `output/boxplot_distance_by_time.png`.

---

## Step 3 — Correlation Analysis

**Teaching goal:** Reinforce that `.corr()` only makes sense on numeric columns, and that a low correlation coefficient has a concrete meaning — not just "weak".

**Narration prompt:**
> *"I only run `.corr()` on numeric columns. `time_of_day`, `payment_type`, and `pickup_neighborhood` are categorical — including them produces garbage results. Always filter first."*

---

### Cell 9 — Correlation Matrix

We select only the three meaningful numeric columns. Including `trip_id` would be meaningless (it's just an index), and `date` is a string here.

```python
# Correlation matrix — numeric columns only
numeric_cols = ["distance_km", "fare_jod", "driver_rating"]
corr = df[numeric_cols].corr()
corr.round(3)
```

**Expected output (approximate):**
```
               distance_km  fare_jod  driver_rating
distance_km          1.000     0.87          -0.01
fare_jod             0.870     1.00          -0.02
driver_rating       -0.010    -0.02           1.00
```

Point out the near-zero correlations for `driver_rating` — this is the "meaningless correlation" teaching moment.

---

### Cell 10 — Correlation Heatmap

The upper triangle is masked (`np.triu`) because the matrix is symmetric — displaying both halves is redundant. `coolwarm` centred at 0 makes strong positive correlations red and strong negative ones blue.

```python
# Correlation heatmap — lower triangle only
fig, ax = plt.subplots(figsize=(6, 5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", mask=mask,
            cmap="coolwarm", center=0, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Correlation Heatmap — Fare and Distance Are Strongly Related")
fig.tight_layout()
fig.savefig("output/correlation_heatmap.png", dpi=120)
plt.show()
```

Saved to `output/correlation_heatmap.png`.

---

### Cell 11 — Scatter Plot with Surge Colouring

The scatter plot adds a third dimension by colouring surge trips red and normal trips blue. This reveals that surge trips cluster **above** the distance-fare trend line — a visual cue for the hypothesis test in Step 4.

**Narration prompt:**
> *"Notice that `driver_rating` has near-zero correlation with fare. That is a meaningless correlation — knowing the rating tells us almost nothing about how much a trip costs. Not every number in your dataset is worth analysing."*

```python
# Scatter: distance vs fare, coloured by surge status
r = corr.loc["distance_km", "fare_jod"]

fig, ax = plt.subplots(figsize=FIGSIZE)
colors = df["is_surge"].map({True: "red", False: "steelblue"})
ax.scatter(df["distance_km"], df["fare_jod"],
           c=colors, alpha=0.3, s=18)
ax.set_title(f"Fare Rises with Distance (r = {r:.2f}) — Surge Trips (red) Cluster Above the Trend")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Fare (JOD)")
ax.legend(handles=[Patch(color="red",      label="Surge"),
                   Patch(color="steelblue", label="Normal")])
fig.tight_layout()
fig.savefig("output/scatter_distance_vs_fare.png", dpi=120)
plt.show()
```

Saved to `output/scatter_distance_vs_fare.png`.

---

## Step 4 — Hypothesis Testing: Surge vs Non-Surge Fare

**Teaching goal:** Show the full hypothesis-testing workflow — state the hypotheses, choose the right test, interpret the result in plain language.

### Hypotheses

- **H₀:** Mean fare (surge) = Mean fare (non-surge)
- **H₁:** Mean fare (surge) > Mean fare (non-surge) *(one-tailed)*

**Narration prompt:**
> *"We already saw surge trips sit above the trend line — but is that difference statistically significant, or could it just be random noise? A hypothesis test gives us a formal answer."*

---

### Cell 12 — Run the t-test

We use Welch's t-test (`equal_var=False`) because the two groups likely have different variances and different sizes. We halve the two-tailed p-value to get the one-tailed p-value because our hypothesis is directional (surge trips earn *more*).

```python
alpha = 0.05

fare_surge  = df.loc[df["is_surge"] == True,  "fare_jod"]
fare_normal = df.loc[df["is_surge"] == False, "fare_jod"]

t_stat, p_two = stats.ttest_ind(fare_surge, fare_normal, equal_var=False)
p_one = p_two / 2  # one-tailed

print(f"Surge trips    : n={len(fare_surge)}, mean fare = {fare_surge.mean():.3f} JOD")
print(f"Non-surge trips: n={len(fare_normal)}, mean fare = {fare_normal.mean():.3f} JOD")
print(f"t-statistic    : {t_stat:.4f}")
print(f"p-value (one-tailed): {p_one:.6f}")

if p_one < alpha and t_stat > 0:
    print(f"\nResult: REJECT H0 (p={p_one:.6f} < {alpha})")
    print("Surge trips have a statistically significantly higher average fare.")
    print(f"The chance of seeing this fare gap by random luck is less than {p_one*100:.4f}%.")
else:
    print(f"\nResult: FAIL TO REJECT H0 (p={p_one:.6f} >= {alpha})")
```

**Expected output (approximate):**
```
Surge trips    : n=..., mean fare = ... JOD
Non-surge trips: n=..., mean fare = ... JOD
t-statistic    : ...
p-value (one-tailed): ...

Result: REJECT H0 — surge trips have a significantly higher average fare.
```

**Parallel to Lab 4:** This is the same Welch's t-test structure students use for the internship vs GPA test. The only difference is the domain (fares vs GPA) and the grouping variable (`is_surge` vs `has_internship`).

---

## Post-Demo Discussion Prompts

After closing the notebook, ask learners to connect it back to their own lab:

1. *"What column in your dataset plays the same role as `fare_jod` — a continuous outcome you're analysing?"*
2. *"Your fare distribution was right-skewed. Was `gpa` right-skewed, left-skewed, or roughly normal? Does that change which summary statistic you should report?"*
3. *"We masked the upper triangle of the heatmap. Why? When would you not want to do that?"*
4. *"In the scatter plot, we coloured by `is_surge`. What categorical variable in your lab could you use to colour a scatter plot?"*
5. *"The `driver_rating` correlation was near zero. Did you find any near-zero correlations in your lab? What does that tell you about that variable?"*

---

## Output Files

| File | Description |
|------|-------------|
| `output/dist_fare.png` | Fare histogram with mean/median lines |
| `output/boxplot_distance_by_time.png` | Distance box plots by time of day |
| `output/correlation_heatmap.png` | Lower-triangle correlation heatmap |
| `output/scatter_distance_vs_fare.png` | Distance vs fare scatter, coloured by surge |

---

> **Reminder:** This file and the notebook are for **instructor preparation only**. Keep them out of the student-facing repository until after the lab submission deadline.
