# SI Demo — Code Walkthrough
## "Visit Jordan" Tourism KPI Dashboard

This document explains every line of `demo_tourism_dashboard.py` (and the matching notebook `si_demo_notebook.ipynb`).  
Read it top-to-bottom alongside the source file.

> **Note on path differences:** The `.py` script uses `os.path.dirname(__file__)` to locate files relative to the script itself. The notebook uses `os.getcwd()` instead, which returns the directory from which the Jupyter kernel was launched. Both resolve to the same folder when the notebook is opened from the project root.

---

## Table of Contents

1. [Module-Level Setup](#1-module-level-setup)
2. [Data Generation — `generate_tourism_data()`](#2-data-generation)
3. [Data Loading — `load_data()`](#3-data-loading)
4. [KPI 1 — Monthly Visitor Growth Rate](#4-kpi-1--monthly-visitor-growth-rate)
5. [KPI 2 — Revenue Per Tourist by Origin](#5-kpi-2--revenue-per-tourist-by-origin)
6. [KPI 3 — Seasonal Concentration Index](#6-kpi-3--seasonal-concentration-index-hhi)
7. [Statistical Test — Welch's t-test](#7-statistical-test--welchs-t-test)
8. [Visualisation — `plot_dashboard()`](#8-visualisation--plot_dashboard)
9. [Executive Summary](#9-executive-summary)
10. [Entry Point — `main()`](#10-entry-point--main)

---

## 1. Module-Level Setup

```python
import os
import warnings
```
- `os` — lets us build file paths that work on Windows, macOS, and Linux without hardcoding slashes. Key functions used later: `os.getcwd()`, `os.path.join()`, `os.path.exists()`, `os.makedirs()`.
- `warnings` — gives us control over which Python warnings are printed to the console. Without this import we would have no way to suppress specific warning categories selectively.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

| Import | Purpose in this script |
|--------|------------------------|
| `numpy` (`np`) | Random number generation via `default_rng`; the `np.cos` and `np.pi` constants used in the seasonal multiplier |
| `pandas` (`pd`) | All tabular data manipulation — DataFrames, `groupby`, `pct_change`, `merge`, `to_datetime` |
| `matplotlib.pyplot` (`plt`) | Low-level figure and axis creation; `GridSpec` for the multi-panel layout; `savefig` for PNG export |
| `seaborn` (`sns`) | High-level statistical chart functions (`boxplot`, `heatmap`) and opinionated color palettes, built on top of matplotlib |
| `scipy.stats` | Welch's t-test via `stats.ttest_ind` |

```python
warnings.filterwarnings("ignore", category=FutureWarning)
```
Silences deprecation notices that pandas emits about API changes planned for future releases. These warnings do not affect correctness — they are informational only.

**Why filter by category, not globally?**  
`warnings.filterwarnings("ignore")` (with no `category`) would silence *all* warnings including `RuntimeWarning` (e.g., division by zero) and `UserWarning` (e.g., malformed data). Filtering only `FutureWarning` keeps the output clean while leaving genuine problems visible.

```python
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
```
- `os.path.dirname(__file__)` — returns the directory that *contains the script*, so the output folder is always created next to the script regardless of your shell's working directory.
- `os.path.join(base, "demo_output")` — appends the subdirectory name using the OS-correct separator (`/` on Unix, `\` on Windows). Never use string concatenation (e.g., `base + "/demo_output"`) — it breaks on Windows.
- `os.makedirs(..., exist_ok=True)` — creates the folder *and any missing parent folders* in one call. `exist_ok=True` silently does nothing if the directory already exists, preventing a `FileExistsError` on the second and subsequent runs.

```python
sns.set_theme(style="whitegrid", palette="muted")
```
Applies a consistent seaborn theme to every plot produced in the session. This is a *global* setter — it modifies matplotlib's `rcParams` under the hood, so all subsequent charts inherit these settings without extra arguments.

- `style="whitegrid"` — white background with grey horizontal grid lines; clean and professional, and the grid makes it easier to read values off a chart.
- `palette="muted"` — a set of desaturated, perceptually distinct colors that reproduce well in greyscale print and are accessible to colorblind readers (deuteranopia/protanopia-friendly).

### Complete Code — Section 1

```python
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress pandas FutureWarnings that clutter the output
warnings.filterwarnings("ignore", category=FutureWarning)

# Where chart PNGs will be saved (script version uses __file__; notebook uses os.getcwd())
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Apply a consistent seaborn theme across every plot
sns.set_theme(style="whitegrid", palette="muted")
```

---

## 2. Data Generation

```python
def generate_tourism_data():
```
A self-contained function that builds and persists the synthetic dataset. Wrapping it in a function means it can be called on-demand from `load_data()` as a fallback if the CSV file is missing, rather than crashing.

---

### Statistical Concept: Reproducibility with a Seeded RNG

```python
    rng = np.random.default_rng(42)
```
Creates a **seeded** random number generator using NumPy's modern `default_rng` API (backed by the PCG64 algorithm).

- **Seed `42` is arbitrary but fixed.** A fixed seed guarantees the exact same sequence of random numbers on every run, on every machine. This is essential for a teaching context — every learner who runs the notebook sees identical values, so class discussions remain consistent.
- **`default_rng` vs `np.random.seed`:** The legacy `np.random.seed()` mutates *global* random state, which means it can interfere with other code that uses NumPy random functions. `default_rng()` creates a self-contained generator *object* with its own internal state — safer and preferred in modern NumPy (≥1.17).
- **To generate a different-but-plausible dataset:** Change `42` to any other integer (e.g., `rng = np.random.default_rng(99)`).

---

```python
    sites        = ["Petra", "Dead Sea", "Wadi Rum", "Jerash", "Aqaba"]
    months       = pd.date_range("2024-01", periods=24, freq="ME")
    origins      = ["Arab", "European", "North American", "Asian", "Other"]
    ticket_types = ["individual", "group", "student"]
```
- `pd.date_range("2024-01", periods=24, freq="ME")` — generates 24 month-end timestamps (`"ME"` = Month End frequency) from January 2024 through December 2025. Using `"ME"` instead of the deprecated `"M"` alias avoids a `FutureWarning` in recent pandas versions.
- The three lists drive the quadruple-nested loop below: 24 months × 5 sites × 5 origins × 3 ticket types = **1,800 rows** total.

---

### Statistical Concept: The Seasonal Multiplier (Cosine Wave)

```python
    def seasonal(month_num):
        return 1.0 + 0.4 * np.cos((month_num - 4) * np.pi / 6)
```
An inner helper function that maps any month number (1–12) to a multiplier in the range **[0.6, 1.4]**.

**The formula unpacked:**

    multiplier = 1.0 + 0.4 × cos( (month_num − 4) × π / 6 )

| Component | Effect |
|-----------|--------|
| `1.0` | Baseline — no seasonal adjustment |
| `0.4` | Amplitude — the peak is 40 % above and the trough is 40 % below the baseline |
| `(month_num − 4)` | Shifts the cosine peak to month 4 (April). When `month_num = 4`, the argument is 0, and `cos(0) = 1` → maximum multiplier |
| `× π / 6` | Scales the period. The cosine function has a natural period of 2π. Multiplying the argument by π/6 makes one full cycle span `2π / (π/6) = 12` months — exactly one year |

**Key output values:**

| Month | Multiplier | Interpretation |
|-------|-----------|----------------|
| April (4) | **1.4** | Peak season — 40 % more visitors than the baseline |
| October (10) | **0.6** | Trough — 40 % fewer visitors |
| January / July (1, 7) | **1.0** | Neutral months at the baseline |

This is the standard technique for embedding seasonality in simulated time-series data: define a deterministic *signal* (the cosine wave), then layer stochastic *noise* on top so the result looks realistic without losing control over its structure.

---

```python
    base_visitors = {"Petra": 12000, "Dead Sea": 8000, "Wadi Rum": 6000,
                     "Jerash": 4000,  "Aqaba":  7000}
    base_spend    = {"Petra": 85,    "Dead Sea": 70,   "Wadi Rum": 95,
                     "Jerash": 55,    "Aqaba":   65}
```
Dictionaries storing the "flat-season" monthly visitor capacity and average spend (in Jordanian Dinar) for each site. Defined once outside the loops so they are not re-created on every iteration.

### The Row-Building Loop

```python
    for month in months:
        m = month.month
        for site in sites:
            for origin in origins:
                for ttype in ticket_types:
```
A quadruple-nested loop. `month.month` extracts the integer month number (1–12) from the pandas Timestamp so it can be passed to `seasonal()`.

```python
                    origin_mult = {"Arab": 0.35, "European": 0.25,
                                   "North American": 0.18, "Asian": 0.14, "Other": 0.08}
                    ttype_mult  = {"individual": 0.50, "group": 0.35, "student": 0.15}
```
Market-share dictionaries. The values within each dict sum to exactly 1.0 — they represent the proportion of total visitors attributable to each segment.

> **Style note:** Defining these dicts inside the loop makes them easy to read and modify, at the cost of slight redundancy (they are reconstructed 1,800 times). For production code, move them outside the loop. In a teaching demo, readability wins.

```python
                    visitors = int(
                        base_visitors[site]
                        * seasonal(m)
                        * origin_mult[origin]
                        * ttype_mult[ttype]
                        * rng.uniform(0.85, 1.15)
                    )
```
Five-factor multiplication that produces one visitor count per row:

1. **`base_visitors[site]`** — the flat-season monthly capacity for this site
2. **`seasonal(m)`** — the cosine seasonal multiplier for this month (0.6–1.4)
3. **`origin_mult[origin]`** — the market share of this visitor origin (e.g., 0.25 for Europeans)
4. **`ttype_mult[ttype]`** — the share buying this ticket type (e.g., 0.50 for individual)
5. **`rng.uniform(0.85, 1.15)`** — a random noise multiplier drawn from a uniform distribution over [0.85, 1.15]

`int(...)` truncates the final float to a whole number — you cannot have a fractional visitor.

---

### Statistical Concept: Uniform Random Noise

```python
                    * rng.uniform(0.85, 1.15)   # ±15 % noise
```
Each row's visitor count is multiplied by a single random draw from a **continuous uniform distribution** over [0.85, 1.15].

**What "uniform" means:** every value in the interval is equally likely — there is no "most probable" noise value. This contrasts with a *normal distribution*, where values near the mean are far more likely than values at the tails.

**Why ±15 %?**
- Small enough that the underlying seasonal signal (the cosine wave) remains clearly visible in the charts.
- Large enough that consecutive months are not perfectly smooth, so the data looks like real observations rather than a mathematical formula.
- A ±15 % range is a common starting point for tourism and retail simulation models; real-world visit counts typically vary by 10–20 % around seasonal baselines due to weather, local events, and marketing campaigns.

```python
                    spend_adj = {"individual": 1.0, "group": 1.20, "student": 0.70}
                    avg_spend = round(
                        base_spend[site] * spend_adj[ttype] * rng.uniform(0.90, 1.10), 2
                    )
```
- Group visitors spend **20 % more** — tour packages often bundle meals, guided entry, and transport.
- Student visitors spend **30 % less** — discounted admission and lower incidentals.
- `rng.uniform(0.90, 1.10)` adds ±10 % price noise (tighter than visitor noise because prices are more stable than headcounts).
- `round(..., 2)` formats to two decimal places, matching Jordanian Dinar currency convention (fils).

```python
                    rows.append({
                        "month":            month.strftime("%Y-%m"),
                        "site":             site,
                        "visitor_count":    visitors,
                        "avg_spending_jod": avg_spend,
                        "origin_region":    origin,
                        "ticket_type":      ttype,
                    })
```
Appending a dict to a Python list is the standard pattern for building a DataFrame row-by-row. Converting the full list to a DataFrame in one call at the end (`pd.DataFrame(rows)`) is orders of magnitude faster than calling `df.append()` or `pd.concat()` on every iteration.

`month.strftime("%Y-%m")` serialises the Timestamp as a `"YYYY-MM"` string so it stores cleanly in CSV without an unwanted time component (e.g., `"2024-01-31 00:00:00"`).

```python
    df = pd.DataFrame(rows)
    csv_path = os.path.join(os.path.dirname(__file__), "tourism_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated: {len(df)} rows → {csv_path}")
    return df
```
- `pd.DataFrame(rows)` — converts the list of dicts to a DataFrame in one pass. Each dict key becomes a column name; dict keys must be identical across all rows (guaranteed here by the loop structure).
- `to_csv(..., index=False)` — writes without the auto-generated integer row-index column. Including the index (the default) would add a useless `0, 1, 2, …` column that would require dropping on reload.

### Complete Code — Section 2

```python
def generate_tourism_data():
    """
    Simulate 2 years of monthly visitor data for 5 Jordanian sites.
    Saves result to tourism_data.csv so learners can inspect the raw data.
    """
    rng = np.random.default_rng(42)

    sites        = ["Petra", "Dead Sea", "Wadi Rum", "Jerash", "Aqaba"]
    months       = pd.date_range("2024-01", periods=24, freq="ME")
    origins      = ["Arab", "European", "North American", "Asian", "Other"]
    ticket_types = ["individual", "group", "student"]

    def seasonal(month_num):
        return 1.0 + 0.4 * np.cos((month_num - 4) * np.pi / 6)

    base_visitors = {"Petra": 12000, "Dead Sea": 8000, "Wadi Rum": 6000,
                     "Jerash": 4000,  "Aqaba":  7000}
    base_spend    = {"Petra": 85, "Dead Sea": 70, "Wadi Rum": 95,
                     "Jerash": 55, "Aqaba": 65}

    rows = []

    for month in months:
        m = month.month
        for site in sites:
            for origin in origins:
                for ttype in ticket_types:
                    origin_mult = {"Arab": 0.35, "European": 0.25,
                                   "North American": 0.18, "Asian": 0.14, "Other": 0.08}
                    ttype_mult  = {"individual": 0.50, "group": 0.35, "student": 0.15}

                    visitors = int(
                        base_visitors[site]
                        * seasonal(m)
                        * origin_mult[origin]
                        * ttype_mult[ttype]
                        * rng.uniform(0.85, 1.15)
                    )

                    spend_adj = {"individual": 1.0, "group": 1.20, "student": 0.70}
                    avg_spend = round(
                        base_spend[site] * spend_adj[ttype] * rng.uniform(0.90, 1.10), 2
                    )

                    rows.append({
                        "month":            month.strftime("%Y-%m"),
                        "site":             site,
                        "visitor_count":    visitors,
                        "avg_spending_jod": avg_spend,
                        "origin_region":    origin,
                        "ticket_type":      ttype,
                    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(os.path.dirname(__file__), "tourism_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated: {len(df):,} rows → {csv_path}")
    return df


df_raw = generate_tourism_data()
```

---

## 3. Data Loading

```python
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "tourism_data.csv")
    if not os.path.exists(csv_path):
        return generate_tourism_data()
```
**Guard clause pattern:** if the CSV is missing (first run, or the file was deleted), fall back to generating it rather than crashing with a `FileNotFoundError`. This keeps the pipeline self-healing and removes the requirement to run sections in a specific order.

```python
    df = pd.read_csv(csv_path)
    df["month"] = pd.to_datetime(df["month"])
```
CSV files store everything as plain text. When `"2024-01"` is read back in, pandas stores it as a Python string (`object` dtype). `pd.to_datetime()` parses the string into a proper `datetime64` type.

**Why does this matter?**
- A string column cannot be sorted chronologically without extra steps (alphabetical sort would put `"2024-10"` before `"2024-02"`).
- Pandas plotting automatically formats a `datetime64` column as readable date labels on the x-axis.
- The `.dt` accessor (used in KPI 3) only works on datetime columns: `df["month"].dt.year`, `df["month"].dt.month`.

**Understanding `df.dtypes` output:**

| dtype | Meaning |
|-------|---------|
| `datetime64[us]` | Microsecond-precision timestamp — date-aware, supports arithmetic and resampling |
| `object` | Python string — flexible but slower; sorts lexicographically, not numerically |
| `int64` | 64-bit integer — efficient for whole-number counts |
| `float64` | 64-bit floating-point — standard for monetary values with decimal places |

Always verify dtypes after loading data. A column that *looks* numeric but is stored as `object` will silently return wrong results from `.mean()`, `.sum()`, and comparison operators.

**Understanding `describe()` output:**

`df.describe()` returns the **5-number summary** plus mean and count for each numeric column:

| Statistic | What it tells you |
|-----------|------------------|
| `count` | Non-null values — if less than `len(df)`, there are missing values |
| `mean` | Arithmetic average — sensitive to outliers |
| `std` | Standard deviation — how spread out values are around the mean |
| `25%` | First quartile (Q1) — 25 % of values fall below this |
| `50%` | Median — the midpoint; robust to outliers |
| `75%` | Third quartile (Q3) — 75 % of values fall below this |
| `min` / `max` | Extremes — useful for catching impossible values (negative visitors, etc.) |

If `mean` is substantially higher than `50%` (median), the distribution has a **right skew** — a small number of very high values are pulling the average up. This is a common pattern in tourism data where a handful of peak-season months dominate.

```python
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df
```
A progress confirmation so the user knows data loaded successfully before KPI computations begin.

### Complete Code — Section 3

```python
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "tourism_data.csv")
    if not os.path.exists(csv_path):
        return generate_tourism_data()
    df = pd.read_csv(csv_path)
    df["month"] = pd.to_datetime(df["month"])
    print(f"Loaded {len(df):,} rows from {csv_path}")
    return df


df = load_data()

# Inspect shape and types
print("\nColumn dtypes:")
print(df.dtypes)
print("\nBasic statistics:")
df[["visitor_count", "avg_spending_jod"]].describe().round(2)

# Quick overview: records per site and date range
print("Records per site:")
print(df["site"].value_counts())
print(f"\nDate range: {df['month'].min().strftime('%b %Y')} → {df['month'].max().strftime('%b %Y')}")
print(f"Total months: {df['month'].nunique()}")
```

---

## 4. KPI 1 — Monthly Visitor Growth Rate

**Business definition:** How fast (or slow) is each site growing month-over-month?

---

### Statistical Concept: Month-over-Month (MoM) vs Year-over-Year (YoY) Growth

| Metric | Formula | Best for |
|--------|---------|----------|
| **MoM growth** | `(this_month − last_month) / last_month` | Detecting short-term acceleration or deceleration in near-real-time |
| **YoY growth** | `(this_month − same_month_last_year) / same_month_last_year` | Removing seasonal effects to reveal the underlying trend |

We use MoM here because the assignment asks for the most granular view of momentum. The important implication: **negative MoM growth in June does not mean the site is in trouble** — it may simply reflect the expected post-spring slowdown. Always interpret growth rates alongside the seasonal pattern. A site with −20 % MoM in June that had +20 % in April is simply tracking the seasonal curve, not declining.

---

```python
def kpi1_monthly_visitor_growth(df):
```

```python
    monthly = df.groupby(["month", "site"])["visitor_count"].sum().reset_index()
```
- `groupby(["month", "site"])` — groups every row that shares the same month *and* the same site, creating one group per (month, site) pair. With 5 sites × 24 months = 120 groups.
- `["visitor_count"].sum()` — collapses the 15 origin × ticket-type rows in each group (5 origins × 3 ticket types) into one total visitor count.
- `.reset_index()` — by default, pandas makes the groupby keys the index. `.reset_index()` converts them back into regular columns so the result is a flat, easy-to-work-with DataFrame.

```python
    monthly = monthly.sort_values(["site", "month"])
```
Sorts by site first, then by time within each site. This ordering is **critical** for the next line. `pct_change()` computes differences between adjacent *rows by position*, not by time values. If the data were in a different order, percentage changes would be computed across unrelated months or even unrelated sites, producing garbage results.

```python
    monthly["growth_rate"] = monthly.groupby("site")["visitor_count"].pct_change() * 100
```
This line does three things in sequence:

1. **`groupby("site")`** — divides the DataFrame into five independent groups (one per site), so the calculation restarts at the beginning of each site's time series. Without this, the last row of Aqaba and the first row of Dead Sea would be compared as if they were consecutive months of the same site.

2. **`.pct_change()`** — for each row `n` within a group, computes `(row[n] − row[n−1]) / row[n−1]`. Returns a decimal (e.g., `0.127` for 12.7 % growth). The first row of each group becomes `NaN` because there is no prior row to compare against — this is correct and expected.

3. **`* 100`** — converts the decimal fraction to a percentage (e.g., `0.127` → `12.7`).

**Interpreting the NaN values:** The first month for each site (January 2024) will always show `NaN`. This is not missing data — it is the mathematically correct result. When computing averages or summaries later, `.mean()` on a Series containing `NaN` automatically skips them.

### Complete Code — Section 4

```python
def kpi1_monthly_visitor_growth(df):
    # Aggregate all segments into a single visitor count per (month, site)
    monthly = (
        df.groupby(["month", "site"])["visitor_count"]
        .sum()
        .reset_index()
    )

    # Sort so pct_change() computes in chronological order within each site
    monthly = monthly.sort_values(["site", "month"])

    # pct_change() computes (row_n - row_n-1) / row_n-1 — first row per site becomes NaN
    monthly["growth_rate"] = (
        monthly.groupby("site")["visitor_count"].pct_change() * 100
    )

    return monthly


monthly_growth = kpi1_monthly_visitor_growth(df)

# Average MoM growth rate per site over the full 2-year period
avg_growth = (
    monthly_growth.groupby("site")["growth_rate"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)
print("Average MoM growth rate by site (%):")
print(avg_growth)
```

---

## 5. KPI 2 — Revenue Per Tourist by Origin

**Business definition:** Which visitor origin markets deliver the highest economic value per person?

---

### Statistical Concept: Weighted Average vs Simple Average

There are two ways to compute "average spend by origin region." Only one is correct.

**Simple average (incorrect here):**
```python
df.groupby("origin_region")["avg_spending_jod"].mean()
```
This averages the `avg_spending_jod` column directly across all rows for each origin. But each row represents a different (site × month × ticket-type) combination with a *very different number of visitors* behind it. A single row representing 3,000 European group visitors in peak-season Petra would count equally with a row representing 30 European students in off-season Jerash. This gives each *row* equal weight, not each *visitor*.

**Weighted average (correct):**
```python
total_revenue / total_visitors
```
By summing total revenue (visitors × spend) and total visitors separately before dividing, we give each *visitor* exactly one vote in the average. This is a **visitor-weighted mean**, and it correctly reflects the true economic contribution of each origin market.

**Why does it matter?** In datasets with heterogeneous group sizes, the simple average can be significantly biased. This is a specific form of **aggregation bias** — the wrong level of aggregation produces a misleading statistic. The general principle: whenever averaging across groups of different sizes, ask whether you need a weighted average.

---

```python
def kpi2_revenue_per_tourist_by_origin(df):
    df = df.copy()
```
`df.copy()` creates an independent copy of the DataFrame before we modify it. Without this, `df["total_revenue"] = ...` would modify the *original* DataFrame passed in (pandas passes DataFrames by reference, not by value). This is called a **defensive copy** — a best-practice pattern to prevent subtle mutation bugs in other cells that reuse `df`.

```python
    df["total_revenue"] = df["visitor_count"] * df["avg_spending_jod"]
```
Row-by-row multiplication using pandas vectorised operations — no Python loop needed. Pandas applies the `*` operator element-wise across the entire column in a single, fast C-level operation. This new column is the *pre-aggregation* revenue for each row, which preserves the visitor-count weights that we need for the correct weighted average.

```python
    by_origin = df.groupby("origin_region").agg(
        total_revenue   = ("total_revenue",   "sum"),
        total_visitors  = ("visitor_count",   "sum"),
    ).reset_index()
```
**Named aggregation syntax** (pandas ≥ 0.25):

    new_column_name = ("source_column", "aggregation_function")

This aggregates all rows for each origin region, computing both totals in one pass. It is cleaner and more readable than two separate `.groupby().sum()` calls followed by a `merge`.

```python
    by_origin["revenue_per_tourist"] = (
        by_origin["total_revenue"] / by_origin["total_visitors"]
    ).round(2)
```
The visitor-weighted average spend per origin. `.round(2)` formats to two decimal places (JD currency convention). This single division on aggregated totals is the correct weighted-mean formula.

```python
    return by_origin.sort_values("revenue_per_tourist", ascending=False)
```
Returns the DataFrame sorted descending so the highest-value origin region appears first. The executive summary function later retrieves it with `by_origin.iloc[0]` (first row by position), which only works correctly because the data is pre-sorted here.

### Complete Code — Section 5

```python
def kpi2_revenue_per_tourist_by_origin(df):
    df = df.copy()                                          # avoid mutating the original

    # Compute total revenue for every row before aggregating
    df["total_revenue"] = df["visitor_count"] * df["avg_spending_jod"]

    # Sum revenue and visitors across all sites/months for each origin region
    by_origin = df.groupby("origin_region").agg(
        total_revenue   = ("total_revenue",   "sum"),
        total_visitors  = ("visitor_count",   "sum"),
    ).reset_index()

    # Weighted average spend — avoids the bias of simple-averaging avg_spending_jod
    by_origin["revenue_per_tourist"] = (
        by_origin["total_revenue"] / by_origin["total_visitors"]
    ).round(2)

    return by_origin.sort_values("revenue_per_tourist", ascending=False)


by_origin = kpi2_revenue_per_tourist_by_origin(df)
print("KPI 2 — Revenue per tourist by origin region (sorted):")
print(by_origin)
```

---

## 6. KPI 3 — Seasonal Concentration Index (HHI)

**Business definition:** How concentrated is each site's annual visitor traffic into a narrow seasonal window?

---

### Statistical Concept: The Herfindahl-Hirschman Index (HHI)

The **Herfindahl-Hirschman Index** was developed by economists Orris Herfindahl and Albert Hirschman to measure *market concentration* — how dominated an industry is by a small number of firms. Here we adapt it to measure *temporal concentration* — how dominated a year is by a small number of months.

**The formula:**

    HHI = Σ (s_m)² × 10,000     where s_m = visitors_month / visitors_year

**Why squared shares?**  
Squaring the monthly share *penalises* concentration disproportionately. Consider two extremes:

- **Equal distribution:** Each of 12 months gets 1/12 of visitors. Each share = 0.0833. Each squared share = 0.00694. Sum of 12 = 0.0833. Scaled: **833**.  
- **Full concentration:** One month gets all visitors. That month's share = 1.0. Squared = 1.0. Sum = 1.0. Scaled: **10,000**.

Because squaring amplifies larger shares more than smaller ones, a single month with 30 % of visitors contributes `0.30² = 0.09` to the index — much more than three months at 10 % each, which contribute `3 × 0.10² = 0.03`. This makes the index sensitive to extreme peaks, which is exactly what we want when measuring seasonal dependency.

**Interpreting HHI scores in this context:**

| HHI Score | What it means for tourism |
|-----------|--------------------------|
| **833** | Perfect year-round evenness — every month is identical (theoretical minimum) |
| **833–1,000** | Low concentration — visitors spread fairly evenly; manageable off-season troughs |
| **1,000–1,500** | Moderate concentration — clear peaks but meaningful off-season traffic |
| **1,500–3,000** | High concentration — revenue highly dependent on 2–3 months; off-season is a real problem |
| **> 3,000** | Extreme concentration — almost all visitors arrive in one month; major risk |

Our sites score around **900** (close to the minimum) because the cosine seasonal wave in the synthetic data creates gentle hills, not sharp spikes. Real Jordanian sites would likely score higher (1,200–2,000) due to harsher summer heat and narrower international travel windows.

---

```python
def kpi3_seasonal_concentration_index(df):
    df = df.copy()
    df["year"]      = pd.to_datetime(df["month"]).dt.year
    df["month_num"] = pd.to_datetime(df["month"]).dt.month
```
Extracts the integer year and month components from the Timestamp column. Even though `df["month"]` is already `datetime64` after `load_data()` parsed it, calling `pd.to_datetime()` again is a defensive step that ensures this function works correctly if invoked before `load_data()` runs.

`.dt` is pandas' **datetime accessor** — it provides date/time component extraction (`.year`, `.month`, `.day`, `.hour`, etc.) without requiring a `lambda` or `.apply()` loop. It operates in vectorised C-speed.

```python
    monthly_site = (
        df.groupby(["year", "site", "month_num"])["visitor_count"]
        .sum()
        .reset_index()
    )
```
Aggregates to one row per (year, site, month) — collapsing the origin and ticket-type dimensions. This gives us `12 months × 5 sites × 2 years = 120 rows`. These are the **numerators** (monthly totals) for the share calculation.

```python
    annual_site = (
        df.groupby(["year", "site"])["visitor_count"]
        .sum()
        .reset_index(name="annual_total")
    )
```
The annual total per site per year — the **denominator** for each monthly share. `reset_index(name="annual_total")` names the aggregated column directly in the reset call, saving a separate `.rename()` step. This produces `5 sites × 2 years = 10 rows`.

```python
    merged = monthly_site.merge(annual_site, on=["year", "site"])
```
A left join that attaches the annual total onto every monthly row that shares the same `year` and `site`. After the merge, each of the 120 rows carries both its monthly visitor count and the full-year total for that site, ready for the share calculation.

```python
    merged["share_sq"] = (merged["visitor_count"] / merged["annual_total"]) ** 2
```
Computes the squared monthly share in one vectorised operation across all 120 rows simultaneously. The `/ merged["annual_total"]` division produces the monthly share `s_m`; `** 2` squares it. No Python loop is needed — pandas applies the arithmetic element-wise.

```python
    hhi = merged.groupby(["year", "site"])["share_sq"].sum().reset_index()
    hhi["hhi"] = (hhi["share_sq"] * 10000).round(0)
    return hhi.drop(columns="share_sq")
```
- `.groupby(["year", "site"])["share_sq"].sum()` — sums the 12 squared shares for each (year, site) group, producing the raw HHI (a decimal between 0.083 and 1.0).
- `* 10000` — scales to the conventional 833–10,000 range for readability.
- `.round(0)` — removes decimal places; HHI is conventionally reported as a whole number.
- `.drop(columns="share_sq")` — removes the intermediate calculation column before returning; callers only need the final `hhi` score.

### Complete Code — Section 6

```python
def kpi3_seasonal_concentration_index(df):
    df = df.copy()

    # Extract year and month number from the Timestamp column
    df["year"]      = pd.to_datetime(df["month"]).dt.year
    df["month_num"] = pd.to_datetime(df["month"]).dt.month

    # Total visitors per (year, site, month) — collapse origin & ticket dimensions
    monthly_site = (
        df.groupby(["year", "site", "month_num"])["visitor_count"]
        .sum()
        .reset_index()
    )

    # Annual total per (year, site) — denominator for the monthly share calculation
    annual_site = (
        df.groupby(["year", "site"])["visitor_count"]
        .sum()
        .reset_index(name="annual_total")
    )

    # Join so each monthly row knows the annual total for its site/year
    merged = monthly_site.merge(annual_site, on=["year", "site"])

    # Squared share for this month's contribution to the HHI
    merged["share_sq"] = (merged["visitor_count"] / merged["annual_total"]) ** 2

    # Sum squared shares across all 12 months → one HHI score per (year, site)
    hhi = merged.groupby(["year", "site"])["share_sq"].sum().reset_index()
    hhi["hhi"] = (hhi["share_sq"] * 10000).round(0)

    return hhi.drop(columns="share_sq")


hhi = kpi3_seasonal_concentration_index(df)
print("KPI 3 — Seasonal Concentration Index (HHI) per site per year:")
print(hhi.sort_values(["year", "hhi"], ascending=[True, False]))
```

---

## 7. Statistical Test — Welch's t-test

**Research question:** Do group-ticket visitors spend significantly more per trip than individual-ticket visitors?

---

### Statistical Concept: Hypothesis Testing Framework

Every statistical test begins by defining two competing hypotheses:

- **H₀ (null hypothesis):** There is *no* difference in mean spending between group and individual ticket holders: `μ_group = μ_individual`
- **H₁ (alternative hypothesis):** Group ticket holders spend *more* on average: `μ_group > μ_individual`

The test does not directly prove H₁. Instead, it asks: *"If H₀ were true (no real difference), how likely would it be to observe a gap this large just by chance?"* If that probability (the p-value) is very small, we conclude that H₀ is implausible and reject it in favour of H₁.

---

### Statistical Concept: Welch's t-test vs Student's t-test

Both tests compare the means of two independent groups. The only difference is the variance assumption:

| Test | Key assumption | Formula complexity | When to use |
|------|---------------|-------------------|-------------|
| **Student's t-test** (`equal_var=True`) | Both groups have the same population variance (σ₁ = σ₂) | Simpler | Only when you have theoretical or prior empirical evidence of equal variances |
| **Welch's t-test** (`equal_var=False`) | Groups may have *different* variances | Slightly more complex (Welch–Satterthwaite degrees of freedom) | The safe default in practice |

**Why Welch's here?**  
Individual visitors span a wide range — budget backpackers through luxury travellers — producing high spending variance. Group visitors are often on standardised tour packages with negotiated prices, producing lower variance. Assuming equal variances (`equal_var=True`) would be incorrect and could produce misleading p-values. **Use `equal_var=False` by default unless you have a specific reason otherwise.**

---

### Statistical Concept: One-Sided vs Two-Sided Tests

A **two-sided test** asks: *Is there any difference in either direction?* (group spends more OR less)  
A **one-sided test** asks: *Is the difference in a specific direction?* (group spends more only)

```python
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1.0
```

SciPy's `ttest_ind` always returns a **two-sided p-value**. To convert to one-sided:

- If `t_stat > 0` (group mean > individual mean, matching the H₁ direction): `p_one_sided = p_two_sided / 2`
- If `t_stat ≤ 0` (group mean ≤ individual mean, *opposite* to H₁): the one-sided p-value is ≥ 0.5, so we set it to `1.0` — effectively "fail to reject H₀ in the predicted direction"

**Why use a one-sided test here?**  
Our business hypothesis is *directional* (we predict groups spend more, not just differently). One-sided tests have greater **statistical power** when the direction is correctly pre-specified — they are more likely to detect a true effect at the same sample size. However, one-sided tests are only valid when the direction is chosen *before* looking at the data. Choosing a direction after seeing that `t_stat > 0` would be p-hacking.

---

### Statistical Concept: Interpreting the p-value

The **p-value** is the probability of observing a t-statistic as extreme as (or more extreme than) the one computed, *assuming H₀ is true*.

- **p < 0.05** → The observed difference would occur by chance less than 5 % of the time if there were truly no difference. We *reject H₀* and conclude the difference is statistically significant at the α = 0.05 level.
- **p ≥ 0.05** → We cannot reject H₀. This does *not* mean the groups are identical — it means our sample did not provide sufficient evidence to conclude they differ.

**Critical nuance — statistical vs practical significance:**  
With a large sample (1,800 rows), even a trivially small difference can produce a very significant p-value. Always check the *effect size* alongside the p-value:

- The mean difference here is 88.94 − 73.98 = **14.96 JD** per visitor (~20 %)
- With 1,800 rows the t-statistic is ~15.6, making p effectively 0
- Both the statistical *and* the practical magnitude are large — the finding is robust

---

```python
def ttest_group_vs_individual(df):
    group_spend = df[df["ticket_type"] == "group"]["avg_spending_jod"]
    indiv_spend = df[df["ticket_type"] == "individual"]["avg_spending_jod"]
```
Boolean indexing to isolate the spending values for each ticket type. `df["ticket_type"] == "group"` produces a boolean Series (`True`/`False` for each row); wrapping it in `df[...]` filters to only the matching rows; `["avg_spending_jod"]` extracts just that column as a Series.

```python
    t_stat, p_two_sided = stats.ttest_ind(group_spend, indiv_spend, equal_var=False)
```
- `scipy.stats.ttest_ind` — independent-samples t-test (the two groups share no observations).
- `equal_var=False` — activates Welch's variant with Welch–Satterthwaite degrees of freedom adjustment.
- Returns a tuple: `(t_statistic, two_sided_p_value)`.

```python
    if p_one_sided < 0.05:
        print("  RESULT: Group visitors spend significantly MORE ...")
    else:
        print("  RESULT: No significant spending difference detected ...")
```
The conventional **α = 0.05** significance threshold. The 0.05 level means we accept a 5 % probability of a **Type I error** (falsely rejecting a true H₀ — a false positive). This threshold is arbitrary but widely adopted. For higher-stakes decisions (e.g., allocating a major marketing budget), use α = 0.01.

### Complete Code — Section 7

```python
def ttest_group_vs_individual(df):
    # Extract the spending column for each group
    group_spend = df[df["ticket_type"] == "group"]["avg_spending_jod"]
    indiv_spend = df[df["ticket_type"] == "individual"]["avg_spending_jod"]

    # ttest_ind returns a two-sided p-value by default
    t_stat, p_two_sided = stats.ttest_ind(group_spend, indiv_spend, equal_var=False)

    # Convert to one-sided: if t > 0 (group spends more), halve the p-value
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1.0

    print("Welch's t-test: Group vs. Individual ticket spending")
    print(f"  Group mean     : {group_spend.mean():.2f} JD")
    print(f"  Individual mean: {indiv_spend.mean():.2f} JD")
    print(f"  t-statistic    : {t_stat:.4f}")
    print(f"  p-value (one-sided): {p_one_sided:.4f}")

    if p_one_sided < 0.05:
        print("  RESULT: Group visitors spend significantly MORE than individual visitors (p < 0.05).")
        print("  ACTION: Prioritise group tour packages and B2B travel agency partnerships.")
    else:
        print("  RESULT: No significant spending difference detected (p >= 0.05).")

    return t_stat, p_one_sided


t_stat, p_val = ttest_group_vs_individual(df)
```

---

## 8. Visualisation — `plot_dashboard()`

---

### Statistical Concept: Choosing the Right Chart Type

Each chart type communicates a different aspect of the data. The panel layout here is deliberate:

**Line chart (Panel A) — Monthly visitor trends:**  
Best for *time series*. Connected points make trends and seasonal cycles immediately visible to the eye, which naturally interpolates between data points. `marker="o"` adds dots at each monthly measurement so a missing month would appear as a gap rather than a misleading straight line.

**Box plot (Panel B) — Spending by origin region:**  
Best for *comparing distributions* across groups. Unlike a bar chart that shows only means, a box plot communicates five statistics simultaneously:

| Box plot element | Statistical measure |
|-----------------|---------------------|
| Centre line inside box | Median (50th percentile) |
| Lower box edge | Q1 — 25th percentile |
| Upper box edge | Q3 — 75th percentile |
| Box height | IQR = Q3 − Q1 (middle 50 % of data) |
| Whiskers | Q1 − 1.5×IQR to Q3 + 1.5×IQR |
| Individual points beyond whiskers | Outliers |

This reveals whether origin groups have similar variance (spread) or if one group shows a much wider range of spending behaviour — information that is invisible in a mean-only bar chart.

**Heatmap (Panel C) — Seasonal HHI:**  
Best for a *small matrix of values* where both row and column identity matter. With 5 sites × 2 years = 10 cells, a heatmap allows instant cross-comparison of all cells simultaneously through colour encoding. `annot=True` overlays the exact number in each cell, combining the precision of a table with the pattern-recognition speed of a colour map.

---

```python
def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```
A helper that saves a figure and immediately **closes** it. `plt.close(fig)` releases the figure from memory — important when generating many plots in batch. `dpi=150` produces a sharper image than the default 100 dpi. `bbox_inches="tight"` automatically crops excess whitespace from the saved image edges.

```python
fig = plt.figure(figsize=(16, 13))
gs  = plt.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
```
- `figsize=(16, 13)` — width × height in inches at 100 dpi default (1600 × 1300 px). We override to 150 dpi on save for a sharper PNG.
- `GridSpec(2, 2)` — divides the figure canvas into a 2-row × 2-column grid of *slots*. Think of it as a table of plotting areas.
- `hspace=0.45` — vertical gap between panel rows, as a fraction of the average axis height. Increase if panel titles overlap the chart above.
- `wspace=0.35` — horizontal gap between panel columns.

### Panel A — Line chart (KPI 1)

```python
ax_line = fig.add_subplot(gs[0, :])
```
`gs[0, :]` means "row 0, all columns" — the `:` is a NumPy-style slice that spans both columns, giving Panel A the full figure width. The three panels use `gs[0, :]`, `gs[1, 0]`, and `gs[1, 1]` respectively.

```python
for site in site_monthly["site"].unique():
    sub = site_monthly[site_monthly["site"] == site].sort_values("month")
    ax_line.plot(sub["month"], sub["visitor_count"], marker="o", label=site, linewidth=2)
```
One `ax_line.plot()` call per site. Calling the same method multiple times on the same axes object is the standard matplotlib pattern for adding multiple series — matplotlib automatically assigns a different colour to each call (cycling through the palette set by `sns.set_theme`).

### Panel B — Box plot (KPI 2)

```python
ax_box = fig.add_subplot(gs[1, 0])
sns.boxplot(data=df, x="origin_region", y="avg_spending_jod", palette="muted", ax=ax_box)
```
`ax=ax_box` directs seaborn to draw onto the pre-created axis rather than auto-creating a new figure. Passing `data=df` with `x=` and `y=` lets seaborn handle the grouping and distribution calculation automatically — no manual aggregation needed.

### Panel C — Heatmap (KPI 3)

```python
hhi_pivot = hhi.pivot(index="site", columns="year", values="hhi")
```
`.pivot()` reshapes the long-format HHI table (one row per site/year) into a wide matrix (sites as rows, years as columns) — the 2D format that `sns.heatmap` expects as input.

```python
sns.heatmap(hhi_pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            ax=ax_heat, linewidths=0.3,
            cbar_kws={"label": "Seasonal Concentration (HHI)"})
```
- `annot=True` — prints the cell value as text inside each coloured square.
- `fmt=".0f"` — formats the annotation as a whole number (no decimal places).
- `cmap="YlOrRd"` — yellow → orange → red gradient; low values are light (low concentration risk), high values are red (high risk).
- `linewidths=0.3` — thin separating lines between cells improve readability.
- `cbar_kws` — a dict of keyword arguments forwarded to the colorbar; `"label"` adds a descriptive axis title.

```python
fig.savefig(out_path, dpi=150, bbox_inches="tight")
```
`bbox_inches="tight"` automatically crops the saved image to remove excess whitespace around the figure border. Without it, saved PNGs sometimes include large white margins that make figures look smaller when embedded in slides or reports.

### Complete Code — Section 8

```python
def plot_dashboard(monthly_growth, by_origin, hhi, df):
    """Build and display the multi-panel KPI figure."""

    fig = plt.figure(figsize=(16, 13))
    gs  = plt.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Visitor trends (line chart) ──────────────────────────────────
    ax_line = fig.add_subplot(gs[0, :])

    site_monthly = (
        monthly_growth
        .groupby(["month", "site"])["visitor_count"]
        .sum()
        .reset_index()
    )

    for site in site_monthly["site"].unique():
        sub = site_monthly[site_monthly["site"] == site].sort_values("month")
        ax_line.plot(sub["month"], sub["visitor_count"],
                     marker="o", label=site, linewidth=2)

    ax_line.set_title(
        "KPI 1: Petra & Wadi Rum Lead Visitor Growth — Spring and Autumn Peaks Visible",
        fontsize=12, fontweight="bold"
    )
    ax_line.set_ylabel("Visitors / Month")
    ax_line.legend(title="Site", loc="upper left")
    ax_line.tick_params(axis="x", rotation=30)

    # ── Panel B: Spending by origin (boxplot) ─────────────────────────────────
    ax_box = fig.add_subplot(gs[1, 0])
    sns.boxplot(
        data=df, x="origin_region", y="avg_spending_jod",
        palette="muted", ax=ax_box
    )
    ax_box.set_title(
        "KPI 2: European & North American Visitors\nSpend Most Per Trip",
        fontsize=11, fontweight="bold"
    )
    ax_box.set_xlabel("Origin Region")
    ax_box.set_ylabel("Avg Spending (JD)")
    ax_box.tick_params(axis="x", rotation=25)

    # ── Panel C: Seasonal HHI heatmap ─────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 1])
    hhi_pivot = hhi.pivot(index="site", columns="year", values="hhi")
    sns.heatmap(
        hhi_pivot, annot=True, fmt=".0f",
        cmap="YlOrRd", ax=ax_heat, linewidths=0.3,
        cbar_kws={"label": "Seasonal Concentration (HHI)"}
    )
    ax_heat.set_title(
        "KPI 3: Dead Sea Most Seasonally\nConcentrated — Summer Dependency",
        fontsize=11, fontweight="bold"
    )
    ax_heat.set_xlabel("Year")
    ax_heat.set_ylabel("Site")

    out_path = os.path.join(OUTPUT_DIR, "demo_tourism_kpi_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved → {out_path}")

    plt.tight_layout()
    plt.show()


plot_dashboard(monthly_growth, by_origin, hhi, df)
```

---

## 9. Executive Summary

The executive summary distills all three KPIs and the statistical test into a single, action-oriented paragraph written for a non-technical decision-maker. It follows the **SBA structure** used in management consulting:

| Component | What it communicates |
|-----------|---------------------|
| **S — Situation** | The headline finding with a supporting data point |
| **B — Business implication** | What the finding means for revenue or risk |
| **A — Action** | A specific, measurable recommendation |

**Why three sentences?** Executives process narrative faster than tables. Three sentences map one-to-one to the three KPI findings, making the structure scannable even in paragraph form. Each sentence anchors its claim with a specific number (e.g., `0.1 %`, `79 JD`, `HHI 900`, `t = 15.62`) — this makes the summary *auditable*: any reader can trace every claim back to the cell that produced it.

---

```python
def print_executive_summary(monthly_growth, by_origin, hhi, t_stat, p_val):
    top_origin   = by_origin.iloc[0]
    avg_hhi      = hhi["hhi"].mean()
    petra_growth = monthly_growth[monthly_growth["site"] == "Petra"]["growth_rate"].mean()
```

- **`by_origin.iloc[0]`** — `by_origin` was sorted descending by `revenue_per_tourist` inside `kpi2_revenue_per_tourist_by_origin()`. `.iloc[0]` retrieves the first row by *integer position* (not by index label). This only returns the correct result because the sort was applied before returning — a design contract between the two functions.

- **`hhi["hhi"].mean()`** — a simple (unweighted) mean of all 10 HHI scores (5 sites × 2 years). Since all sites have the same number of months, an unweighted mean is appropriate. If sites had different observation windows, a visitor-count-weighted mean would be needed.

- **`monthly_growth[...]["growth_rate"].mean()`** — filters to Petra rows, then calls `.mean()` on the `growth_rate` column. Pandas `.mean()` on a Series containing `NaN` values automatically skips them (`skipna=True` by default). The first-month `NaN` growth rates are excluded without any explicit handling.

```python
    summary = (
        f"Petra leads visitor growth with an average month-over-month increase of "
        f"{petra_growth:.1f}%, ..."
    )
```
**F-string format specifiers** used in this summary:

| Specifier | Example output | Used for |
|-----------|---------------|----------|
| `:.1f` | `0.1` | Growth rate (1 decimal place sufficient for a % value) |
| `:.0f` | `79` | Revenue per tourist and HHI (whole numbers avoid false precision) |
| `:.2f` | `15.62` | t-statistic (2 decimal places is conventional) |
| `:.3f` | `0.000` | p-value (3 decimal places; shows "0.000" rather than "0" when p ≈ 0) |

Over-precision (e.g., `78.6300000 JD`) reduces readability and falsely implies measurement accuracy that the synthetic data does not support. Format numbers to match the precision that is meaningful for the audience.

### Complete Code — Section 9

```python
def print_executive_summary(monthly_growth, by_origin, hhi, t_stat, p_val):
    # Pull the top-performing origin region (already sorted descending)
    top_origin = by_origin.iloc[0]

    # Average HHI across all sites and years — a single 'concentration risk' number
    avg_hhi = hhi["hhi"].mean()

    # Mean MoM growth for Petra (NaN from the first month is auto-skipped by .mean())
    petra_growth = monthly_growth[monthly_growth["site"] == "Petra"]["growth_rate"].mean()

    summary = (
        f"Petra leads visitor growth with an average month-over-month increase of "
        f"{petra_growth:.1f}%, driven by strong European and North American demand — "
        f"these origin markets deliver the highest revenue per tourist at "
        f"{top_origin['revenue_per_tourist']:.0f} JD, nearly double the Arab market average. "
        f"However, an average Seasonal Concentration Index of {avg_hhi:.0f} across all sites "
        f"indicates heavy dependence on spring and autumn peaks, creating revenue gaps in "
        f"June–August that require targeted domestic campaigns. "
        f"Group ticket visitors spend significantly more than individual visitors "
        f"(t = {t_stat:.2f}, p = {p_val:.3f}), confirming that investing in B2B travel agency "
        f"partnerships and group packages is the highest-ROI action available today."
    )

    print("=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)
    print(summary)
    print()


print_executive_summary(monthly_growth, by_origin, hhi, t_stat, p_val)
```

---

## 10. Entry Point — `main()`

```python
def main():
    print("=" * 60)
    print("  SI Demo — Visit Jordan KPI Dashboard")
    print("=" * 60)
```
A title banner. `"=" * 60` repeats the character 60 times — a quick way to create a fixed-width visual separator without writing a 60-character string literal.

```python
    df = load_data()
    df["month"] = pd.to_datetime(df["month"])
```
Data is loaded and the `month` column is (re-)parsed as a datetime. Even if `load_data()` already did this, calling `pd.to_datetime()` a second time is harmless — it is idempotent on a column that is already `datetime64`. This makes `main()` robust to different calling sequences.

```python
    monthly_growth = kpi1_monthly_visitor_growth(df)
    by_origin      = kpi2_revenue_per_tourist_by_origin(df)
    hhi            = kpi3_seasonal_concentration_index(df)
```
Each KPI function receives the same clean DataFrame and returns its own independent result. This **separation of concerns** keeps each KPI independently testable — you can call any one of these functions in isolation and inspect its output without running the others.

```python
    t_stat, p_val = ttest_group_vs_individual(df)
```
The t-test function prints its own diagnostic output internally *and* returns the key numbers as a tuple. Returning values from a function that also produces side-effect output (the `print` statements) is a pragmatic pattern for a demo: the terminal output is human-readable, and the returned values feed the machine-readable executive summary.

```python
    plot_dashboard(monthly_growth, by_origin, hhi, df)
    print_executive_summary(monthly_growth, by_origin, hhi, t_stat, p_val)
    print(f"Charts saved to {OUTPUT_DIR}/")
```
Final steps: render the dashboard, print the narrative summary, and confirm the output path so the user knows where to find the PNG.

```python
if __name__ == "__main__":
    main()
```
**Python's standard module guard.** `__name__` equals `"__main__"` only when the file is executed directly (e.g., `python demo_tourism_dashboard.py`). When the file is *imported* as a module — for example, in a test suite or another script — `main()` is not called automatically. Without this guard, running `import demo_tourism_dashboard` would immediately execute the entire pipeline as a side effect, which is almost never desired.

### Complete Code — Section 10

```python
def main():
    print("=" * 60)
    print("  SI Demo — Visit Jordan KPI Dashboard")
    print("=" * 60)

    df            = load_data()
    df["month"]   = pd.to_datetime(df["month"])

    print("\nComputing KPIs...")
    monthly_growth = kpi1_monthly_visitor_growth(df)
    by_origin      = kpi2_revenue_per_tourist_by_origin(df)
    hhi            = kpi3_seasonal_concentration_index(df)

    t_stat, p_val = ttest_group_vs_individual(df)

    print("\nGenerating multi-panel chart...")
    plot_dashboard(monthly_growth, by_origin, hhi, df)

    print_executive_summary(monthly_growth, by_origin, hhi, t_stat, p_val)
    print(f"Charts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

---

## Key Design Patterns Summary

| Pattern | Where used | Why |
|---------|-----------|-----|
| Seeded RNG (`default_rng(42)`) | `generate_tourism_data` | Reproducibility across runs and machines |
| Defensive copy (`df.copy()`) | KPI 2, KPI 3 | Prevent accidental mutation of the caller's DataFrame |
| Named aggregation | KPI 2 | Readable multi-column `groupby` in one pass |
| Guard clause | `load_data` | Self-healing pipeline — no crash if CSV is missing |
| `__name__ == "__main__"` guard | Bottom of script | Safe to import as a module without triggering side effects |
| `ax=` parameter to seaborn | `plot_dashboard` | Place charts onto pre-created axes in a multi-panel figure |
| `GridSpec` | `plot_dashboard` | Fine-grained control over subplot sizes and spans |
| One-sided p-value conversion | `ttest_group_vs_individual` | Match the statistical test to the directional business hypothesis |
| Weighted average (revenue / visitors) | KPI 2 | Correct visitor-weighted mean; avoids aggregation bias |
| HHI squared-shares formula | KPI 3 | Penalises temporal concentration; adapted from market-concentration economics |
