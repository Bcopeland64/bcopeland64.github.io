# Module 2 Applied Lab — Code Walkthrough

This document walks every line of [pipeline.py](pipeline.py). It is a six-function agricultural data pipeline (crop yields in Jordan) built as the Support Instructor demo for M2-L2 — it uses the same technical operations as the learner assignment (electronics retail sales) but a different domain, so it can be shown live without being copyable as a submission. Read alongside the script.

---

## 0. Setup — module docstring, imports, and the matplotlib backend

### Why this section exists
Before any function runs, the script needs its dependencies loaded and, critically, needs to tell matplotlib not to try to open a GUI window. Get this wrong and the pipeline crashes the instant `create_visualizations` runs on a headless machine (CI runners, cloud VMs, most instructor demo laptops during screen share).

### Line-by-line

- `"""pipeline.py — Agricultural Crop Yield Data Pipeline ..."""` — the module docstring, a triple-quoted string as the very first statement in the file. Python attaches this to `pipeline.__doc__`, so anyone who runs `import pipeline; help(pipeline)` sees this description. It states the file's purpose, that it's for Support Instructor use only, and the domain (Jordan farming records) — all useful orientation for a reader who opens the file with no other context.
- `import os` — the standard library's OS-interface module. Used twice later: `os.path.exists(filepath)` in `load_data` (check a file is really there before trying to read it) and `os.makedirs(output_dir, exist_ok=True)` in `create_visualizations` (create the output folder before saving into it).
- `import sys` — imported but not referenced anywhere else in the file. It costs nothing at runtime (the module is already cached after the first import anywhere in the process) and doesn't change behavior; it's effectively unused boilerplate. Worth noting because a reader scanning for `sys.argv` or `sys.exit()` usage elsewhere in the script won't find any — this import can be safely removed without affecting the pipeline.
- `import pandas as pd` — the DataFrame library every other function in this file depends on. `pd` is the near-universal alias convention, so matching it here means the code reads the same as any other pandas codebase.
- `import matplotlib` — imported on its own line, separately from `matplotlib.pyplot`, specifically so the next line can configure the backend **before** pyplot is loaded.
- `matplotlib.use("Agg")` — switches matplotlib to the "Agg" (Anti-Grain Geometry) backend, a non-interactive renderer that draws straight to raster files (PNG, etc.) instead of opening a screen window. This line must run before `import matplotlib.pyplot as plt` — pyplot picks its backend at import time, and once it has picked a GUI backend (the default on most systems), you cannot switch to Agg after the fact. Skip this line on a headless server or CI runner and the default backend tries to open a display, throwing an error before a single chart is drawn. The inline comment (`# Non-interactive backend — works on all platforms`) flags exactly why this exists.
- `import matplotlib.pyplot as plt` — the plotting API used throughout `create_visualizations` (`plt.subplots`, `plt.tight_layout`, `plt.close`). Because it's imported after `matplotlib.use("Agg")`, it inherits the Agg backend automatically.

### Full code block

```python
"""
pipeline.py — Agricultural Crop Yield Data Pipeline
Demo pipeline for M2-L2 (Support Instructor use only).
Domain: Regional farming records in Jordan.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works on all platforms
import matplotlib.pyplot as plt
```

---

## 1. `load_data`

### Why this section exists
The pipeline's single point of contact with the outside world (the CSV file). Keeping it to "read the file and hand back a DataFrame — nothing else" means every other function can be tested with an in-memory DataFrame instead of a real file on disk, and a broken file path fails immediately with a clear message instead of a confusing pandas traceback three functions later.

### Line-by-line

- `# --- 1. load_data ---` (comment banner) — a visual divider marking the start of this function's block. Because `pipeline.py` is a plain script rather than a notebook, there are no cells to separate logical sections — these ASCII banners are the stand-in, making it easy to jump to a given function while scrolling.
- `def load_data(filepath: str = "demo_data/crop_yields.csv") -> pd.DataFrame:` — the function signature. The type hints (`filepath: str`, `-> pd.DataFrame`) document the contract for readers and IDEs without enforcing anything at runtime (Python doesn't check hints unless you add a separate tool like `mypy`). The default argument means `load_data()` with no arguments "just works" for the common case, while tests can still override it with `load_data("some/other/path.csv")`.
- The docstring (`"""Load the crop yield CSV into a DataFrame. ..."""`) — Google-style docstring with `Args`, `Returns`, and `Raises` sections. Documenting `Raises: FileNotFoundError` up front tells a caller exactly what to catch or guard against without having to read the function body.
- `if not os.path.exists(filepath):` — checks the path exists *before* attempting to read it. This is a deliberate, explicit guard rather than relying on `pd.read_csv` to fail on its own.
- `raise FileNotFoundError(f"Dataset not found at '{filepath}'. " "Make sure demo_data/crop_yields.csv is present.")` — raises the same exception type pandas would eventually raise, but with a custom, actionable message. The two string literals on adjacent lines are concatenated automatically by Python (implicit string concatenation — no `+` needed) into one message. Without this guard, a missing file would still raise `FileNotFoundError`, but with pandas' generic C-parser error text, which is harder for a learner to diagnose than "make sure demo_data/crop_yields.csv is present."
- `df = pd.read_csv(filepath)` — reads the CSV into a DataFrame, inferring a dtype for each column (numbers become `int64`/`float64`, text stays `object`) by scanning the file's contents. Missing cells (like the blank `yield_tons` value in some rows of `crop_yields.csv`) become `NaN` automatically at this step.
- `print(f"[load_data] Loaded {len(df)} rows × {len(df.columns)} columns.")` — a status line prefixed with `[load_data]`. Every function in this file follows this same `[function_name]` prefix convention for its prints, which turns plain stdout into a lightweight, greppable log even though there's no real logging framework wired up. `len(df)` is the row count; `len(df.columns)` is the column count.
- `return df` — hands back the **raw, unmodified** DataFrame. `load_data` intentionally does no cleaning or feature work — that separation is the whole point of the six-function design (each function is one responsibility, one unit that can be tested and reasoned about in isolation).

### Full code block

```python
# ---------------------------------------------------------------------------
# 1. load_data
# ---------------------------------------------------------------------------

def load_data(filepath: str = "demo_data/crop_yields.csv") -> pd.DataFrame:
    """Load the crop yield CSV into a DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame with original column names preserved.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Make sure demo_data/crop_yields.csv is present."
        )
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {len(df)} rows × {len(df.columns)} columns.")
    return df
```

---

## 2. `clean_data`

### Why this section exists
Raw data is messy: the `season` column mixes casing (`"summer 2024"` vs `"Summer 2024"`), some `yield_tons` values are blank, and a few `area_hectares` values are missing or zero/negative (which would make later division nonsensical). This function is where every one of those problems gets fixed, in one auditable place, before anything downstream trusts the numbers.

### Line-by-line

- `# --- 2. clean_data ---` (comment banner) — same visual-divider convention as section 1.
- `def clean_data(df: pd.DataFrame) -> pd.DataFrame:` — takes the raw DataFrame from `load_data` and returns a cleaned one. It does not take a `filepath` — this function only knows about DataFrames, which is exactly why it can be unit-tested with a small hand-built DataFrame instead of a real CSV.
- The docstring — lists the three operations performed (season formatting, missing yield fill, bad-area drop) so a reader gets the "what" before diving into the "how" of the line-by-line code.
- `df = df.copy()` — makes an explicit copy of the incoming DataFrame before mutating it. Without this, the assignments below (`df["season"] = ...`, etc.) could mutate the caller's original DataFrame in place — since pandas DataFrames are passed by reference, modifying `df` here would otherwise be visible to whatever code called `clean_data(df_raw)`, a classic source of hard-to-trace bugs. `.copy()` also avoids pandas' `SettingWithCopyWarning`, which fires when you write to a DataFrame that might be a view into another one.
- `df["season"] = df["season"].str.strip().str.title()` — the `.str` accessor lets you run vectorized string methods across an entire column at once. `.str.strip()` removes leading/trailing whitespace; `.str.title()` converts to Title Case (`"summer 2024"` → `"Summer 2024"`). This directly fixes the inconsistency visible in the raw CSV — the header/first rows show both `"Spring 2024"` (already title case) and lowercase variants elsewhere — which would otherwise cause `groupby("season")` to silently treat `"summer 2024"` and `"Summer 2024"` as two different groups.
- `missing_before = df["yield_tons"].isna().sum()` — `.isna()` returns a boolean Series (`True` where the value is `NaN`), and `.sum()` on a boolean Series counts the `True`s (Python treats `True` as `1`). This captures the missing-value count *before* the fill, purely so it can be reported in the print statement below.
- `df["yield_tons"] = df.groupby("crop_type")["yield_tons"].transform(lambda s: s.fillna(s.median()))` — the core imputation step. `groupby("crop_type")` splits the DataFrame into one group per crop type (Tomatoes, Olives, etc.); `["yield_tons"]` selects just that column within each group; `.transform(...)` applies the lambda to each group's Series and — unlike `.agg()`, which collapses each group to one value — returns a result the same length as the original, correctly aligned back to every row's original index. The lambda itself, `lambda s: s.fillna(s.median())`, computes that group's median (ignoring existing `NaN`s) and fills only the missing entries with it. Net effect: a missing Olives yield gets filled with the median of *other* Olives rows, not the median of the whole dataset — a more accurate estimate than a single global fill.
- `# Fallback: if a crop group consists entirely of NaNs, use the global median` — a comment explaining the edge case handled by the next two lines: if every row for some crop type happened to have a missing `yield_tons`, that group's own median would itself be `NaN` (median of an all-`NaN` Series is `NaN`), so the `transform` call above would leave those rows still missing.
- `global_median = df["yield_tons"].median()` — computes one median across the entire (partially filled) column as the fallback value.
- `df["yield_tons"] = df["yield_tons"].fillna(global_median)` — a second `fillna` pass that mops up any values the group-level fill couldn't fix. `fillna` only touches cells that are still `NaN`, so already-filled values are untouched.
- `missing_after = df["yield_tons"].isna().sum()` — recount after both fill passes, expected to be `0`.
- `print(f"[clean_data] Filled {missing_before - missing_after} missing yield_tons values using crop-type median.")` — reports how many values were actually filled, using the same `[function_name]` print-prefix convention as `load_data`.
- `before = len(df)` — captures the row count before the next filtering step, again purely for the print statement that follows.
- `df = df[df["area_hectares"].notna() & (df["area_hectares"] > 0)]` — boolean-mask filtering: `df["area_hectares"].notna()` is `True` where the value isn't missing, `(df["area_hectares"] > 0)` is `True` where the value is a physically sensible positive number, and `&` is pandas' elementwise **AND** (not Python's `and`, which doesn't work element-by-element on a Series — this is why the two conditions are wrapped in parentheses, since `&` binds tighter than `>` and would otherwise parse incorrectly without them). Indexing `df[...]` with the resulting boolean Series keeps only rows where both conditions are `True`, dropping rows with missing or non-positive farm area — values that would make `yield_per_hectare` undefined or nonsensical in `add_features`.
- `print(f"[clean_data] Dropped {before - len(df)} rows with invalid area_hectares.")` — reports how many rows were removed by the filter above.
- `return df.reset_index(drop=True)` — after filtering out rows, the DataFrame's index has gaps (e.g., `0, 1, 3, 4, ...` if row `2` was dropped). `reset_index()` renumbers the index sequentially from `0`; `drop=True` discards the old, gap-filled index instead of inserting it as a new column (the default behavior without `drop=True`).

### Full code block

```python
# ---------------------------------------------------------------------------
# 2. clean_data
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw crop yield DataFrame.

    Operations performed:
    - Standardise season strings to Title Case.
    - Fill missing yield_tons with the per-crop-type median.
    - Drop rows where area_hectares is missing or non-positive.

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # --- Standardise season formatting ---
    df["season"] = df["season"].str.strip().str.title()

    # --- Fill missing yield_tons with crop-type median ---
    missing_before = df["yield_tons"].isna().sum()
    df["yield_tons"] = df.groupby("crop_type")["yield_tons"].transform(
        lambda s: s.fillna(s.median())
    )
    # Fallback: if a crop group consists entirely of NaNs, use the global median
    global_median = df["yield_tons"].median()
    df["yield_tons"] = df["yield_tons"].fillna(global_median)
    missing_after = df["yield_tons"].isna().sum()
    print(
        f"[clean_data] Filled {missing_before - missing_after} missing "
        f"yield_tons values using crop-type median."
    )

    # --- Drop rows with bad area values ---
    before = len(df)
    df = df[df["area_hectares"].notna() & (df["area_hectares"] > 0)]
    print(f"[clean_data] Dropped {before - len(df)} rows with invalid area_hectares.")

    return df.reset_index(drop=True)
```

---

## 3. `add_features`

### Why this section exists
Raw and cleaned columns aren't yet analysis-ready. This function derives the two engineered columns every later function depends on: a normalized productivity metric (`yield_per_hectare`) and a simple boolean flag (`is_irrigated`) that makes groupby filtering trivial. This is the same feature-engineering pattern (division for normalization, categorical-to-boolean mapping) learners will reuse in every ML preprocessing pipeline afterward.

### Line-by-line

- `# --- 3. add_features ---` (comment banner) — section divider, same convention as above.
- `def add_features(df: pd.DataFrame) -> pd.DataFrame:` — takes the cleaned DataFrame and returns it with two new columns. Like `clean_data`, it only touches DataFrames, so it's independently testable.
- The docstring — names the two new columns and their formulas up front.
- `df = df.copy()` — same defensive-copy rationale as in `clean_data`: prevents mutating the caller's DataFrame in place and avoids `SettingWithCopyWarning`.
- `df["yield_per_hectare"] = df["yield_tons"] / df["area_hectares"]` — a vectorized elementwise division: pandas divides each row's `yield_tons` by that same row's `area_hectares` and assigns the result as a new column. This normalizes production by farm size, so a 2-hectare plot and a 40-hectare plot become directly comparable — the raw `yield_tons` numbers alone are not comparable across wildly different farm sizes.
- `df["is_irrigated"] = df["irrigation_method"].str.lower() != "rainfed"` — `.str.lower()` lowercases every value in `irrigation_method` first, so the comparison `!= "rainfed"` is case-insensitive regardless of how the source data capitalized it (e.g., `"Rainfed"`, `"RAINFED"`, `"rainfed"` all lowercase to the same string). The `!=` comparison against a Series produces a boolean Series: `True` for any irrigation method that isn't rainfed (`"Sprinkler"`, `"Drip"`, `"Flood"`), `False` for rainfed. This turns a free-text categorical column into a clean boolean that's easy to filter or group by later (used directly in `create_visualizations`'s color-mapping for the rainfall scatter plot).
- `print("[add_features] Added 'yield_per_hectare' and 'is_irrigated' columns.")` — status line, same `[function_name]` print convention.
- `return df` — returns the DataFrame with both new columns attached; the two original columns they were derived from (`yield_tons`, `area_hectares`, `irrigation_method`) remain untouched.

### Full code block

```python
# ---------------------------------------------------------------------------
# 3. add_features
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive analytical features from the cleaned DataFrame.

    New columns added:
    - yield_per_hectare: yield_tons / area_hectares
    - is_irrigated: True if irrigation_method is not 'Rainfed'

    Args:
        df: Cleaned DataFrame from clean_data().

    Returns:
        DataFrame with two additional columns.
    """
    df = df.copy()
    df["yield_per_hectare"] = df["yield_tons"] / df["area_hectares"]
    df["is_irrigated"] = df["irrigation_method"].str.lower() != "rainfed"
    print("[add_features] Added 'yield_per_hectare' and 'is_irrigated' columns.")
    return df
```

---

## 4. `generate_summary`

### Why this section exists
Someone reading the pipeline's output on a terminal (or a PR description, per the instructor notes) needs headline numbers without opening a chart. This function computes three summary statistics, prints them in a readable block, and — critically — also returns them as a dict so calling code (tests, `main()`, a notebook) can use the numbers programmatically instead of scraping stdout.

### Line-by-line

- `# --- 4. generate_summary ---` (comment banner) — section divider.
- `def generate_summary(df: pd.DataFrame) -> dict:` — takes the fully feature-enriched DataFrame and returns a `dict`, not `None` — this is the first function in the file whose primary output is a return value rather than just a printed side effect.
- The docstring — lists the three printed metrics and documents the exact dict keys the return value will contain (`total_production`, `top_crop`, `avg_yield_per_hectare`), so a caller doesn't have to read the function body to know what keys to expect.
- `total_production = round(df["yield_tons"].sum(), 2)` — `.sum()` adds up every row's `yield_tons` into a single number; `round(..., 2)` rounds it to 2 decimal places for a clean tons figure.
- `avg_yield_per_hectare = round(df["yield_per_hectare"].mean(), 4)` — `.mean()` averages the `yield_per_hectare` column across all rows; rounded to 4 decimal places rather than 2, since these values are typically well under 10 and the extra precision keeps small differences visible.
- `top_crop = df.groupby("crop_type")["yield_per_hectare"].mean().idxmax()` — groups by `crop_type`, averages `yield_per_hectare` within each group (same pattern used in `clean_data`'s fill and later reused in `create_visualizations`), then calls `.idxmax()`. This is the non-obvious part: `.idxmax()` returns the **index label** of the row with the maximum value (here, the crop type name itself, e.g. `"Olives"`), not the maximum value — that's what `.max()` would give you. Using the wrong one here would return a number instead of a crop name.
- `summary = {"total_production": total_production, "top_crop": top_crop, "avg_yield_per_hectare": avg_yield_per_hectare}` — bundles the three computed values into a single dict with descriptive keys. This dict is both what gets printed below and what the function ultimately returns — one source of truth for the numbers, rather than recomputing them in two places.
- `print("\n========== SUMMARY ==========")` — the leading `\n` inserts a blank line before the header for visual spacing when this prints after `add_features`'s own status lines.
- `print(f"  Total production (tons)      : {total_production:,.2f}")` — `{total_production:,.2f}` is a format spec: `,` inserts a thousands separator (e.g., `12,345.67`), `.2f` fixes it to 2 decimal places. The extra spaces before `:` in the literal string are hand-aligned so the colons line up across all three printed lines.
- `print(f"  Top crop by yield/hectare    : {top_crop}")` — printed as plain text (a crop name), no numeric formatting needed.
- `print(f"  Average yield per hectare    : {avg_yield_per_hectare:.4f}")` — `.4f` forces exactly 4 decimal places (padding with trailing zeros if needed), matching the precision used when the value was rounded above.
- `print("=================================\n")` — closing divider line; the trailing `\n` adds a blank line after the block, separating it from whatever prints next (the visualization save-confirmation lines).
- `return summary` — hands the dict back to the caller. In `main()`, this becomes the pipeline's overall return value.

### Full code block

```python
# ---------------------------------------------------------------------------
# 4. generate_summary
# ---------------------------------------------------------------------------

def generate_summary(df: pd.DataFrame) -> dict:
    """Compute and print key summary statistics.

    Printed metrics:
    - Total production (tons) across all crops.
    - Top crop by average yield per hectare.
    - Average yield per hectare across all rows.

    Args:
        df: Feature-enriched DataFrame from add_features().

    Returns:
        Dictionary with keys: total_production, top_crop, avg_yield_per_hectare.
    """
    total_production = round(df["yield_tons"].sum(), 2)
    avg_yield_per_hectare = round(df["yield_per_hectare"].mean(), 4)
    top_crop = df.groupby("crop_type")["yield_per_hectare"].mean().idxmax()

    summary = {
        "total_production": total_production,
        "top_crop": top_crop,
        "avg_yield_per_hectare": avg_yield_per_hectare,
    }

    print("\n========== SUMMARY ==========")
    print(f"  Total production (tons)      : {total_production:,.2f}")
    print(f"  Top crop by yield/hectare    : {top_crop}")
    print(f"  Average yield per hectare    : {avg_yield_per_hectare:.4f}")
    print("=================================\n")

    return summary
```

---

## 5. `create_visualizations`

### Why this section exists
Numbers in a terminal only go so far — this function turns the same feature-enriched DataFrame into three saved PNG charts, giving both the instructor and CI a visual, file-based artifact proving the pipeline ran end-to-end (per the instructor notes' deliverable table: "output/ (3 PNG files) — Charts proving the pipeline ran end-to-end in CI").

### Line-by-line

- `# --- 5. create_visualizations ---` (comment banner) — section divider.
- `def create_visualizations(df: pd.DataFrame, output_dir: str = "output") -> None:` — takes the enriched DataFrame plus an output directory (defaulting to `"output"`). The `-> None` hint makes explicit that this function's job is side effects (writing files to disk), not producing a value to use downstream — unlike `generate_summary`, nothing calls `create_visualizations(...)` expecting a return.
- The docstring — names all three output filenames and what each chart shows, so a reader knows what to expect in `output/` without running the code.
- `os.makedirs(output_dir, exist_ok=True)` — creates the output directory (and any missing parent directories) if it doesn't already exist. `exist_ok=True` means "don't raise an error if the directory is already there" — without it, running the pipeline a second time would raise `FileExistsError` on this exact line. Without this call at all, the first `fig.savefig(...)` below would raise `FileNotFoundError` because matplotlib does not create missing directories for you.

**Chart 1 — yield by crop type:**
- `fig, ax = plt.subplots(figsize=(8, 5))` — creates one Figure and one Axes object in a single call, sized 8×5 inches. This is matplotlib's object-oriented API (as opposed to the implicit `plt.plot(...)` state-machine style) — using explicit `fig`/`ax` handles means each chart's figure can be individually closed later with `plt.close(fig)`, which matters in a script that builds three charts back-to-back without leaking memory.
- `crop_avg = df.groupby("crop_type")["yield_per_hectare"].mean().sort_values(ascending=False)` — the same groupby-mean pattern from `generate_summary`, computed independently here (each function is self-contained rather than sharing state), then `.sort_values(ascending=False)` orders the resulting Series from highest to lowest average yield — so the bar chart reads left-to-right as "best crop first."
- `crop_avg.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")` — pandas' `.plot()` is a convenience wrapper around matplotlib; `kind="bar"` selects a bar chart, `ax=ax` tells it to draw onto our specific Axes instead of creating a new implicit figure, and `edgecolor="white"` outlines each bar in white so adjacent bars of the same color stay visually distinct.
- `ax.set_title("Average Yield per Hectare by Crop Type", fontsize=14, fontweight="bold")` — sets the chart title with explicit font size and boldness for readability, especially when projected on a screen.
- `ax.set_xlabel("Crop Type")` / `ax.set_ylabel("Yield per Hectare (tons/ha)")` — axis labels, including units in the y-axis label so the chart is self-explanatory without a caption.
- `ax.tick_params(axis="x", rotation=30)` — rotates the x-axis tick labels (crop type names) by 30 degrees, preventing longer names from overlapping each other along the axis.
- `plt.tight_layout()` — automatically adjusts subplot spacing/padding so titles, labels, and rotated tick text don't get clipped or overlap the plot area.
- `path1 = os.path.join(output_dir, "yield_by_crop.png")` — builds the output path using `os.path.join`, which inserts the correct path separator (`/` on Linux/macOS, `\` on Windows) automatically, keeping the script cross-platform.
- `fig.savefig(path1, dpi=120)` — writes the figure to disk as a PNG at 120 dots-per-inch (higher than matplotlib's default ~100 dpi, for a crisper image).
- `plt.close(fig)` — releases this figure from matplotlib's internal figure registry. Without closing each figure explicitly, generating many charts in one script run accumulates open figures in memory and can eventually trigger matplotlib's "too many open figures" warning.
- `print(f"[create_visualizations] Saved: {path1}")` — status line, same print-prefix convention as every other function.

**Chart 2 — rainfall vs. yield:**
- `fig, ax = plt.subplots(figsize=(8, 5))` — a fresh Figure/Axes pair for the second chart; the previous figure was already closed, so this cannot reuse it.
- `colors = df["is_irrigated"].map({True: "steelblue", False: "coral"})` — `.map()` with a dict translates each boolean value in `is_irrigated` (engineered back in `add_features`) into a color name, producing a per-row color Series aligned with the DataFrame. This is a direct callback to the feature engineered earlier — a visible payoff for why that column exists.
- `ax.scatter(df["rainfall_mm"], df["yield_per_hectare"], c=colors, alpha=0.6, edgecolors="none")` — plots one point per row: x = rainfall in mm, y = yield per hectare. `c=colors` passes the per-point color array built above, so irrigated and rainfed farms render in different colors on the same chart. `alpha=0.6` makes points 60% opaque, so overlapping points in dense regions are still visually distinguishable rather than forming a single solid blob. `edgecolors="none"` removes the marker border, keeping the plot clean with many overlapping points.
- `ax.set_title("Rainfall vs Yield per Hectare", fontsize=14, fontweight="bold")` / `ax.set_xlabel("Rainfall (mm)")` / `ax.set_ylabel("Yield per Hectare (tons/ha)")` — title and axis labels, same styling convention as chart 1.
- `# Simple legend` — a comment flagging that the following lines exist solely to build a legend.
- `from matplotlib.patches import Patch` — an import placed inside the function body rather than at the top of the file, because `Patch` is only needed here, for this one chart's legend. This is a deliberate localized import for a rarely-used dependency, distinct from the rest of the file's convention of importing everything at the top — worth noticing if you're used to top-of-file-only imports.
- `legend_elements = [Patch(facecolor="steelblue", label="Irrigated"), Patch(facecolor="coral", label="Rainfed")]` — `ax.scatter` with a raw color array (as used above) does not automatically know how to build a legend, since there's no single labeled "series" per color — only a per-point color list. `Patch` objects are proxy artists: swatches with a `facecolor` and a `label` that exist purely to populate a legend, without being drawn on the chart itself.
- `ax.legend(handles=legend_elements)` — attaches a legend built from those two manual proxy patches, mapping "steelblue" → "Irrigated" and "coral" → "Rainfed" for the reader.
- `plt.tight_layout()` — same layout-fitting call as chart 1.
- `path2 = os.path.join(output_dir, "rainfall_vs_yield.png")` — output path for this chart.
- `fig.savefig(path2, dpi=120)` / `plt.close(fig)` / `print(f"[create_visualizations] Saved: {path2}")` — save, close, and report, identical pattern to chart 1.

**Chart 3 — yield by irrigation method:**
- `fig, ax = plt.subplots(figsize=(8, 5))` — third fresh Figure/Axes pair.
- `irr_avg = df.groupby("irrigation_method")["yield_per_hectare"].mean().sort_values(ascending=False)` — the same groupby-mean-then-sort pattern as chart 1, but grouped by `irrigation_method` (Sprinkler, Drip, Flood, Rainfed) instead of `crop_type`.
- `irr_avg.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white")` — bar chart using `mediumseagreen` instead of chart 1's `steelblue`, a deliberate color change purely so the three saved PNGs are visually distinguishable from one another at a glance.
- `ax.set_title("Average Yield per Hectare by Irrigation Method", fontsize=14, fontweight="bold")` / `ax.set_xlabel("Irrigation Method")` / `ax.set_ylabel("Yield per Hectare (tons/ha)")` — title and axis labels, same convention as the other two charts.
- `ax.tick_params(axis="x", rotation=20)` — a smaller rotation (20 degrees vs. chart 1's 30) since irrigation method names ("Sprinkler", "Drip", "Flood", "Rainfed") are shorter than crop type names and need less rotation to avoid overlapping.
- `plt.tight_layout()` — same layout call.
- `path3 = os.path.join(output_dir, "yield_by_irrigation.png")` — output path for the third chart.
- `fig.savefig(path3, dpi=120)` / `plt.close(fig)` / `print(f"[create_visualizations] Saved: {path3}")` — save, close, report — the function has no `return` statement at the end, consistent with its `-> None` type hint; all of its value comes from files written to `output_dir`.

### Full code block

```python
# ---------------------------------------------------------------------------
# 5. create_visualizations
# ---------------------------------------------------------------------------

def create_visualizations(df: pd.DataFrame, output_dir: str = "output") -> None:
    """Generate and save three analysis charts as PNG files.

    Charts produced:
    1. yield_by_crop.png   — Bar chart: average yield per hectare by crop type.
    2. rainfall_vs_yield.png — Scatter: rainfall_mm vs yield_per_hectare.
    3. yield_by_irrigation.png — Bar chart: average yield per hectare by irrigation method.

    Args:
        df: Feature-enriched DataFrame from add_features().
        output_dir: Directory where PNG files will be saved.
    """
    # Ensure output directory exists — this prevents FileNotFoundError on savefig
    os.makedirs(output_dir, exist_ok=True)

    # --- Chart 1: Average yield per hectare by crop type ---
    fig, ax = plt.subplots(figsize=(8, 5))
    crop_avg = df.groupby("crop_type")["yield_per_hectare"].mean().sort_values(ascending=False)
    crop_avg.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Average Yield per Hectare by Crop Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Crop Type")
    ax.set_ylabel("Yield per Hectare (tons/ha)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "yield_by_crop.png")
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"[create_visualizations] Saved: {path1}")

    # --- Chart 2: Rainfall vs yield per hectare ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = df["is_irrigated"].map({True: "steelblue", False: "coral"})
    ax.scatter(df["rainfall_mm"], df["yield_per_hectare"], c=colors, alpha=0.6, edgecolors="none")
    ax.set_title("Rainfall vs Yield per Hectare", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Yield per Hectare (tons/ha)")
    # Simple legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="steelblue", label="Irrigated"),
                       Patch(facecolor="coral", label="Rainfed")]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "rainfall_vs_yield.png")
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"[create_visualizations] Saved: {path2}")

    # --- Chart 3: Average yield per hectare by irrigation method ---
    fig, ax = plt.subplots(figsize=(8, 5))
    irr_avg = df.groupby("irrigation_method")["yield_per_hectare"].mean().sort_values(ascending=False)
    irr_avg.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white")
    ax.set_title("Average Yield per Hectare by Irrigation Method", fontsize=14, fontweight="bold")
    ax.set_xlabel("Irrigation Method")
    ax.set_ylabel("Yield per Hectare (tons/ha)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    path3 = os.path.join(output_dir, "yield_by_irrigation.png")
    fig.savefig(path3, dpi=120)
    plt.close(fig)
    print(f"[create_visualizations] Saved: {path3}")
```

---

## 6. `main` and the entry-point guard

### Why this section exists
Something has to call the five functions above in the right order and hand data from one to the next — that's `main()`. The `if __name__ == "__main__":` guard beneath it is what lets this same file be safely `import`ed (e.g. by `pytest` to test individual functions) without automatically re-running the entire pipeline as a side effect of the import.

### Line-by-line

- `# --- 6. main ---` (comment banner) — final section divider.
- `def main():` — no parameters; it orchestrates the whole pipeline end-to-end using every function's own default arguments (`load_data()`'s default `filepath`, `create_visualizations()`'s default `output_dir`).
- The docstring — a single line, `"""Run the full pipeline end-to-end."""`, since the function's job is fully described by the sequence of calls inside it.
- `print("=" * 50)` — string multiplication: repeats the `"="` character 50 times to build a divider line, a common lightweight way to draw a banner without importing a formatting library.
- `print(" Agricultural Crop Yield Pipeline — Demo")` — a leading space before the title lightly indents it under the `=` border.
- `print("=" * 50)` — closes the banner with a matching bottom border.
- `df_raw = load_data()` — calls with no arguments, using `load_data`'s default `filepath="demo_data/crop_yields.csv"`. Produces the raw, unmodified DataFrame.
- `df_clean = clean_data(df_raw)` — feeds the raw DataFrame in, gets back the cleaned one (fixed seasons, filled yields, dropped bad-area rows).
- `df_features = add_features(df_clean)` — feeds the cleaned DataFrame in, gets back one with `yield_per_hectare` and `is_irrigated` added.
- `summary = generate_summary(df_features)` — computes and prints the three summary statistics, and captures the returned dict in `summary`.
- `create_visualizations(df_features)` — called for its side effect only (saving three PNGs to `output/`); its `None` return value isn't captured because there's nothing useful to do with it.
- `print("Pipeline complete. Check the output/ directory for charts.")` — a final, plain-language confirmation that the whole run finished, telling the user where to look for the visual output.
- `return summary` — returns the summary dict from `main()` itself. This matters for testability: a test (or an interactive session) can call `main()` once and assert against the returned dict's values, rather than having to re-run the whole pipeline or parse stdout to check the numbers.
- `# Guard: prevents main() from running when the module is imported during tests` — a comment directly explaining the purpose of the line below, for a reader who might not already know the `__name__ == "__main__"` idiom.
- `if __name__ == "__main__":` — every Python module has a built-in `__name__` variable. When the file is run directly (`python pipeline.py`), Python sets `__name__` to the string `"__main__"`. When the same file is instead imported by another module (e.g. `from pipeline import clean_data` inside `tests/test_pipeline.py`), Python sets `__name__` to the module's own name (`"pipeline"`) instead. The `if` condition is therefore only `True` when the script is executed directly.
- `main()` — the actual call, indented inside the guard. Without the guard around it, importing `pipeline` for its individual functions (exactly what the test suite needs to do) would immediately execute the entire pipeline as a side effect of the `import` statement — printing output, reading the CSV, and writing three PNG files every time `pytest` collects the test module, which is both slow and liable to break CI if the CSV or output directory isn't set up the way `main()` expects.

### Full code block

```python
# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------

def main():
    """Run the full pipeline end-to-end."""
    print("=" * 50)
    print(" Agricultural Crop Yield Pipeline — Demo")
    print("=" * 50)

    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_features = add_features(df_clean)
    summary = generate_summary(df_features)
    create_visualizations(df_features)

    print("Pipeline complete. Check the output/ directory for charts.")
    return summary


# Guard: prevents main() from running when the module is imported during tests
if __name__ == "__main__":
    main()
```
