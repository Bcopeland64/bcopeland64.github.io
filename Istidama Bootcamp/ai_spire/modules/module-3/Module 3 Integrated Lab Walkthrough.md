# Module 3 Integrated Lab — Code Walkthrough

This document walks every line of [etl_pipeline.ipynb](files/etl_pipeline.ipynb). It builds a complete Extract → Transform → Validate → Load pipeline against a public-library dataset — the SI demo version of the same four-function pattern you implement against the Amman Digital Market data for the graded assignment. Read alongside the notebook.

---

## 0. Environment Setup

### Why this section exists
Before any database call can succeed, the kernel needs the right packages installed and a working, credentialed connection to PostgreSQL. Splitting install from connect into two cells means a package failure and a connection failure show up as two distinct, easy-to-diagnose errors instead of one tangled traceback.

### 0a. Install dependencies

#### Why this section exists
Guarantees the notebook's kernel — not just some other Python on the machine — has the libraries the rest of the notebook assumes exist.

#### Line-by-line
- `%pip install -q pandas sqlalchemy psycopg2-binary python-dotenv` — the `%pip` **magic command** (not the shell `!pip`) is Jupyter's IPython-native way of installing packages. It targets the exact Python environment backing the running kernel, which avoids the classic "I installed it but the notebook still can't import it" bug that happens when `!pip` resolves to a different Python on `PATH` than the kernel uses. `-q` suppresses the verbose per-package download log. The four packages: `pandas` (DataFrames — the pipeline's core data structure), `sqlalchemy` (a database-agnostic engine/connection layer so the same code works against Postgres, MySQL, SQLite, etc. with only the connection string changing), `psycopg2-binary` (the actual PostgreSQL driver that SQLAlchemy's `postgresql+psycopg2` dialect calls under the hood — the `-binary` variant ships a precompiled C extension so no local compiler/toolchain is needed), and `python-dotenv` (loads key=value pairs from a `.env` file into environment variables, keeping credentials out of the notebook source).
- The printed note *"you may need to restart the kernel to use updated packages"* is pip's standard warning after any install — Python caches imported modules, so a package installed mid-session sometimes isn't visible until the kernel restarts. It didn't matter here because none of these packages were previously imported in this kernel.

#### Full code block
```python
%pip install -q pandas sqlalchemy psycopg2-binary python-dotenv
```

### 0b. Load credentials and connect

#### Why this section exists
Centralizes every piece of environment-specific configuration (host, port, credentials) in one place at the top of the notebook, and proves the connection works before any downstream cell relies on it.

#### Line-by-line
- `import os` — used to read environment variables via `os.getenv(...)`.
- `import pandas as pd` — the DataFrame library; imported here because it's used starting in the very next section, not just inside `load()`.
- `from dotenv import load_dotenv` — the loader function from `python-dotenv`.
- `from sqlalchemy import create_engine, text` — `create_engine` builds a connection-pool object from a URL; `text()` wraps a raw SQL string so SQLAlchemy treats it as literal, executable SQL rather than rejecting it (SQLAlchemy 1.4+/2.0 require raw strings to be explicitly wrapped this way — it's a guardrail against accidentally executing unvalidated string-built SQL).
- `from sqlalchemy.engine import URL` — a structured URL-builder class, used two lines later instead of hand-formatting a connection string.
- `load_dotenv()` — reads a `.env` file in the current working directory (if present) and injects its key=value pairs into `os.environ`. If no `.env` file exists, this is a harmless no-op and the `os.getenv(...)` calls below fall back to their defaults.
- `DB_USER = os.getenv("DB_USER", "postgres")` — reads the `DB_USER` environment variable, defaulting to `"postgres"` if it isn't set. The same pattern repeats for the next four variables.
- `DB_PASSWORD = os.getenv("DB_PASSWORD", "")` — defaults to an empty string (many local Postgres setups have no password on the default user).
- `DB_HOST = os.getenv("DB_HOST", "localhost")` — defaults to the local machine.
- `DB_PORT = int(os.getenv("DB_PORT", "5433"))` — environment variables are always strings, so this wraps the result in `int(...)` to get a real port number; without the cast, `URL.create(port=...)` would receive `"5433"` (a string) instead of `5433` (an int) and could behave unpredictably depending on the driver. Note the default is `5433`, not Postgres's standard `5432` — this SI demo instance runs on a non-default port, likely to avoid colliding with a learner's own local Postgres on `5432`.
- `DB_NAME = os.getenv("DB_NAME", "library_demo")` — the database this notebook talks to; the graded assignment instead uses `amman_market`.
- `DB_URL = URL.create(drivername="postgresql+psycopg2", username=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, database=DB_NAME)` — builds a structured `URL` object rather than an f-string connection string. This matters because passwords or usernames containing special characters (`@`, `:`, `/`) would silently corrupt a hand-built string; `URL.create` percent-encodes each component correctly. `drivername="postgresql+psycopg2"` tells SQLAlchemy which SQL dialect (`postgresql`) and which underlying driver (`psycopg2`) to use — the two halves before/after the `+` are independently swappable (e.g. `postgresql+asyncpg` for async code).
- `engine = create_engine(DB_URL)` — creates a SQLAlchemy `Engine`, which manages a pool of connections. Creating an engine does **not** immediately open a connection — it's lazy — so this line can't fail due to bad credentials; the next block is what actually proves connectivity.
- `with engine.connect() as conn:` — opens an actual connection to Postgres for the block, and the `with` context manager guarantees it's returned to the pool (or closed) automatically even if an exception is raised inside.
- `result = conn.execute(text("SELECT current_database()"))` — runs a trivial query that just asks Postgres to report its own currently-connected database name. This is a common "is this thing actually alive and pointed at the right place" smoke test.
- `print("✅ Connected to:", result.scalar())` — `.scalar()` pulls the single value out of the first column of the first row of the result set (equivalent to `.fetchone()[0]`, but safer/clearer for a known single-value query). If this prints `library_demo`, the connection and credentials are confirmed correct before any other cell runs.

#### Full code block
```python
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ── Load credentials from .env ─────────────────────────────────────────────
load_dotenv()  # reads .env in the current directory

DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5433"))
DB_NAME     = os.getenv("DB_NAME", "library_demo")

DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)
engine = create_engine(DB_URL)

# Quick connectivity test
with engine.connect() as conn:
    result = conn.execute(text("SELECT current_database()"))
    print("✅ Connected to:", result.scalar())
```

---

## 1. Explore the Source Data

### Why this section exists
Real ETL work always starts with looking at the raw data before writing a single line of transform logic. This section previews all three tables and quantifies the two intentional data-quality problems the `transform()`/`validate()` functions will need to handle later — so their behavior in Section 3 doesn't come as a surprise.

### 1a. Preview all three tables

#### Line-by-line
- `with engine.connect() as conn:` — opens a fresh connection scoped to just this cell (a new one from the previous section's, though it draws from the same underlying pool).
- `for table in ["members", "books", "loans"]:` — the three source tables that make up this library dataset, hardcoded as a literal list since the notebook works with exactly these three and no others.
- `df = pd.read_sql(text(f"SELECT * FROM {table} LIMIT 5"), conn)` — `pd.read_sql` executes the SQL and returns the result directly as a DataFrame (no separate cursor-to-DataFrame conversion step needed). `LIMIT 5` keeps the preview small — this cell is diagnostic, not the real extract. Note this is an f-string interpolating `table` directly into SQL; that's safe here only because `table` comes from a hardcoded list, never from user input — doing this with untrusted input would be a SQL-injection risk.
- `print(f"── {table} (preview) ──")` — a visual separator so the three previews are easy to tell apart when scrolling the output.
- `display(df)` — Jupyter's rich-output function (a global provided by the IPython kernel, not something imported from pandas). Unlike `print(df)`, which renders a plain-text table, `display(df)` renders pandas' styled HTML table — this is why the notebook uses `display()` here but `print()` a few lines later for status text.

#### Full code block
```python
# Preview all three tables
with engine.connect() as conn:
    for table in ["members", "books", "loans"]:
        df = pd.read_sql(text(f"SELECT * FROM {table} LIMIT 5"), conn)
        print(f"── {table} (preview) ──")
        display(df)
```

### 1b. Check for the intentional data-quality issues

#### Line-by-line
- `null_cities = pd.read_sql(text("SELECT COUNT(*) AS null_cities FROM members WHERE city IS NULL"), conn)` — counts NULL `city` values by pushing the aggregation down into Postgres itself (`COUNT(*) ... WHERE city IS NULL`) rather than pulling the whole `members` table into pandas and counting there. This is the more efficient pattern once a table is too large to comfortably load in full, and it's good habit-forming even on a 100-row table.
- `print("NULL city rows:", null_cities["null_cities"].iloc[0])` — the query returns a one-row, one-column DataFrame; `["null_cities"]` selects that column as a Series, and `.iloc[0]` grabs its single value by position. Output: 30 — matching the ~30 NULL cities the `INSTRUCTOR_GUIDE.md` describes as an intentional seed-data quirk that must be passed through untouched, not dropped or imputed.
- `bad_overdue = pd.read_sql(text("SELECT COUNT(*) AS bad_rows FROM loans WHERE days_overdue < 0"), conn)` — counts loans with a **negative** `days_overdue`, which is nonsensical (you can't be overdue by a negative number of days) and represents the seeded data-entry error this domain uses in place of the assignment's `quantity > 100` error. Output: 10 rows.
- `overdue = pd.read_sql(text("SELECT COUNT(*) AS overdue FROM loans WHERE days_overdue > 0"), conn)` — counts genuinely overdue loans (`days_overdue > 0`). This is **not** an error — it's a legitimate business signal the pipeline will later aggregate into `overdue_loans` per member. Output: 357.
- This cell fixes nothing; it's purely diagnostic so the numbers seen here (30 NULL cities, 10 bad rows, 357 genuinely overdue) can be cross-checked against the transform/validate logic's behavior later in the notebook.

#### Full code block
```python
# Check for the intentional data-quality issues
with engine.connect() as conn:
    # NULL cities
    null_cities = pd.read_sql(text("SELECT COUNT(*) AS null_cities FROM members WHERE city IS NULL"), conn)
    print("NULL city rows:", null_cities["null_cities"].iloc[0])

    # Negative days_overdue (data-entry error)
    bad_overdue = pd.read_sql(text("SELECT COUNT(*) AS bad_rows FROM loans WHERE days_overdue < 0"), conn)
    print("Rows with days_overdue < 0:", bad_overdue["bad_rows"].iloc[0])

    # Overdue loans (returned late)
    overdue = pd.read_sql(text("SELECT COUNT(*) AS overdue FROM loans WHERE days_overdue > 0"), conn)
    print("Overdue loans (days_overdue > 0):", overdue["overdue"].iloc[0])
```

---

## 2. EXTRACT

### Why this section exists
`extract()` is deliberately "dumb" — it pulls the three source tables into memory with **no filtering, no joins, no business logic at all**. Keeping extraction pure and separate from transformation is what makes each stage independently testable (this is exactly what `INSTRUCTOR_GUIDE.md`'s grading rubric checks for: *"no filtering in extract"*).

> Note: the notebook's own markdown header for this section (`## 2 · EXTRACT`) has a small formatting glitch in the source file — an inline-code span that should read `` `extract` `` renders as a blank space (*"The `` function pulls all three tables…"*). This is a pre-existing quirk in the notebook's markdown source, not something introduced here; the code itself is unaffected.

### Line-by-line
- `def extract(engine):` — takes a SQLAlchemy `Engine` (not an open connection) as its only argument, so the function can open and close its own connection scope internally. That makes it safe to call `extract()` more than once without leaking connections.
- The docstring documents the parameter and return shape: `dict[str, pd.DataFrame]` keyed by `'members'`, `'books'`, `'loans'`.
- `tables = ["members", "books", "loans"]` — the same three-table list used in Section 1's preview; `extract()`'s whole job is to pull exactly these and nothing else.
- `data = {}` — the dictionary that will be built up and returned.
- `with engine.connect() as conn:` — one connection is opened and reused for all three table reads inside the loop below, rather than opening a fresh connection per table.
- `for table in tables: data[table] = pd.read_sql(text(f"SELECT * FROM {table}"), conn)` — an **unfiltered** `SELECT *` with no `WHERE` and no `LIMIT`. This is the crucial difference from Section 1's preview query — this is the real, complete extract of every row in every table, exactly as it exists in the database, warts (NULL cities, negative `days_overdue`) included.
- `print(f"  {table}: {len(data[table]):,} rows")` — the `:,` format spec inserts thousands separators (e.g. `1,200` instead of `1200`). Printing a per-table row count immediately surfaces an obviously wrong extract (e.g. `0 rows` from a typo'd table name) before any downstream code runs.
- `return data` — hands back the dict of three raw DataFrames.
- `print("Extracting...")` / `raw_data = extract(engine)` / `print("Sample — loans:")` / `raw_data["loans"].head()` — the driver code: calls the function, then previews the first 5 rows (`.head()`'s default `n=5`) of the `loans` table specifically, since that's the table with the most rows and the most columns worth eyeballing before transform.
- Output confirms 100 members, 30 books, 455 loans extracted — matching the row counts referenced in `INSTRUCTOR_GUIDE.md`'s schema description.

### Full code block
```python
def extract(engine):
    """
    Pull all three source tables from the database into a dictionary of DataFrames.

    Parameters
    ----------
    engine : sqlalchemy Engine

    Returns
    -------
    dict[str, pd.DataFrame]  keys: 'members', 'books', 'loans'
    """
    tables = ["members", "books", "loans"]
    data = {}
    with engine.connect() as conn:
        for table in tables:
            data[table] = pd.read_sql(text(f"SELECT * FROM {table}"), conn)
            print(f"  {table}: {len(data[table]):,} rows")
    return data


# ── Run it ───────────────────────────────────────────────────────────────────
print("Extracting...")
raw_data = extract(engine)
print("Sample — loans:")
raw_data["loans"].head()
```

---

## 3. TRANSFORM

### Why this section exists
This is the pipeline's core business logic: filter out the seeded bad rows, join the three tables into one, compute derived columns, and aggregate down to a one-row-per-member summary. It's the direct analogue of the graded assignment's `transform()` and is worth the most points in `INSTRUCTOR_GUIDE.md`'s rubric (30/100) precisely because it's where most of the reasoning happens.

> Note: like Section 2, this section's markdown source (`## 3 · TRANSFORM`) has several inline-code spans that render blank in the notebook itself (e.g. *"1. Exclude loans where (data-entry errors)"* is missing the `` `days_overdue < 0` `` that belongs there, and the "Join diagram" is empty). The intent, filled in from the code and from `etl_pipeline_second_build.ipynb`'s intact version of the same markdown, is: join `loans` → `books` on `book_id`, then → `members` on `member_id`; exclude loans where `days_overdue < 0`; compute `loan_duration_days = return_date - loan_date`; aggregate to `total_loans`, `avg_loan_duration_days`, `overdue_loans`, `favourite_genre` per member.

### 3a. `transform()`

#### Line-by-line
- `def transform(data_dict):` — takes exactly the dict shape `extract()` returns, keeping a hard functional boundary: transform's input type is extract's output type, and nothing else.
- `members = data_dict["members"]`, `books = data_dict["books"]`, `loans = data_dict["loans"]` — unpacked into local names purely for readability in the code that follows; avoids repeated `data_dict["..."]` indexing.
- **Step 1 — filter bad loans:** `valid_loans = loans[loans["days_overdue"] >= 0].copy()` — a boolean-mask filter: `loans["days_overdue"] >= 0` produces a Series of `True`/`False`, and indexing `loans[...]` with it keeps only the `True` rows. This is the line that removes the 10 negative-`days_overdue` data-entry errors counted in Section 1b. The trailing `.copy()` is not optional decoration — without it, `valid_loans` would be a pandas *view* into `loans`, and the next block adds a new column (`loan_duration_days`) onto a filtered slice, which would raise a `SettingWithCopyWarning` (or in some pandas versions silently write to the wrong object). `.copy()` makes `valid_loans` fully independent.
- `print(f"  Valid loans after removing negative days_overdue: {len(valid_loans):,}")` — prints 445 (455 total − 10 bad), directly confirming the count from Section 1b's diagnostic query.
- **Step 2 — join loans → books → members:** `df = valid_loans.merge(books[["book_id", "genre"]], on="book_id", how="left").merge(members[["member_id", "name", "city"]], on="member_id", how="left")` — two chained `.merge()` calls. `books[["book_id", "genre"]]` and `members[["member_id", "name", "city"]]` subset each table to only the columns actually needed *before* merging, which avoids pulling in irrelevant columns (`title`, `year_published`, `join_date`) and prevents unrelated column-name collisions. `how="left"` on both merges means every row of `valid_loans` is kept even if (hypothetically) a matching book or member were missing — a defensive choice, since foreign-key integrity should guarantee a match every time in this seeded dataset.
- `print(f"  Rows after join: {len(df):,}")` — should equal `len(valid_loans)` exactly, since both merges are on unique keys (`book_id`, `member_id`) — a 1-to-1/many-to-1 join can't fan out row counts. If this number were larger than `valid_loans`, that would be a red flag for an accidental many-to-many join.
- **Step 3 — compute loan duration:** `df["loan_duration_days"] = (pd.to_datetime(df["return_date"]) - pd.to_datetime(df["loan_date"])).dt.days` — `pd.to_datetime(...)` converts the string date columns into pandas `Timestamp` objects. Subtracting two datetime Series produces a Series of `Timedelta` objects (durations). `.dt.days` is the datetime **accessor** — it exposes datetime-like properties (`.dt.days`, `.dt.year`, etc.) on a Series and here extracts the whole-number day count from each `Timedelta`. This is the library-domain equivalent of the assignment's `line_total = quantity × unit_price` derived column.
- **Step 4 — aggregate per member:**
  ```python
  loan_counts = (
      df.groupby("member_id")
        .agg(
            total_loans            =("loan_id",            "count"),
            avg_loan_duration_days =("loan_duration_days", "mean"),
            overdue_loans          =("days_overdue",        lambda x: (x > 0).sum()),
        )
        .reset_index()
  )
  ```
  - `df.groupby("member_id")` — groups all rows by `member_id`, so every following `.agg(...)` call operates per-member.
  - `.agg(name=(column, func), ...)` — **named aggregation** syntax: each keyword argument becomes an output column name, paired with a `(source_column, aggregation_function)` tuple. This is the modern pandas idiom specifically because it avoids the ugly `MultiIndex` columns you'd get from the older `.agg({"col": "func"})` dict form, which then need a manual flattening step.
  - `total_loans=("loan_id", "count")` — counts non-null `loan_id` values per group. Because `loan_id` is the grain of one row per loan in this table (unlike the assignment's `order_items`, where `count` vs `nunique` matters because an order can span multiple item rows), plain `"count"` is correct here — there's no risk of inflating the total.
  - `avg_loan_duration_days=("loan_duration_days", "mean")` — the average of that member's per-loan durations computed in Step 3.
  - `overdue_loans=("days_overdue", lambda x: (x > 0).sum())` — a custom aggregation. Within the lambda, `x` is the Series of `days_overdue` values for one member's group. `(x > 0)` produces a boolean Series; `.sum()` on booleans counts the `True` values (Python treats `True` as `1`, `False` as `0`). This "boolean mask, then `.sum()`" pattern is an extremely common pandas idiom for "count rows matching a condition" — it shows up constantly outside this lab too.
  - `.reset_index()` — `groupby("member_id")` leaves `member_id` as the DataFrame's *index* rather than a normal column; `.reset_index()` moves it back into a regular column so it can be used as a merge key in Step 6.
  - `loan_counts["avg_loan_duration_days"] = loan_counts["avg_loan_duration_days"].round(1)` — rounds the average to 1 decimal place for a cleaner display — the same "round your derived numeric columns" discipline `INSTRUCTOR_GUIDE.md` calls out for `avg_order_value` in the graded assignment (Mistake 4), just applied to a duration instead of a currency value.
- **Step 5 — favourite genre per member:**
  ```python
  top_genre = (
      df.groupby(["member_id", "genre"])["loan_id"]
        .count().reset_index()
        .sort_values("loan_id", ascending=False)
        .drop_duplicates("member_id")[["member_id", "genre"]]
        .rename(columns={"genre": "favourite_genre"})
  )
  ```
  - `df.groupby(["member_id", "genre"])["loan_id"].count().reset_index()` — a **two-key** groupby, counting how many loans each member took out per genre. The result has one row per (member, genre) combination that actually occurred, with a `loan_id` count column.
  - `.sort_values("loan_id", ascending=False)` — sorts the entire exploded (member, genre) table so the highest-count rows come first, globally (not per group).
  - `.drop_duplicates("member_id")` — because the frame is already sorted descending by count, keeping only the *first* occurrence of each `member_id` (`drop_duplicates`'s default behavior) means keeping that member's **highest-count** genre — a common and compact "top-1-per-group" trick that avoids `.idxmax()` or a second groupby.
  - `[["member_id", "genre"]]` — after deduplication, keeps only the two columns actually needed going forward.
  - `.rename(columns={"genre": "favourite_genre"})` — renames the column so the final merged table has a clearly-named `favourite_genre` field rather than an ambiguous `genre`.
- **Step 6 — join back to member info:**
  ```python
  result = (
      members[["member_id", "name", "city"]]
      .merge(loan_counts, on="member_id", how="inner")
      .merge(top_genre,   on="member_id", how="left")
  )
  ```
  - `members[["member_id", "name", "city"]]` — starts from the **full** member roster (all 100 members, including any with zero valid loans), then subsets to just the display columns needed in the final output.
  - `.merge(loan_counts, on="member_id", how="inner")` — an **inner** join: any member who appears in the full roster but has *zero* rows in `loan_counts` (because every one of their loans was filtered out in Step 1, or they never borrowed anything) is dropped here. This is a deliberate choice, directly mirroring the pitfall `INSTRUCTOR_GUIDE.md` calls out for the graded assignment (Mistake 3): a `left` join here would incorrectly keep zero-activity members in a table that's supposed to summarize *loan activity* — there'd be nothing meaningful to summarize for them.
  - `.merge(top_genre, on="member_id", how="left")` — a **left** join here instead, because every member surviving the previous inner join is guaranteed to have at least one valid loan, and therefore a row in `top_genre`. `how="left"` here functions as a safety net rather than an actual filter.
- `print(f"  Final summary rows: {len(result):,}")` — prints 100 in the sample run, meaning every one of the 100 seeded members had at least one valid (non-negative-`days_overdue`) loan.
- `return result`
- Driver code: `print("Transforming...")`, `summary_df = transform(raw_data)`, `print()`, `summary_df.head(10)` — previews 10 rows instead of the default 5, since this is the richer, final output worth a slightly larger sample.

#### Full code block
```python
def transform(data_dict):
    """
    Join tables, apply quality filters, compute loan duration, and aggregate
    to a member-level summary DataFrame.
    """
    members = data_dict["members"]
    books   = data_dict["books"]
    loans   = data_dict["loans"]

    # ── Step 1: filter bad loans (negative days_overdue = data-entry error) ──
    valid_loans = loans[loans["days_overdue"] >= 0].copy()
    print(f"  Valid loans after removing negative days_overdue: {len(valid_loans):,}")

    # ── Step 2: join loans → books → members ─────────────────────────────────
    df = (
        valid_loans
        .merge(books[["book_id", "genre"]],            on="book_id",   how="left")
        .merge(members[["member_id", "name", "city"]], on="member_id", how="left")
    )
    print(f"  Rows after join: {len(df):,}")

    # ── Step 3: compute loan duration ────────────────────────────────────────
    df["loan_duration_days"] = (
        pd.to_datetime(df["return_date"]) - pd.to_datetime(df["loan_date"])
    ).dt.days

    # ── Step 4: aggregate per member ─────────────────────────────────────────
    loan_counts = (
        df.groupby("member_id")
          .agg(
              total_loans            =("loan_id",            "count"),
              avg_loan_duration_days =("loan_duration_days", "mean"),
              overdue_loans          =("days_overdue",        lambda x: (x > 0).sum()),
          )
          .reset_index()
    )
    loan_counts["avg_loan_duration_days"] = loan_counts["avg_loan_duration_days"].round(1)

    # ── Step 5: favourite genre per member ───────────────────────────────────
    top_genre = (
        df.groupby(["member_id", "genre"])["loan_id"]
          .count().reset_index()
          .sort_values("loan_id", ascending=False)
          .drop_duplicates("member_id")[["member_id", "genre"]]
          .rename(columns={"genre": "favourite_genre"})
    )

    # ── Step 6: join back to member info ─────────────────────────────────────
    result = (
        members[["member_id", "name", "city"]]
        .merge(loan_counts, on="member_id", how="inner")
        .merge(top_genre,   on="member_id", how="left")
    )

    print(f"  Final summary rows: {len(result):,}")
    return result


# ── Run it ───────────────────────────────────────────────────────────────────
print("Transforming...")
summary_df = transform(raw_data)
print()
summary_df.head(10)
```

### 3b. Distribution check

#### Why this section exists
A quick, human-eyeballed sanity check that the aggregated columns land in plausible ranges (no negative loan counts, no absurd durations) before handing the DataFrame to `validate()`.

#### Line-by-line
- `summary_df[["total_loans", "avg_loan_duration_days", "overdue_loans"]]` — selects three columns via a list of column names, which returns a DataFrame (not a Series, since a list — even a one-element one — always yields a DataFrame slice).
- `.describe()` — pandas' built-in summary-statistics method: for each numeric column it computes `count`, `mean`, `std`, `min`, the 25th/50th/75th percentiles, and `max`.
- `.round(2)` — rounds every statistic in the resulting table to 2 decimal places for a cleaner printed view.
- `print(...)` — renders the `describe()` table as plain text in the cell output.

#### Full code block
```python
# Quick distribution check
print(summary_df[["total_loans", "avg_loan_duration_days", "overdue_loans"]].describe().round(2))
```

---

## 4. VALIDATE

### Why this section exists
`validate()` is the pipeline's quality gate, deliberately kept as its own function separate from `transform()` so the business logic and the quality checks can be tested independently. Its two hard requirements — raise `ValueError` (never `AssertionError`), and collect *every* problem before raising rather than stopping at the first one — are exactly the two "common mistakes" `INSTRUCTOR_GUIDE.md` flags for the graded assignment's `validate()` (Mistakes 5 and 6).

### 4a. `validate()`

#### Line-by-line
- `def validate(df):` — takes the transformed summary DataFrame; returns nothing on success (prints instead) and raises on failure.
- `errors = []` — an accumulator list. This is the mechanism that makes "collect all errors, then raise once" possible: each check below only *appends* to the list, none of them raise immediately.
- `dupes = df["member_id"].duplicated().sum()` — `.duplicated()` returns a boolean Series marking every row *after* the first occurrence of a repeated value as `True` (the first occurrence itself is `False`). `.sum()` counts how many `True`s there are — i.e., how many duplicate `member_id` rows exist. Since this table is supposed to be one row per member, `member_id` is effectively its primary key and must be unique.
- `if dupes > 0: errors.append(f"Duplicate member_ids: {dupes}")` — appended, not raised, so later checks still get a chance to run in this same call.
- `null_ids = df["member_id"].isna().sum()` — `.isna()` flags NULL/NaN values; summing counts them. A NULL primary key would silently break any downstream join, filter, or lookup keyed on `member_id`.
- `bad_loans = (df["total_loans"] < 1).sum()` — every surviving row should have `total_loans >= 1` by construction (Step 6's inner join in `transform()` already excludes zero-loan members) — so this check is mostly a redundant safety net, catching cases where the data was corrupted or manually edited after transform (exactly what Section 4b does on purpose).
- `bad_duration = (df["avg_loan_duration_days"] <= 0).sum()` — a non-positive average duration would indicate a date-parsing bug or a `return_date` earlier than `loan_date` somewhere upstream.
- `if errors: raise ValueError("Validation failed:\n  • " + "\n  • ".join(errors))` — the critical design choice: **raises `ValueError`, not `AssertionError`.** This matters for two reasons: calling code and test suites can catch `ValueError` deterministically and expect it as documented API behavior, whereas `assert` statements are conventionally for internal debugging invariants and — more importantly — are silently *stripped out entirely* when Python runs with the `-O` (optimize) flag, which would make an `assert`-based validator a no-op in that mode. `"\n  • ".join(errors)` stitches every collected error message into one multi-line, bulleted string, so a single `raise` surfaces *all* problems at once rather than forcing a "fix one, rerun, hit the next one" cycle.
- `print(f"✅  Validation passed — {len(df):,} rows are clean.")` — only reached if `errors` stayed empty; the function returns `None` implicitly.
- `validate(summary_df)` — the driver call on the real, clean data; in the sample run this prints the success message for all 100 rows.

#### Full code block
```python
def validate(df):
    """
    Run data-quality checks. Raises ValueError if any check fails.
    """
    errors = []

    dupes = df["member_id"].duplicated().sum()
    if dupes > 0:
        errors.append(f"Duplicate member_ids: {dupes}")

    null_ids = df["member_id"].isna().sum()
    if null_ids > 0:
        errors.append(f"NULL member_ids: {null_ids}")

    bad_loans = (df["total_loans"] < 1).sum()
    if bad_loans > 0:
        errors.append(f"Rows with total_loans < 1: {bad_loans}")

    bad_duration = (df["avg_loan_duration_days"] <= 0).sum()
    if bad_duration > 0:
        errors.append(f"Non-positive avg_loan_duration: {bad_duration}")

    if errors:
        raise ValueError("Validation failed:\n  • " + "\n  • ".join(errors))

    print(f"✅  Validation passed — {len(df):,} rows are clean.")


# ── Validate the real data ───────────────────────────────────────────────────
validate(summary_df)
```

### 4b. Demonstrate a validation failure

#### Why this section exists
Proves the `ValueError` path actually fires rather than trusting it blindly, and models the `try`/`except` pattern learners will need when writing their own test suite against `validate()`.

#### Line-by-line
- `bad_df = summary_df.copy()` — an explicit copy so the injected bad value below doesn't mutate `summary_df`, which is still needed, clean, in Section 5's `load()` call.
- `bad_df.loc[0, "total_loans"] = 0` — `.loc[row_label, column_label]` sets a single cell by label. Row label `0` here is the first row's index value (not necessarily `member_id == 0`); this deliberately trips the `bad_loans = (df["total_loans"] < 1).sum()` check in `validate()`.
- `try: validate(bad_df) except ValueError as e: print(...)` — calls `validate()` inside a `try` block and catches specifically `ValueError` — the exact exception type `validate()` is documented to raise. `print(e)` prints the exception's string representation, which is the full bulleted multi-line message built inside `validate()` (`"Validation failed:\n  • Rows with total_loans < 1: 1"`).

#### Full code block
```python
# ── Demonstrate what happens when validation FAILS ──────────────────────────
bad_df = summary_df.copy()
bad_df.loc[0, "total_loans"] = 0   # inject a bad value

try:
    validate(bad_df)
except ValueError as e:
    print("❌ Caught expected error:")
    print(e)
```

---

## 5. LOAD

### Why this section exists
`load()` is the pipeline's final stage: it writes the already-validated DataFrame to two destinations — a PostgreSQL table and a CSV file. It's deliberately called *after* `validate()` succeeds, never before, so nothing bad ever gets persisted. Its two required behaviors — `if_exists="replace"` and creating the output directory if missing — are the exact two pitfalls `INSTRUCTOR_GUIDE.md` calls out for the graded assignment's `load()` (Mistakes 7 and 8).

### Line-by-line
- `import pathlib` — imported right here rather than at the top of the notebook, since `pathlib.Path` is only needed inside this cell's `load()` function. (In a notebook, imports are often placed close to first use for narrative clarity; a production module would put this at the top of the file instead — which is exactly what `etl_pipeline_functions.py` does.)
- `def load(df, engine, csv_path="outputs/member_summary.csv"):` — `csv_path` has a default value, so the common call `load(summary_df, engine)` works without specifying a path, while still allowing an override.
- `csv_path = pathlib.Path(csv_path)` — converts the string (default or passed-in) into a `Path` object, which is what makes the next line's `.parent.mkdir(...)` call available.
- `csv_path.parent.mkdir(parents=True, exist_ok=True)` — `csv_path.parent` is the `outputs/` directory (everything but the filename). `.mkdir(parents=True, exist_ok=True)` creates that directory — and any missing intermediate directories (`parents=True`) — if it doesn't already exist, and does **not** raise an error if it already does (`exist_ok=True`), making this line safe to run repeatedly. Skipping this line is exactly `INSTRUCTOR_GUIDE.md`'s Mistake 8: writing to a nonexistent `outputs/` folder would raise `FileNotFoundError`.
- `df.to_sql("member_summary", con=engine, if_exists="replace", index=False)` — writes the entire DataFrame to a Postgres table named `member_summary`. `if_exists="replace"` drops and recreates the table fresh on every run — this is what makes re-running the notebook idempotent; `INSTRUCTOR_GUIDE.md`'s Mistake 7 flags that using `"append"` instead would double (or triple, etc.) the row count on every subsequent run. `index=False` prevents pandas from writing its own auto-generated integer row index as an extra, unwanted column in the destination table.
- `print(f"  Wrote {len(df):,} rows → PostgreSQL table 'member_summary'")` — visible confirmation.
- `df.to_csv(csv_path, index=False)` — writes the same DataFrame to the CSV path built above, again with `index=False` for the same reason as the SQL write.
- `print(f"  Wrote CSV → {csv_path}")` — visible confirmation.
- `load(summary_df, engine)` — the driver call, using the function's default `csv_path`.

### Full code block
```python
import pathlib

def load(df, engine, csv_path="outputs/member_summary.csv"):
    """
    Write the transformed summary to PostgreSQL and CSV.
    """
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to Postgres
    df.to_sql("member_summary", con=engine, if_exists="replace", index=False)
    print(f"  Wrote {len(df):,} rows → PostgreSQL table 'member_summary'")

    # Write to CSV
    df.to_csv(csv_path, index=False)
    print(f"  Wrote CSV → {csv_path}")


# ── Run it ───────────────────────────────────────────────────────────────────
load(summary_df, engine)
```

---

## 6. Verify the Outputs

### Why this section exists
Confirms `load()` didn't just run without crashing, but actually persisted the *correct* data — reading each output back independently is a stronger check than trusting the print statements from Section 5.

### 6a. Read back from Postgres

#### Line-by-line
- `with engine.connect() as conn: db_result = pd.read_sql(text("SELECT * FROM member_summary ORDER BY total_loans DESC LIMIT 10"), conn)` — queries the table that `load()` just wrote. `ORDER BY total_loans DESC LIMIT 10` surfaces the 10 most active members — a deliberately different, more interesting slice than `transform()`'s earlier `head(10)` preview, which was in default `member_id` order.
- `print("Top 10 members by total loans (from Postgres):")` / `display(db_result)` — labeled output, using `display()` again for the rich HTML table rendering.

#### Full code block
```python
# ── Read back from Postgres ──────────────────────────────────────────────────
with engine.connect() as conn:
    db_result = pd.read_sql(
        text("SELECT * FROM member_summary ORDER BY total_loans DESC LIMIT 10"), conn
    )

print("Top 10 members by total loans (from Postgres):")
display(db_result)
```

### 6b. Read back from CSV

#### Line-by-line
- `csv_result = pd.read_csv("outputs/member_summary.csv")` — reads the CSV file back from disk using a hardcoded path that matches `load()`'s default `csv_path`.
- `print(f"CSV rows: {len(csv_result):,}")` — row-count check; should match the Postgres table's row count and `summary_df`'s row count.
- `print(f"CSV columns: {list(csv_result.columns)}")` — `csv_result.columns` is a pandas `Index` object; wrapping it in `list(...)` converts it to a plain Python list so it prints as a clean, ordinary list rather than an `Index([...], dtype='object')` repr.
- `csv_result.head()` — previews the first 5 rows (default) as a final visual check.

#### Full code block
```python
# ── Read back from CSV ───────────────────────────────────────────────────────
csv_result = pd.read_csv("outputs/member_summary.csv")
print(f"CSV rows: {len(csv_result):,}")
print(f"CSV columns: {list(csv_result.columns)}")
csv_result.head()
```

---

## 7. Run the Full Pipeline End-to-End

### Why this section exists
Every stage up to this point ran manually, one cell at a time, so intermediate state (`raw_data`, `summary_df`, `bad_df`) could be inspected between steps. `run_pipeline()` wires the four functions together into the single call a real scheduled job would actually invoke. Per `INSTRUCTOR_GUIDE.md`'s FAQ, this wrapper is a convenience shown for completeness — it is *not* separately graded in the assignment; students get full credit for correctly implementing and calling the four individual functions on their own.

### Line-by-line
- `def run_pipeline(db_url, csv_path="outputs/member_summary.csv"):` — takes a raw `db_url` (a connection string or `URL` object) rather than an already-built `engine`, so the function is fully self-contained and doesn't depend on any notebook-level state existing beforehand.
- `engine = create_engine(db_url)` — builds a brand-new `Engine` inside the function. This intentionally shadows the module-level `engine` variable created back in Section 0b — `run_pipeline()` gets its own fresh connection pool rather than reusing the notebook's.
- `print("── EXTRACT ───")` / `raw = extract(engine)` — calls Section 2's function.
- `print("── TRANSFORM ──")` / `summary = transform(raw)` — calls Section 3a's function.
- `print("── VALIDATE ───")` / `validate(summary)` — calls Section 4a's function. Its return value (`None`) is discarded because `validate()`'s only two behaviors are "print and return `None`" or "raise `ValueError`" — there's nothing to capture. Critically, if `validate()` raises here, the exception propagates straight out of `run_pipeline()` uncaught, and `load()` on the next line never executes — bad data can never reach either output destination.
- `print("── LOAD ───────")` / `load(summary, engine, csv_path)` — calls Section 5's function, passing through the `csv_path` parameter the caller supplied.
- `print("── DONE ✅ ────")` — only reached if all three prior stages succeeded without raising.
- `return summary` — hands the final DataFrame back to the caller for any further use.
- `final_df = run_pipeline(DB_URL)` — `DB_URL` here is the `URL` object built back in Section 0b (not the string `DB_NAME` or similar) — `create_engine()` accepts either a `URL` object or a plain connection string, so passing the object through as the `db_url` parameter works even though the parameter name suggests a plain string.

### Full code block
```python
def run_pipeline(db_url, csv_path="outputs/member_summary.csv"):
    engine = create_engine(db_url)
    print("── EXTRACT ───")
    raw = extract(engine)
    print("── TRANSFORM ──")
    summary = transform(raw)
    print("── VALIDATE ───")
    validate(summary)
    print("── LOAD ───────")
    load(summary, engine, csv_path)
    print("── DONE ✅ ────")
    return summary


# DB_URL was set in the Environment Setup cell above (loaded from .env)
final_df = run_pipeline(DB_URL)
```

---

## 8. Run the Tests

### Why this section exists
Everything up to this point has been the pipeline exercised interactively, inside a running notebook, against a live database. This final section runs an **automated, independent** test suite that checks the pipeline's logic in isolation — fast, deterministic, and without needing a live Postgres connection at all.

### Line-by-line
- `!pytest test_etl_pipeline.py -v` — the `!` prefix is Jupyter's shell-escape: everything after it runs as a shell command in a subprocess, not as Python. `pytest test_etl_pipeline.py` runs the test suite in that specific file; `-v` (verbose) prints one line per individual test function with its `PASSED`/`FAILED` status, rather than only a final summary count.
- **What's actually being tested:** `test_etl_pipeline.py` does not import anything from this notebook — it can't, since pytest needs importable `.py` modules, and a live, executed `.ipynb` isn't one. Instead, every test in the file does `from etl_pipeline_functions import extract` (or `transform`, `validate`, `load`). `etl_pipeline_functions.py` is a separate, plain-Python file in the same folder containing standalone copies of the exact same four functions defined inline in Sections 2, 3a, 4a, and 5 above — same logic, same signatures, differing only in comment style (plain `# Step 1: ...` instead of the notebook's unicode-box `── Step 1: ... ──` comments) and the presence of a bundled `run_pipeline()` convenience wrapper. Because this is a hand-maintained mirror rather than something generated from the notebook automatically, any future edit to the notebook's cells needs a matching edit in `etl_pipeline_functions.py` — otherwise the test suite would end up validating logic that's already drifted from what the notebook actually does.
- The suite's tests avoid needing a real database by building small, in-memory DataFrames (a few rows of `members`, `books`, `loans`, including a NULL city and a negative `days_overdue` row — the same seeded issues explored in Section 1b) and, for `extract()`, patching `pandas.read_sql` with `unittest.mock.patch` so no actual SQL connection is required. This is what makes the whole suite run in under 2 seconds.
- The captured run shows 20 tests across `TestExtract`, `TestTransform`, `TestValidate`, `TestLoad`, and `TestEndToEnd`, all `PASSED`, plus a few unrelated `SyntaxWarning`/`DeprecationWarning` messages coming from the `pytz` library itself (not from this pipeline's code) and a `UserWarning` from `df.to_sql(...)` about connection object types, which is expected and harmless given how the test mocks the engine.

### Full code block
```python
!pytest test_etl_pipeline.py -v
```

---

## Note on etl_pipeline_second_build.ipynb

`etl_pipeline_second_build.ipynb` is an alternate, more feature-rich build of this same Library Demo pipeline — same four ETL functions, same overall section structure — rather than a separate lab. Its markdown cells also don't have the inline-code-span rendering glitch present in `etl_pipeline.ipynb` (a useful place to cross-check what the main notebook's blanked-out headers were probably meant to say). The functional differences: `transform()` adds a derived `overdue_rate = overdue_loans / total_loans` metric per member; `validate()` checks that `overdue_rate` falls within `[0, 1]` as a hard failure, and adds a *soft* check — a NULL-`city` count that triggers a `warnings.warn(...)` instead of raising, modeling the difference between a pipeline-halting error and a non-fatal data-quality notice; Section 1's exploration cell is richer, additionally printing each table's per-column null counts (`df.isnull().sum()`) and dtypes, not just a row preview; `load()` adds a post-write verification step that reads `SELECT COUNT(*) FROM member_summary` back from Postgres immediately after writing and raises `RuntimeError` if the count doesn't match `len(df)`, catching a silent write failure that the main notebook's `load()` would miss entirely; and `run_pipeline()` wraps each of the four stages with `time.perf_counter()` calls to time it individually, printing a final "Pipeline summary" block with per-step and total timings. None of these differences change the core ETL pattern being taught — they're additive robustness and observability features layered onto the same four-function design.
