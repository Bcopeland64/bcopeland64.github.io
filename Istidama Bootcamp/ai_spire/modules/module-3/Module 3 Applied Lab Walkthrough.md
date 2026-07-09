# SQL Analytics Demo (Jordan National University) — Code Walkthrough

This document walks every cell in [m3-l3-sql-analytics.ipynb](m3-l3-sql-analytics.ipynb), the Module 3 SI (Support Instructor) demo notebook. Each section explains the Python and SQL line-by-line — every `SELECT`/`JOIN`/`WHERE`/`GROUP BY`/window clause included — then shows the full code or SQL block at the end so it can be copy-pasted cleanly. The notebook builds a small in-memory SQLite database for a fictional "Jordan National University" and runs five demo queries (`JOIN`, `GROUP BY` + `HAVING`, window `RANK`, `LEFT JOIN` + `COALESCE`, CTE) that mirror the exact concepts learners then apply independently to the real Levant Tech Solutions database in `lab_queries.sql`. Read alongside the notebook.

---

## 1. Setup — install packages and connect to SQLite

### Why this section exists
Before any `%%sql` cell can run, the notebook needs the `ipython-sql` magic registered and a live database connection open. The notebook opens with a markdown title cell and a table mapping each of the five demos to its SQL concept and business question (JOIN → "who teaches what," GROUP BY/HAVING → "which courses average above 80," window RANK → "top courses per faculty," LEFT JOIN/COALESCE → "which courses have zero enrollments," CTE → "which faculties are above-average staffed"). This section is the plumbing that makes the rest of those demos possible.

### 1a. Install `ipython-sql` and `prettytable`

#### Line-by-line
- `%pip install ipython-sql prettytable==0.7.2 --quiet` — `%pip` is IPython's magic form of `pip install`; unlike a shell-escaped `!pip install`, it's guaranteed to install into the same Python environment the running kernel uses, which matters on machines with multiple Python installs. `ipython-sql` is the package that adds the `%sql` (single-line) and `%%sql` (whole-cell) magics used throughout the rest of the notebook. `prettytable==0.7.2` is pinned to the legacy 0.7.x release line because newer `prettytable` releases (1.0+) renamed and removed methods that older `ipython-sql` versions call internally to render result tables — pinning avoids a runtime `AttributeError` when a query result is displayed. `--quiet` suppresses pip's normal per-package download/resolve chatter so the cell output stays short.
- `print("✅ ipython-sql + prettytable installed")` — a visible confirmation line so the demo instructor can see at a glance that the install step succeeded before moving on. The cell's own output also shows a `SyntaxWarning` from `pytz` (an old regex with an unescaped `\s`); that warning is benign and unrelated to this lab — it fires any time `pytz` is imported on this Python version and can be ignored.

#### Full code block
```python
%pip install ipython-sql prettytable==0.7.2 --quiet
print("✅ ipython-sql + prettytable installed")
```

### 1b. Load the SQL magic and connect

#### Line-by-line
- `%load_ext sql` — loads the `ipython-sql` extension into the current kernel, which registers the `%sql` and `%%sql` magics. Without this line, any `%%sql` cell later in the notebook would fail with `UsageError: Cell magic '%%sql' not found`.
- `%sql sqlite:///:memory:` — a single-line `%sql` magic whose argument is a SQLAlchemy-style connection string. Breaking it down: `sqlite://` is the dialect/driver prefix, and `:memory:` (in place of a file path) tells SQLite to create a private, transient, in-RAM database rather than a file on disk. This means the entire schema and every seeded row exist only for the lifetime of the kernel — restarting the kernel wipes everything, which is exactly what you want for a repeatable demo: every run starts from a clean slate.
- `print("✅ In-memory SQLite database connected")` — visible confirmation that the connection was accepted before any `CREATE TABLE` statement runs.

#### Full code block
```python
%load_ext sql
%sql sqlite:///:memory:
print("✅ In-memory SQLite database connected")
```

---

## 2. Schema — create the four tables

### Why this section exists
Every demo query in this notebook joins across the same four tables, so they must exist with the right primary keys, foreign keys, and column types before any data can be inserted. The relationships mirror the real Levant Tech schema (`departments → employees → projects → project_assignments`) one-for-one: `faculties → instructors → courses → enrollments`.

### Line-by-line
- `%%sql` — the cell magic (must be the very first line of the cell) that tells `ipython-sql` to treat the *entire rest of the cell* as one or more SQL statements to send to the connected database, rather than as Python.
- `CREATE TABLE IF NOT EXISTS faculties (...)` — `IF NOT EXISTS` makes the statement idempotent: re-running the cell (e.g., after a kernel restart didn't happen but the instructor re-runs cells top to bottom) won't error with "table already exists." `faculty_id INTEGER PRIMARY KEY AUTOINCREMENT` creates a surrogate key — `AUTOINCREMENT` guarantees SQLite never reuses an id even if a row is later deleted, which matters because other tables store `faculty_id` as a foreign key reference. `name TEXT NOT NULL` is required; `dean` and `building` are optional descriptive columns.
- `CREATE TABLE IF NOT EXISTS instructors (...)` — `instructor_id` is the surrogate primary key. `faculty_id INTEGER NOT NULL REFERENCES faculties(faculty_id)` is a foreign key: every instructor must belong to exactly one faculty, and SQLite (when foreign-key enforcement is on) rejects an insert that references a nonexistent `faculty_id`. `rank` stores the academic rank as free text (`'Professor'`, `'Associate Professor'`, etc.). `hire_date TEXT` stores a full ISO-8601 datetime string (`'YYYY-MM-DD HH:MM:SS'`) rather than using a native date type — SQLite has no dedicated `DATETIME` storage class, so the convention is to store dates as `TEXT` in a format that also sorts correctly lexicographically.
- `CREATE TABLE IF NOT EXISTS courses (...)` — `course_id` is the surrogate key used internally for joins. `course_code TEXT NOT NULL UNIQUE` is a separate *business key* — the human-readable code like `ENG-001` — with a `UNIQUE` constraint so no two courses can share a code, independent of the numeric `course_id`. `faculty_id` is a required FK. `instructor_id INTEGER REFERENCES instructors(instructor_id)` is deliberately **nullable** (no `NOT NULL`) — a course can exist without an assigned instructor yet. `credits INTEGER DEFAULT 3` supplies a sensible default so most `INSERT` statements don't need to specify it. `semester` is free text (`'2023-Fall'`).
- `CREATE TABLE IF NOT EXISTS enrollments (...)` — `enrollment_id` is the surrogate key. `course_id INTEGER NOT NULL REFERENCES courses(course_id)` ties every enrollment to exactly one course. `student_name TEXT NOT NULL` is required. `grade REAL` is nullable (a real-world enrollments table would have ungraded/in-progress rows). `enrolled_date` follows the same ISO-8601 `TEXT` convention as `hire_date`.

### Full code block
```sql
%%sql
-- ============================================================
-- SCHEMA: Jordan National University
-- ============================================================
CREATE TABLE IF NOT EXISTS faculties (
    faculty_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    dean        TEXT,
    building    TEXT
);

CREATE TABLE IF NOT EXISTS instructors (
    instructor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    faculty_id    INTEGER NOT NULL REFERENCES faculties(faculty_id),
    rank          TEXT,
    hire_date     TEXT  -- stored as ISO-8601 datetime: 'YYYY-MM-DD HH:MM:SS'
);

CREATE TABLE IF NOT EXISTS courses (
    course_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code   TEXT NOT NULL UNIQUE,  -- e.g. ENG-001, CS-002
    title         TEXT NOT NULL,
    faculty_id    INTEGER NOT NULL REFERENCES faculties(faculty_id),
    instructor_id INTEGER REFERENCES instructors(instructor_id),
    credits       INTEGER DEFAULT 3,
    semester      TEXT
);

CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id     INTEGER NOT NULL REFERENCES courses(course_id),
    student_name  TEXT NOT NULL,
    grade         REAL,
    enrolled_date TEXT NOT NULL  -- stored as ISO-8601 datetime: 'YYYY-MM-DD HH:MM:SS'
);
```

---

## 3. Seed data — insert faculties, instructors, courses, enrollments

### Why this section exists
The demo needs realistic, hand-crafted data with two intentional "gotchas" baked in — a course with zero enrollments and an uneven instructor headcount across faculties — because those exact gaps are what Demo 4 and Demo 5 are designed to surface.

### Line-by-line
- `INSERT INTO faculties (name, dean, building) VALUES (...)` — five rows, one per faculty. `faculty_id` is omitted from the column list because it's `AUTOINCREMENT`; SQLite assigns `1` through `5` in insertion order, which is why later `INSERT` statements can safely hardcode `faculty_id = 1` to mean Engineering.
- `INSERT INTO instructors (name, faculty_id, rank, hire_date) VALUES (...)` — twelve rows. Counting the `faculty_id` values: faculty `1` (Engineering) appears **three** times (Omar Saleh, Rania Khoury, Tarek Fahed), while every other faculty appears exactly **twice**. This 3-vs-2 imbalance is not incidental — it's the exact condition Demo 5's CTE query is built to detect (Engineering and Computer Science end up above the university-wide average).
- `INSERT INTO courses (course_code, title, faculty_id, instructor_id, credits, semester) VALUES (...)` — thirteen rows. `course_code` encodes the faculty as a prefix (`ENG-`, `CS-`, `BUS-`, `ART-`, `MED-`) followed by a zero-padded sequence number. The trailing comment `-- zero enrollments` flags the last row, `('ENG-004', 'Advanced Robotics', 1, 1, 3, '2023-Fall')` — note it reuses `instructor_id = 1` (Dr. Omar Saleh), meaning one instructor teaches two courses (`ENG-001` and `ENG-004`), and `ENG-004` is deliberately never referenced by any `enrollments` row below.
- `INSERT INTO enrollments (course_id, student_name, grade, enrolled_date) VALUES (...)` — forty-eight rows, grouped in comment-labeled blocks per course (`-- Course 1: Structural Engineering I (avg ~75)` and so on). Each block's grades were chosen so the resulting average lands near the number stated in its comment — this is what lets an instructor "know the data cold" and recite expected query outputs without re-running anything mid-demo. `enrolled_date` values increment by 5 minutes per row (`09:00:00`, `09:05:00`, `09:10:00`, …) purely to keep timestamps unique and plausible; the actual gap of minutes carries no business meaning here. The trailing comment `-- Course 13 (Advanced Robotics / ENG-004): ZERO enrollments — intentional for Demo 4` documents, in the SQL itself, that no row in this `INSERT` references `course_id = 13` — that absence is the whole point of Demo 4.

### Full code block
```sql
%%sql
-- ============================================================
-- SEED: faculties (5 rows)
-- ============================================================
INSERT INTO faculties (name, dean, building) VALUES
    ('Faculty of Engineering',       'Dr. Riad Haddad',   'Block A'),
    ('Faculty of Computer Science',  'Dr. Layla Mansour', 'Block B'),
    ('Faculty of Business',          'Dr. Samir Khalil',  'Block C'),
    ('Faculty of Arts & Humanities', 'Dr. Hana Nimri',    'Block D'),
    ('Faculty of Medicine',          'Dr. Ziad Barakat',  'Block E');

-- ============================================================
-- SEED: instructors (12 rows)  — hire_date as full datetime
-- ============================================================
INSERT INTO instructors (name, faculty_id, rank, hire_date) VALUES
('Dr. Omar Saleh',    1, 'Professor',          '2010-09-01 08:00:00'),
('Dr. Rania Khoury',  1, 'Associate Professor','2015-02-15 08:00:00'),
('Dr. Tarek Fahed',   1, 'Assistant Professor','2019-08-20 08:00:00'),
('Dr. Mona Issa',     2, 'Professor',          '2008-09-01 08:00:00'),
('Dr. Ali Hamdan',    2, 'Associate Professor','2013-01-10 08:00:00'),
('Dr. Sara Jabri',    2, 'Assistant Professor','2020-03-01 08:00:00'),
('Dr. Nadia Obeid',   3, 'Professor',          '2009-09-01 08:00:00'),
('Dr. Wael Masri',    3, 'Associate Professor','2016-06-15 08:00:00'),
('Dr. Lara Bitar',    4, 'Professor',          '2011-09-01 08:00:00'),
('Dr. Faris Shahin',  4, 'Assistant Professor','2021-01-20 08:00:00'),
('Dr. Rana Zawaideh', 5, 'Professor',          '2007-09-01 08:00:00'),
('Dr. Heba Srour',    5, 'Associate Professor','2014-05-01 08:00:00');

-- ============================================================
-- SEED: courses (13 rows — course 13 has ZERO enrollments)
-- course_code format: <FACULTY_PREFIX>-<AUTOINCREMENT_ID padded to 3>
-- ============================================================
INSERT INTO courses (course_code, title, faculty_id, instructor_id, credits, semester) VALUES
('ENG-001', 'Structural Engineering I',     1,  1, 3, '2023-Fall'),
('ENG-002', 'Thermodynamics',               1,  2, 3, '2023-Fall'),
('ENG-003', 'Circuit Design',               1,  3, 3, '2023-Fall'),
('CS-001',  'Algorithms & Data Structures', 2,  4, 3, '2023-Fall'),
('CS-002',  'Database Systems',             2,  5, 3, '2023-Fall'),
('CS-003',  'Machine Learning Fundamentals',2,  6, 3, '2023-Fall'),
('BUS-001', 'Business Strategy',            3,  7, 3, '2023-Fall'),
('BUS-002', 'Financial Accounting',         3,  8, 3, '2023-Fall'),
('ART-001', 'Arabic Literature',            4,  9, 3, '2023-Fall'),
('ART-002', 'Modern History',               4, 10, 3, '2023-Fall'),
('MED-001', 'Human Anatomy',                5, 11, 4, '2023-Fall'),
('MED-002', 'Pharmacology',                 5, 12, 4, '2023-Fall'),
('ENG-004', 'Advanced Robotics',            1,  1, 3, '2023-Fall');  -- zero enrollments

-- ============================================================
-- SEED: enrollments (48 rows)  — enrolled_date as full datetime
-- ============================================================
INSERT INTO enrollments (course_id, student_name, grade, enrolled_date) VALUES
-- Course 1: Structural Engineering I  (avg ~75)
(1,'Ahmad Zreiqat', 72.0,'2023-09-01 09:00:00'),(1,'Dina Rawashdeh',80.0,'2023-09-01 09:05:00'),
(1,'Yara Twal',     68.0,'2023-09-01 09:10:00'),(1,'Murad Aqrabawi',76.0,'2023-09-01 09:15:00'),
(1,'Reem Khaled',   79.0,'2023-09-01 09:20:00'),
-- Course 2: Thermodynamics  (avg ~83)
(2,'Loay Salti',    85.0,'2023-09-01 09:00:00'),(2,'Hana Balbisi',  82.0,'2023-09-01 09:05:00'),
(2,'Firas Hijazi',  81.0,'2023-09-01 09:10:00'),(2,'Nour Atiyeh',   84.0,'2023-09-01 09:15:00'),
(2,'Lina Jado',     83.0,'2023-09-01 09:20:00'),
-- Course 3: Circuit Design  (avg ~71)
(3,'Basel Shatnawi',70.0,'2023-09-01 09:00:00'),(3,'Rawan Zawahreh',73.0,'2023-09-01 09:05:00'),
(3,'Salam Bisharat',69.0,'2023-09-01 09:10:00'),(3,'Nura Haj',      72.0,'2023-09-01 09:15:00'),
-- Course 4: Algorithms & Data Structures  (avg ~88)
(4,'Omar Barakat',  90.0,'2023-09-01 09:00:00'),(4,'Lina Nasser',   87.0,'2023-09-01 09:05:00'),
(4,'Sara Mansour',  88.0,'2023-09-01 09:10:00'),(4,'Tarek Haddad',  86.0,'2023-09-01 09:15:00'),
(4,'Mona Fahed',    89.0,'2023-09-01 09:20:00'),(4,'Ali Khalil',    88.0,'2023-09-01 09:25:00'),
-- Course 5: Database Systems  (avg ~79)
(5,'Dalia Rashid',  78.0,'2023-09-01 09:00:00'),(5,'Kareem Saad',   81.0,'2023-09-01 09:05:00'),
(5,'Heba Qasem',    77.0,'2023-09-01 09:10:00'),(5,'Yousef Saleh',  80.0,'2023-09-01 09:15:00'),
-- Course 6: Machine Learning Fundamentals  (avg ~84)
(6,'Rania Hamdan',  85.0,'2023-09-01 09:00:00'),(6,'Basem Ayyad',   83.0,'2023-09-01 09:05:00'),
(6,'Rana Nimri',    84.0,'2023-09-01 09:10:00'),(6,'Ziad Khoury',   84.0,'2023-09-01 09:15:00'),
(6,'Lara Issa',     84.0,'2023-09-01 09:20:00'),
-- Course 7: Business Strategy  (avg ~76)
(7,'Maya Jabri',    75.0,'2023-09-01 09:00:00'),(7,'Sami Awad',     78.0,'2023-09-01 09:05:00'),
(7,'Heba Houri',    74.0,'2023-09-01 09:10:00'),(7,'Jad Dahlan',    77.0,'2023-09-01 09:15:00'),
-- Course 8: Financial Accounting  (avg ~81)
(8,'Lama Kanaan',   82.0,'2023-09-01 09:00:00'),(8,'Ahmad Nassar',  80.0,'2023-09-01 09:05:00'),
(8,'Rami Dabbas',   83.0,'2023-09-01 09:10:00'),(8,'Lena Farhat',   79.0,'2023-09-01 09:15:00'),
-- Course 9: Arabic Literature  (avg ~72)
(9,'Khalid Madi',   72.0,'2023-09-01 09:00:00'),(9,'Sana Jaber',    74.0,'2023-09-01 09:05:00'),
(9,'Nidal Zreiqat', 71.0,'2023-09-01 09:10:00'),
-- Course 10: Modern History  (avg ~82)
(10,'Hana Asmar',   83.0,'2023-09-01 09:00:00'),(10,'Wael Hadid',   81.0,'2023-09-01 09:05:00'),
(10,'Deema Halabi', 82.0,'2023-09-01 09:10:00'),(10,'Nadia Farraj', 82.0,'2023-09-01 09:15:00'),
-- Course 11: Human Anatomy  (avg ~86)
(11,'Eyad Sabbah',  87.0,'2023-09-01 09:00:00'),(11,'Roula Qudah',  85.0,'2023-09-01 09:05:00'),
(11,'Waseem Khalaf',86.0,'2023-09-01 09:10:00'),(11,'Abeer Salti',  86.0,'2023-09-01 09:15:00'),
(11,'Dina Hijazi',  86.0,'2023-09-01 09:20:00'),
-- Course 12: Pharmacology  (avg ~77)
(12,'Tariq Obeid',  78.0,'2023-09-01 09:00:00'),(12,'Yara Bitar',   76.0,'2023-09-01 09:05:00'),
(12,'Firas Masri',  77.0,'2023-09-01 09:10:00'),(12,'Nour Shahin',  77.0,'2023-09-01 09:15:00');
-- Course 13 (Advanced Robotics / ENG-004): ZERO enrollments — intentional for Demo 4
```

---

## 4. Verify the seed load

### Why this section exists
A one-cell sanity check, run immediately after seeding, so the instructor confirms the database is in the expected state before any demo query runs live in front of the room.

### Line-by-line
- `SELECT 'faculties' AS tbl, COUNT(*) AS rows FROM faculties` — a literal string `'faculties'` is selected alongside a real aggregate `COUNT(*)`, producing a single row that labels the count with the table name it came from.
- `UNION ALL SELECT 'instructors', COUNT(*) FROM instructors` — stacks a second labeled count row underneath the first. `UNION ALL` (rather than plain `UNION`) is used deliberately: `UNION` would first deduplicate identical rows across the combined result, which costs extra work and is pointless here since every `(label, count)` pair is already guaranteed unique by its label.
- The pattern repeats for `courses` and `enrollments`, producing four rows total.
- The trailing comment `-- Expected: 5 / 12 / 13 / 48` states the exact counts an instructor should see and cross-checks directly against the seed block in Section 3 — 5 faculties, 12 instructors, 13 courses, 48 enrollment rows (course 13/`ENG-004` contributes 0 of those 48, by design).

### Full code block
```sql
%%sql
SELECT 'faculties'   AS tbl, COUNT(*) AS rows FROM faculties   UNION ALL
SELECT 'instructors',         COUNT(*)         FROM instructors UNION ALL
SELECT 'courses',             COUNT(*)         FROM courses     UNION ALL
SELECT 'enrollments',         COUNT(*)         FROM enrollments;
-- Expected: 5 / 12 / 13 / 48
```

---

## 5. Demo 1 — JOIN

### Why this section exists
> *"Which instructor teaches each course, and what faculty are they from?"*

Right before this demo, the notebook shows a markdown "Schema Reference" cell with an ASCII entity diagram connecting `faculties → instructors` and `courses → enrollments` via their foreign keys, plus three callouts worth carrying into every later demo: `course_code` follows the `<FACULTY_PREFIX>-<3-digit sequence>` pattern; dates are ISO-8601 `TEXT`; `ENG-004` has zero enrollments (Demo 4's target) and Engineering has 3 instructors versus 2 everywhere else (Demo 5's target). Demo 1 itself teaches the most basic multi-table pattern: combining two tables on a shared foreign key.

### 5a. Explore instructors

#### Line-by-line
- `SELECT * FROM instructors LIMIT 5;` — a raw look at the table before joining anything. `LIMIT 5` caps the result to the first 5 rows by insertion order, keeping the projected output short. The point of this step is purely pedagogical: it lets the room see that `faculty_id` in this table is just an opaque integer (`1`, `1`, `1`, `2`, `2`, …) with no human-readable meaning on its own — motivating why a `JOIN` to `faculties` is needed next.

#### Full code block
```sql
%%sql
-- Step 1: explore instructors — notice faculty_id is just a number
SELECT * FROM instructors LIMIT 5;
```

### 5b. Explore faculties

#### Line-by-line
- `SELECT * FROM faculties;` — the full 5-row faculties table (no `LIMIT` needed since there are only 5 rows). This is where the human-readable `name` column (`'Faculty of Engineering'`, etc.) actually lives — the piece missing from the `instructors` table above.

#### Full code block
```sql
%%sql
-- Step 2: explore faculties — this has the human-readable name
SELECT * FROM faculties;
```

### 5c. Write the JOIN

#### Line-by-line
- `SELECT c.course_code, c.title AS course_title, i.name AS instructor_name, i.rank, f.name AS faculty_name` — the projection. Table aliases (`c`, `i`, `f`) keep the query compact once three tables are in play. `i.name` and `f.name` are both aliased (`instructor_name`, `faculty_name`) because both `instructors` and `faculties` have a column literally called `name` — without aliasing, the result set would have two ambiguous `name` columns.
- `FROM courses c` — the anchor table; the query is framed as "for each course, look up its instructor and faculty," so `courses` is where iteration starts.
- `JOIN instructors i ON c.instructor_id = i.instructor_id` — an **inner join** (writing bare `JOIN` in SQL always means `INNER JOIN`) that attaches the one matching instructor row for each course, matched on the foreign key `courses.instructor_id = instructors.instructor_id`. Because it's an inner join, any course whose `instructor_id` didn't match a row in `instructors` would simply be dropped from the output with no error — a behavior this same notebook deliberately exploits as a bug in Demo 4.
- `JOIN faculties f ON c.faculty_id = f.faculty_id` — a second inner join, chained onto the result of the first, attaching the faculty name via `courses.faculty_id` (note: this joins on the *course's own* `faculty_id`, not the instructor's — in this seed data the two always agree, but the query is explicit about which table's FK it trusts).
- `ORDER BY f.name, c.course_code` — sorts the final output alphabetically by faculty, then by course code within each faculty, producing a business-readable grouped listing rather than raw insertion order.
- **Key insight** (from the notebook's own clause table): `JOIN` only returns rows with a match in *both* tables. All 13 courses appear in this output because every course row has a non-null `instructor_id` and `faculty_id` that successfully matches — there's no missing-match case to expose yet in this demo.

#### Full code block
```sql
%%sql
-- Step 3: JOIN on the shared key (faculty_id) to combine both tables
-- Also show course_code so students see it alongside the instructor
SELECT
    c.course_code,
    c.title         AS course_title,
    i.name          AS instructor_name,
    i.rank,
    f.name          AS faculty_name
FROM courses c
JOIN instructors i ON c.instructor_id = i.instructor_id
JOIN faculties   f ON c.faculty_id    = f.faculty_id
ORDER BY f.name, c.course_code;
```

---

## 6. Demo 2 — GROUP BY + HAVING

### Why this section exists
> *"Which courses have a class average above 80?"*

This demo teaches the distinction between filtering rows (`WHERE`, before aggregation) and filtering groups (`HAVING`, after aggregation) — one of the most common sources of runtime errors for SQL beginners.

### 6a. Raw per-course averages

#### Line-by-line
- `SELECT course_id, AVG(grade) AS avg_grade FROM enrollments GROUP BY course_id;` — `AVG(grade)` is an aggregate function computed once per group; it automatically ignores any `NULL` grade values rather than treating them as 0. `GROUP BY course_id` collapses the 48 individual enrollment rows into one row per distinct `course_id` value that actually appears in `enrollments` — which is why the result has only 12 rows, not 13: `course_id = 13` (`ENG-004`) never appears in `enrollments` at all, so it can't be grouped and is silently absent rather than shown with a `0` or `NULL` average. That absence is the exact gap Demo 4 exists to catch.
- The displayed output includes `(9, 72.33333333333333)` — Arabic Literature's 3 grades (72, 74, 71) don't divide evenly by 3, so the raw floating-point average is shown with full precision; this motivates the `ROUND()` used in the next step.

#### Full code block
```sql
%%sql
-- Step 1: raw average per course (course_id only — no labels yet)
SELECT course_id, AVG(grade) AS avg_grade
FROM enrollments
GROUP BY course_id;
```

### 6b. Join for labels, filter with HAVING

#### Line-by-line
- `SELECT c.course_code, c.title AS course_title, ROUND(AVG(e.grade), 2) AS avg_grade, COUNT(e.enrollment_id) AS student_count` — adds the human-readable course code/title via a join, wraps the average in `ROUND(..., 2)` to display two decimal places, and adds `COUNT(e.enrollment_id)` so the reader can see how many students each average is based on.
- `FROM enrollments e JOIN courses c ON e.course_id = c.course_id` — starts from `enrollments` this time (rather than `courses`, as in Demo 1) and inner-joins in the course details. Starting from `enrollments` still excludes `course_id = 13` automatically, since it has no rows there to begin with.
- `GROUP BY c.course_id, c.course_code, c.title` — every non-aggregated column in the `SELECT` list must appear here. SQLite is actually lenient about this rule (it would tolerate an incomplete `GROUP BY`), but the query is written to satisfy the *strict* rule that PostgreSQL enforces — the same engine learners use in the real lab — so the pattern transfers directly.
- `HAVING AVG(e.grade) > 80` — filters *groups*, not individual rows, after the `GROUP BY` and its aggregates have already been computed. This is the critical distinction from `WHERE`: `WHERE` executes before grouping, when no aggregate value like `AVG(e.grade)` exists yet to compare against, so `WHERE AVG(e.grade) > 80` would raise an error. `HAVING` runs *after* aggregation, when `AVG(e.grade)` is a real per-group value that can be compared.
- `ORDER BY avg_grade DESC` — highest-performing courses listed first, using the column's output alias (`avg_grade`) rather than repeating the full `ROUND(AVG(...))` expression.
- Result: 6 of the 12 non-empty courses clear the 80-average bar, led by `CS-001` at 88.0.

#### Full code block
```sql
%%sql
-- Step 2: join course title + filter with HAVING
-- HAVING filters AFTER grouping; WHERE cannot see AVG() results
SELECT
    c.course_code,
    c.title                  AS course_title,
    ROUND(AVG(e.grade), 2)   AS avg_grade,
    COUNT(e.enrollment_id)   AS student_count
FROM enrollments e
JOIN courses c ON e.course_id = c.course_id
GROUP BY c.course_id, c.course_code, c.title
HAVING AVG(e.grade) > 80
ORDER BY avg_grade DESC;
```

---

## 7. Demo 3 — Window Function: RANK

### Why this section exists
> *"For each faculty, which courses have the highest enrollment? Rank them."*

This is the conceptually hardest demo: it introduces window functions, which compute a per-row value (like a rank) without collapsing rows the way `GROUP BY` does.

### 7a. Base query — enrollment count per course

#### Line-by-line
- `SELECT f.name AS faculty_name, c.course_code, c.title AS course_title, COUNT(e.enrollment_id) AS enrollment_count` — projects the faculty name, course identifiers, and a count of matching enrollment rows per course.
- `FROM courses c JOIN faculties f ON c.faculty_id = f.faculty_id` — inner join to attach the readable faculty name (safe here since every course has a valid `faculty_id`).
- `LEFT JOIN enrollments e ON c.course_id = e.course_id` — a **left join**, not an inner join, is used here even though this demo's official topic is window functions, not joins — it's required so `ENG-004` (zero enrollments) still appears in the ranking with a count of `0` rather than vanishing. `COUNT(e.enrollment_id)` naturally returns `0` for that course because a left join with no match produces `NULL` for every `enrollments` column, and `COUNT(column)` skips `NULL`s (as opposed to `COUNT(*)`, which would count the phantom joined row itself).
- `GROUP BY f.name, c.course_code, c.title` — one row per course, consistent with the aggregate `COUNT()` in the `SELECT` list.
- `ORDER BY f.name, COUNT(e.enrollment_id) DESC` — groups the output visually by faculty and, within each faculty, shows highest-enrollment courses first — previewing by eye what the window function will compute formally in the next step.

#### Full code block
```sql
%%sql
-- Step 1: enrollment count per course (inner query)
SELECT
    f.name                 AS faculty_name,
    c.course_code,
    c.title                AS course_title,
    COUNT(e.enrollment_id) AS enrollment_count
FROM courses c
JOIN faculties f ON c.faculty_id = f.faculty_id
LEFT JOIN enrollments e ON c.course_id = e.course_id
GROUP BY f.name, c.course_code, c.title
ORDER BY f.name, COUNT(e.enrollment_id) DESC;
```

### 7b. Wrap in a subquery and add RANK()

#### Line-by-line
- `FROM ( ... ) sub` — the entire query from Step 1 (7a) is nested unchanged as a subquery aliased `sub`. This is necessary because window functions can't be applied directly alongside a `GROUP BY` in the same `SELECT` — the aggregation has to be fully resolved first, then the window function operates on the *already-grouped* rows.
- `RANK() OVER (PARTITION BY faculty_name ORDER BY enrollment_count DESC) AS enrollment_rank` — the window function itself:
  - `PARTITION BY faculty_name` divides the rows into independent buckets, one per distinct faculty. Unlike `GROUP BY`, partitioning does **not** collapse rows — every course row survives; each just gets a new computed column.
  - `ORDER BY enrollment_count DESC` (inside the `OVER(...)` clause) defines the ranking order *within* each partition — highest enrollment count first, so that course becomes rank 1 for its faculty, and ranking restarts at 1 for the next faculty.
  - `RANK()` specifically assigns **tied rows the same rank number**, and the next distinct value's rank skips ahead by the number of ties. This is visible in the actual output: in Engineering, `ENG-001` and `ENG-002` both have 5 enrollments and both get rank `1`; `ENG-003` (4 enrollments) gets rank `3` — not `2` — because two rows already occupy rank 1. `ENG-004` (0 enrollments) gets rank `4`. Contrast with `ROW_NUMBER()`, mentioned in the notebook's clause-summary table, which would force strictly unique numbers even among ties, arbitrarily breaking them.
- `ORDER BY faculty_name, enrollment_rank` — the outer, final sort: alphabetical by faculty, then by the computed rank within each.

#### Full code block
```sql
%%sql
-- Step 2: wrap in subquery and add RANK() partitioned by faculty
-- RANK() ties share the same rank number; the next rank skips accordingly
SELECT
    faculty_name,
    course_code,
    course_title,
    enrollment_count,
    RANK() OVER (
        PARTITION BY faculty_name
        ORDER BY enrollment_count DESC
    ) AS enrollment_rank
FROM (
    SELECT
        f.name                 AS faculty_name,
        c.course_code,
        c.title                AS course_title,
        COUNT(e.enrollment_id) AS enrollment_count
    FROM courses c
    JOIN faculties f ON c.faculty_id = f.faculty_id
    LEFT JOIN enrollments e ON c.course_id = e.course_id
    GROUP BY f.name, c.course_code, c.title
) sub
ORDER BY faculty_name, enrollment_rank;
```

---

## 8. Demo 4 — LEFT JOIN + COALESCE

### Why this section exists
> *"Are there any courses that no student has enrolled in this semester?"*

This is flagged in the instructor's guide as "the most important demo moment" — it shows, live, how an `INNER JOIN` can silently drop real data with no error, and how `LEFT JOIN` fixes it.

### 8a. The broken INNER JOIN version

#### Line-by-line
- `SELECT c.course_code, c.title, COUNT(e.enrollment_id) AS enroll_count FROM courses c JOIN enrollments e ON c.course_id = e.course_id GROUP BY c.course_code, c.title ORDER BY enroll_count;` — this is deliberately written wrong, as a teaching device. `JOIN` here is an inner join, which requires a matching row in `enrollments` for a course to appear at all. Since `ENG-004` has zero rows in `enrollments`, it never enters the joined result set in the first place — `COUNT()` never even gets a chance to return `0` for it, because there's no row to group. The output has 12 rows, not the expected 13, and nothing in the query raises an error or warning — it just quietly reports incomplete data, which is the dangerous part.

#### Full code block
```sql
%%sql
-- Step 1: INNER JOIN — intentionally broken; shows only 12 courses (ENG-004 disappears)
SELECT
    c.course_code,
    c.title,
    COUNT(e.enrollment_id) AS enroll_count
FROM courses c
JOIN enrollments e ON c.course_id = e.course_id
GROUP BY c.course_code, c.title
ORDER BY enroll_count;
```

### 8b. The corrected LEFT JOIN + COALESCE version

#### Line-by-line
- `SELECT c.course_code, c.title AS course_title, f.name AS faculty_name, COALESCE(COUNT(e.enrollment_id), 0) AS enrollment_count` — same shape as before, plus the faculty name and a `COALESCE` wrapper around the count.
- `FROM courses c JOIN faculties f ON c.faculty_id = f.faculty_id` — inner join for the faculty label (safe — every course has one).
- `LEFT JOIN enrollments e ON c.course_id = e.course_id` — the fix. `LEFT JOIN` keeps **every** row from the left-hand table (`courses`, per the FROM/JOIN chain built so far) regardless of whether a matching `enrollments` row exists. For `ENG-004`, every `e.*` column comes back `NULL` instead of the row being dropped.
- `COALESCE(COUNT(e.enrollment_id), 0)` — `COUNT(column)` already returns `0` (not `NULL`) for a group made entirely of `NULL`s, so the `COALESCE` here is defensive "belt-and-suspenders" — it would only matter if the aggregate were swapped for something that *can* return `NULL` over an empty/all-`NULL` group, like `SUM()` or `AVG()`. The real trap this demo is teaching (per the instructor's guide) is `COUNT(*)` versus `COUNT(column)`: `COUNT(*)` would count the phantom joined row itself and wrongly report `1` for `ENG-004` instead of `0`; `COUNT(e.enrollment_id)` correctly counts only non-`NULL` enrollment ids.
- `GROUP BY c.course_id, c.course_code, c.title, f.name` — grouping columns match every non-aggregated `SELECT` column.
- `ORDER BY enrollment_count ASC, c.course_code` — ascending order surfaces the zero-enrollment course at the very top of the result — exactly the gap the business question was asking to find.
- Result: `ENG-004` / *Advanced Robotics* appears first with `enrollment_count = 0`.

#### Full code block
```sql
%%sql
-- Step 2: LEFT JOIN — keeps ALL courses; missing enrollments become 0
-- COUNT(e.enrollment_id) skips NULLs naturally, returning 0 for no-match rows
SELECT
    c.course_code,
    c.title                             AS course_title,
    f.name                              AS faculty_name,
    COALESCE(COUNT(e.enrollment_id), 0) AS enrollment_count
FROM courses c
JOIN faculties f ON c.faculty_id = f.faculty_id
LEFT JOIN enrollments e ON c.course_id = e.course_id
GROUP BY c.course_id, c.course_code, c.title, f.name
ORDER BY enrollment_count ASC, c.course_code;
```

---

## 9. Demo 5 — CTE

### Why this section exists
> *"Which faculties employ more instructors than the university average?"*

This demo introduces Common Table Expressions (CTEs) as a way to name and stack intermediate query results, built up one step at a time so the room can inspect each intermediate result before combining them.

### 9a. First CTE alone

#### Line-by-line
- `WITH university_avg AS ( ... )` — defines a CTE: a named, temporary result set scoped to the single statement that immediately follows it. It behaves like a view that exists only for the duration of this query.
- `SELECT AVG(instructor_count) AS avg_instructors FROM ( SELECT faculty_id, COUNT(*) AS instructor_count FROM instructors GROUP BY faculty_id ) counts` — the body of the CTE. The inner subquery `counts` first collapses the 12 `instructors` rows into 5 rows — one per `faculty_id` — each holding that faculty's instructor headcount (`COUNT(*)`). The outer `AVG(instructor_count)` then averages those 5 per-faculty counts into a single number: the university-wide average instructors-per-faculty.
- `SELECT * FROM university_avg;` — the statement that actually runs the CTE and displays its result on its own, before it's combined with anything else. This is a debugging habit worth teaching explicitly: verify an intermediate CTE's output (here, a single row: `2.4`) before building further logic on top of it.

#### Full code block
```sql
%%sql
-- Step 1: run the first CTE alone to inspect its output (one row = university avg)
WITH university_avg AS (
    SELECT AVG(instructor_count) AS avg_instructors
    FROM (
        SELECT faculty_id, COUNT(*) AS instructor_count
        FROM instructors
        GROUP BY faculty_id
    ) counts
)
SELECT * FROM university_avg;
```

### 9b. Add the second CTE and compare

#### Line-by-line
- `WITH university_avg AS ( ... ), faculty_counts AS ( ... )` — a second CTE, separated from the first by a comma, in the same `WITH` clause. Multiple CTEs in one `WITH` are each computed independently and can then both be referenced in the final query.
- `faculty_counts AS ( SELECT f.name AS faculty_name, COUNT(i.instructor_id) AS instructor_count FROM faculties f LEFT JOIN instructors i ON f.faculty_id = i.faculty_id GROUP BY f.faculty_id, f.name )` — uses the same `LEFT JOIN` + `COUNT(column)` pattern taught in Demo 4, applied preemptively: starting from `faculties` and left-joining `instructors` means a faculty with zero instructors would still show up with a count of `0` rather than disappearing (no faculty in this seed data actually has zero instructors, but the query is written defensively regardless).
- `SELECT fc.faculty_name, fc.instructor_count, ROUND(ua.avg_instructors, 2) AS university_avg` — the final projection, pulling one column from each CTE.
- `FROM faculty_counts fc, university_avg ua` — a comma-separated `FROM` list is an **implicit cross join**. Because `university_avg` has exactly one row, cross-joining it onto `faculty_counts` doesn't multiply the row count — one row × five rows = five rows — it effectively broadcasts that single average value as an extra constant column on every faculty's row.
- `WHERE fc.instructor_count > ua.avg_instructors` — filters faculties above the average. Plain `WHERE` (not `HAVING`) is correct here because no new aggregation is happening at this point in the query — both `instructor_count` and `avg_instructors` were already computed inside their respective CTEs; this outer query is just comparing two already-materialized columns row by row.
- `ORDER BY fc.instructor_count DESC` — highest headcount first.
- Result: only **Faculty of Engineering** (3) and **Faculty of Computer Science** (3) exceed the university average of **2.4** — exactly the 3-vs-2 imbalance that was deliberately seeded in Section 3.
- The notebook's own clause table adds a closing note worth repeating: CTEs exist for **readability**, not guaranteed performance — some engines optimize them, some don't, and the same logic could always be written as nested subqueries instead.

#### Full code block
```sql
%%sql
-- Step 2: add faculty_counts CTE and compare each faculty against the avg
-- university_avg has one row, so the implicit cross-join just adds that column everywhere
WITH university_avg AS (
    SELECT AVG(instructor_count) AS avg_instructors
    FROM (
        SELECT faculty_id, COUNT(*) AS instructor_count
        FROM instructors
        GROUP BY faculty_id
    ) counts
),
faculty_counts AS (
    SELECT
        f.name                 AS faculty_name,
        COUNT(i.instructor_id) AS instructor_count
    FROM faculties f
    LEFT JOIN instructors i ON f.faculty_id = i.faculty_id
    GROUP BY f.faculty_id, f.name
)
SELECT
    fc.faculty_name,
    fc.instructor_count,
    ROUND(ua.avg_instructors, 2) AS university_avg
FROM faculty_counts fc, university_avg ua
WHERE fc.instructor_count > ua.avg_instructors
ORDER BY fc.instructor_count DESC;
```

---

## Concept summary (from the notebook's closing cell)

The notebook ends on a markdown-only recap cell — no code to walk, but its content is worth carrying forward:

| Demo | Concept | When to use it |
|------|---------|---------------|
| 1 | `JOIN` | Combine columns from two or more tables on a shared key |
| 2 | `GROUP BY` + `HAVING` | Aggregate rows into groups; filter on aggregate results |
| 3 | Window — `RANK` | Rank rows **within** partitions without collapsing them |
| 4 | `LEFT JOIN` + `COUNT(col)` | Include rows with no right-side match (zeros, missing data) |
| 5 | CTE | Break a multi-step query into named, readable intermediate results |

**Common errors called out in the notebook:**
- **`WHERE` instead of `HAVING`** on aggregates — `WHERE` runs before aggregation and can't see `AVG()`, `SUM()`, etc.
- **Missing `GROUP BY` columns** — every non-aggregate `SELECT` column must appear in `GROUP BY` (PostgreSQL enforces this strictly; SQLite, used in this demo, is more lenient — don't rely on that leniency in the real lab).
- **`INNER JOIN` when you need `LEFT JOIN`** — if a course with zero enrollments disappears from your result, you used the wrong join.
- **Window function in `WHERE`** — wrap it in a subquery or CTE first, then filter in the outer query.

This same set of five patterns — `JOIN`, `GROUP BY`/`HAVING`, window `RANK`, `LEFT JOIN`/`COALESCE`, and CTE — is exactly what learners then apply independently, against a different domain (Levant Tech Solutions: departments, employees, projects, project_assignments), across `lab_queries.sql`'s nine questions.
