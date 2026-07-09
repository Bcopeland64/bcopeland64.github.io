# Module 9B — Music KG SI Pipeline: Complete Code Walkthrough

This document traces every line of every code cell in the demo notebook.
Each section ends with the **complete code block** for that section so you can
copy it without hunting through the annotations.

---

## Table of Contents

1. [Prerequisites — Install Dependencies](#1-prerequisites--install-dependencies)
2. [Setup — Start Docker & Connect to Neo4j](#2-setup--start-docker--connect-to-neo4j)
3. [Stage 2 — The 15-Template Surface](#3-stage-2--the-15-template-surface)
4. [Stage 3 — End-to-End Happy Path (Hierarchical Traversal)](#4-stage-3--end-to-end-happy-path-hierarchical-traversal)
5. [Stage 4 — Parameterization & Cypher Injection](#5-stage-4--parameterization--cypher-injection)
6. [Stage 5 — Fail-Loud UnsupportedQueryError](#6-stage-5--fail-loud-unsupportedqueryerror)
7. [Stage 6 — Tier 3 Swap-in & the Allowlist](#7-stage-6--tier-3-swap-in--the-allowlist)
8. [Stage 7 — Pivot to the Recipe Domain](#8-stage-7--pivot-to-the-recipe-domain)
9. [Cleanup](#9-cleanup)

---

## 1. Prerequisites — Install Dependencies

### Line-by-line explanation

```python
import subprocess
```
`subprocess` is a Python standard-library module that lets Python launch shell
commands as child processes. We use it here instead of a `!pip install` magic
command so the cell works identically inside and outside Jupyter.

```python
import sys
```
`sys` exposes runtime information about the Python interpreter.
`sys.executable` gives the absolute path to the Python binary that is currently
running this notebook — important because a system may have multiple Python
installations, and we want packages installed into *this* one.

```python
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', ...],
    check=True
)
```
`subprocess.run` executes a command and waits for it to finish.
`[sys.executable, '-m', 'pip', 'install', ...]` builds the command as a list
of strings — equivalent to running `python -m pip install ...` in a terminal.
`check=True` tells `subprocess` to raise a `CalledProcessError` if `pip` exits
with a non-zero return code (i.e., if any package fails to install).

The packages installed:

| Package | Purpose |
|---------|---------|
| `neo4j` | Official Neo4j Python driver — manages connections and Cypher execution |
| `langchain` | LLM orchestration framework used in Tier 3 |
| `langchain-community` | Community integrations, includes `GraphCypherQAChain` |
| `tabulate` | Formats Python lists into human-readable tables |

`--quiet` suppresses the verbose progress output that pip normally prints.

### Complete code block

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, '-m', 'pip', 'install',
     'neo4j',
     'langchain',
     'langchain-community',
     'tabulate',
     '--quiet'],
    check=True
)
print('All dependencies installed successfully.')
```

---

## 2. Setup — Start Docker & Connect to Neo4j

### 2a. Docker Container Check

#### Line-by-line explanation

```python
import subprocess
import time
```
`time` is used later for `time.sleep(15)` — we pause to let Neo4j finish
initializing before the driver attempts a connection.

```python
result = subprocess.run(
    ['docker', 'ps', '--filter', 'name=music-neo4j', '--format', '{{.Names}}'],
    capture_output=True,
    text=True
)
```
`docker ps` lists running containers.
`--filter name=music-neo4j` restricts output to containers whose name contains
`music-neo4j`.
`--format {{.Names}}` prints only the container name column — no table headers,
no other columns, making the result easy to check programmatically.
`capture_output=True` stores stdout and stderr inside the `result` object
instead of printing them to the terminal.
`text=True` automatically decodes the byte output to a Python `str`.

```python
if 'music-neo4j' in result.stdout:
    print('music-neo4j container is already running.')
else:
    subprocess.run(['docker', 'compose', 'up', '-d', 'music-neo4j'], check=True)
    time.sleep(15)
```
If the container name appears in `result.stdout`, the container is already up.
Otherwise, `docker compose up -d music-neo4j` starts it in detached mode (the
`-d` flag means the command returns immediately — the container runs in the
background).
`time.sleep(15)` pauses Python for 15 seconds while Neo4j's internal startup
sequence completes and the Bolt port becomes available.

### 2b. Neo4j Connection

#### Line-by-line explanation

```python
from neo4j import GraphDatabase
```
`GraphDatabase` is the top-level factory class in the Neo4j Python driver.
Its `.driver()` method creates a driver object that manages a pool of
reusable connections.

```python
NEO4J_URI      = 'bolt://localhost:7687'
NEO4J_USER     = 'neo4j'
NEO4J_PASSWORD = 'music1234'
```
`bolt://` specifies the Bolt wire protocol — Neo4j's binary protocol, faster
than HTTP for repeated query execution.
Port `7687` is Neo4j's default Bolt port.
The username and password must match what is declared in `docker-compose.yml`.

```python
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
```
Creates the driver. The driver maintains an internal connection pool; it does
not open a connection to Neo4j until the first query is run.
`auth=` accepts a `(username, password)` tuple.

```python
driver.verify_connectivity()
```
Sends a lightweight ping to Neo4j. Raises `neo4j.exceptions.ServiceUnavailable`
if the container is not reachable — surfaces the problem immediately rather
than letting the first query fail with a cryptic error.

```python
with driver.session() as session:
    total = session.run('MATCH (n) RETURN count(n) AS c').single()['c']
    labels = [r['label'] for r in
              session.run('CALL db.labels() YIELD label RETURN label ORDER BY label')]
```
`driver.session()` opens a logical transaction scope.
The `with` block ensures the session is closed (and its connection returned to
the pool) even if an exception is raised.
`MATCH (n) RETURN count(n) AS c` is a basic node count — a quick health check
that the KG is populated.
`.single()` retrieves the one expected result row; `['c']` accesses the named
column.
`CALL db.labels()` is a built-in Neo4j procedure that returns every node label
in the database — useful for verifying the KG schema at a glance.
The list comprehension `[r['label'] for r in ...]` collects all label strings
into a plain Python list.

### Complete code block

```python
import subprocess
import time

result = subprocess.run(
    ['docker', 'ps', '--filter', 'name=music-neo4j', '--format', '{{.Names}}'],
    capture_output=True,
    text=True
)

if 'music-neo4j' in result.stdout:
    print('music-neo4j container is already running.')
else:
    print('Container not running. Starting music-neo4j...')
    subprocess.run(['docker', 'compose', 'up', '-d', 'music-neo4j'], check=True)
    print('Waiting 15 s for Neo4j to initialize...')
    time.sleep(15)
    print('Container ready.')

# ──────────────────────────────────────────────────────────────────────────────

from neo4j import GraphDatabase

NEO4J_URI      = 'bolt://localhost:7687'
NEO4J_USER     = 'neo4j'
NEO4J_PASSWORD = 'music1234'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print(f'Connected to Neo4j at {NEO4J_URI}')

with driver.session() as session:
    total  = session.run('MATCH (n) RETURN count(n) AS c').single()['c']
    labels = [r['label'] for r in
              session.run('CALL db.labels() YIELD label RETURN label ORDER BY label')]

print(f'KG loaded: {total} nodes')
print(f'Node labels: {labels}')
```

---

## 3. Stage 2 — The 15-Template Surface

### 3a. The ShapeId Enum

#### Line-by-line explanation

```python
from enum import Enum, auto
```
`Enum` is the base class for creating enumerated types in Python.
`auto()` automatically assigns sequential integer values to enum members so
you do not need to maintain magic numbers.

```python
class ShapeId(Enum):
    Q1 = auto()  # Tracks by instrument
    ...
    Q15 = auto()  # Most popular tracks
```
Each `ShapeId` member represents one of the 15 valid question shapes the system
supports.
The bounded schema (small, fixed node types and relationship types) is what
makes this enumeration finite — a larger schema would require a different
architecture.
Using an `Enum` rather than bare strings gives you type safety: a typo like
`ShapeId.Q99` raises an `AttributeError` at definition time rather than
silently producing a wrong result at query time.

### 3b. The TEMPLATES Dictionary

#### Line-by-line explanation

```python
TEMPLATES = {
    ShapeId.Q1: '''
        MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {name: $instrument})
        ...
    ''',
    ...
}
```
`TEMPLATES` is a plain Python dict mapping every `ShapeId` to a multi-line
Cypher string.
The triple-quoted string `'''...'''` preserves newlines, making the Cypher
readable.
**Every slot value is a `$parameter` placeholder** — never a Python variable
interpolated directly into the string. This is the single most important
rule in the entire codebase: it prevents Cypher injection (covered in Stage 4).

```python
    ShapeId.Q4: '''
        MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: $genre})
        MATCH (t:Track)-[:HAS_GENRE]->(g)
        RETURN DISTINCT t.title AS title, g.name AS genre
        ORDER BY title
    ''',
```
The `Q4` template is the **hierarchical traversal shape**.
`[:SUBCLASS_OF*0..]` means: "traverse `SUBCLASS_OF` edges zero or more times."
Zero hops means the root genre itself is included.
One or more hops reaches Bebop, Swing, Cool Jazz, and any deeper sub-genres.
The `DISTINCT` keyword prevents duplicate track rows when a track is tagged
with multiple genres that all match the traversal.

### 3c. Display Table

#### Line-by-line explanation

```python
from tabulate import tabulate
```
Imports the `tabulate` function from the `tabulate` library.

```python
first_clause = next(ln.strip() for ln in cypher.strip().splitlines() if ln.strip())
```
`cypher.strip()` removes leading/trailing whitespace from the template string.
`.splitlines()` splits it into a list of individual lines.
The generator expression `ln.strip() for ln in ... if ln.strip()` strips
whitespace from each line and skips blank lines.
`next(...)` returns the **first non-empty line** — which is always the `MATCH`
clause, giving a useful one-line summary of the template.

```python
print(tabulate(rows, headers=['Shape', 'First Clause (truncated to 80 chars)'], tablefmt='github'))
```
`tablefmt='github'` renders the table using GitHub-Flavored Markdown `|` pipes,
which displays cleanly in both GitHub READMEs and Jupyter notebooks.

### Complete code block

```python
from enum import Enum, auto

class ShapeId(Enum):
    Q1  = auto()  # Tracks by instrument
    Q2  = auto()  # Tracks by artist
    Q3  = auto()  # Tracks on an album
    Q4  = auto()  # Tracks by genre — HIERARCHICAL ([:SUBCLASS_OF*0..])
    Q5  = auto()  # Albums by artist (discography)
    Q6  = auto()  # Artists on an album
    Q7  = auto()  # Artists who play a specific instrument
    Q8  = auto()  # Artists associated with a genre
    Q9  = auto()  # Tracks in a BPM range
    Q10 = auto()  # Tracks in a musical key
    Q11 = auto()  # Tracks in a year range
    Q12 = auto()  # Collaborating artists (co-appeared on a track)
    Q13 = auto()  # Similar artists (shared genre membership)
    Q14 = auto()  # Genre hierarchy — list subgenres of a genre
    Q15 = auto()  # Most popular tracks by play count

TEMPLATES = {
    ShapeId.Q1: '''
        MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {name: $instrument})
        RETURN t.title AS title, t.year AS year ORDER BY t.year
    ''',
    ShapeId.Q2: '''
        MATCH (a:Artist {name: $artist})-[:PERFORMED]->(t:Track)
        RETURN t.title AS title, t.year AS year ORDER BY t.year
    ''',
    ShapeId.Q3: '''
        MATCH (al:Album {name: $album})-[:CONTAINS]->(t:Track)
        RETURN t.title AS title, t.track_number AS track_number ORDER BY t.track_number
    ''',
    ShapeId.Q4: '''
        MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: $genre})
        MATCH (t:Track)-[:HAS_GENRE]->(g)
        RETURN DISTINCT t.title AS title, g.name AS genre ORDER BY title
    ''',
    ShapeId.Q5: '''
        MATCH (a:Artist {name: $artist})-[:RELEASED]->(al:Album)
        RETURN al.name AS album, al.year AS year ORDER BY al.year
    ''',
    ShapeId.Q6: '''
        MATCH (al:Album {name: $album})<-[:RELEASED]-(a:Artist)
        RETURN a.name AS artist ORDER BY artist
    ''',
    ShapeId.Q7: '''
        MATCH (a:Artist)-[:PLAYS]->(:Instrument {name: $instrument})
        RETURN a.name AS artist ORDER BY artist
    ''',
    ShapeId.Q8: '''
        MATCH (a:Artist)-[:ASSOCIATED_WITH]->(:Genre {name: $genre})
        RETURN a.name AS artist ORDER BY artist
    ''',
    ShapeId.Q9: '''
        MATCH (t:Track)
        WHERE t.bpm >= $min_bpm AND t.bpm <= $max_bpm
        RETURN t.title AS title, t.bpm AS bpm ORDER BY t.bpm
    ''',
    ShapeId.Q10: '''
        MATCH (t:Track {key: $key})
        RETURN t.title AS title, t.key AS key ORDER BY title
    ''',
    ShapeId.Q11: '''
        MATCH (t:Track)
        WHERE t.year >= $start_year AND t.year <= $end_year
        RETURN t.title AS title, t.year AS year ORDER BY t.year
    ''',
    ShapeId.Q12: '''
        MATCH (a1:Artist {name: $artist})-[:PERFORMED]->(t:Track)<-[:PERFORMED]-(a2:Artist)
        WHERE a1 <> a2
        RETURN DISTINCT a2.name AS collaborator, t.title AS track ORDER BY collaborator
    ''',
    ShapeId.Q13: '''
        MATCH (a1:Artist {name: $artist})-[:ASSOCIATED_WITH]->(g:Genre)<-[:ASSOCIATED_WITH]-(a2:Artist)
        WHERE a1 <> a2
        RETURN DISTINCT a2.name AS similar_artist, g.name AS shared_genre ORDER BY similar_artist
    ''',
    ShapeId.Q14: '''
        MATCH (sub:Genre)-[:SUBCLASS_OF*1..]->(root:Genre {name: $genre})
        RETURN sub.name AS subgenre ORDER BY subgenre
    ''',
    ShapeId.Q15: '''
        MATCH (t:Track)
        RETURN t.title AS title, t.play_count AS plays
        ORDER BY t.play_count DESC LIMIT $limit
    ''',
}

print(f'Loaded {len(TEMPLATES)} Cypher templates.')

# ── Display table ─────────────────────────────────────────────────────────────
from tabulate import tabulate

rows = []
for sid, cypher in TEMPLATES.items():
    first_clause = next(ln.strip() for ln in cypher.strip().splitlines() if ln.strip())
    rows.append([sid.name, first_clause[:80]])

print(tabulate(rows, headers=['Shape', 'First Clause (truncated to 80 chars)'], tablefmt='github'))
```

---

## 4. Stage 3 — End-to-End Happy Path (Hierarchical Traversal)

### 4a. Pipeline Functions

#### Line-by-line explanation

```python
import re
```
The `re` module provides compiled regular expressions.
Compiling a pattern with `re.compile()` once and reusing it is faster than
calling `re.search(pattern, string)` on every invocation, because the pattern
is only compiled once into a finite state machine.

```python
class UnsupportedQueryError(Exception):
    def __init__(self, question):
        supported = ', '.join(s.name for s in ShapeId)
        super().__init__(
            f"No matching shape for: '{question}'. "
            f"Supported shapes: {supported}"
        )
        self.question = question
```
`UnsupportedQueryError` inherits from `Exception`, making it a custom
checked exception.
`', '.join(s.name for s in ShapeId)` iterates over all 15 `ShapeId` members
and joins their names into a comma-separated string.
`super().__init__(...)` passes the error message to the base `Exception` class,
so it appears when the exception is printed or logged.
`self.question = question` preserves the original user question on the
exception object so a caller can access it programmatically (e.g., an
autograder that wants to log which questions caused failures).

```python
INTENT_PATTERNS = [
    (re.compile(r'find.+by instrument|played on.+(saxophone|trumpet|piano|guitar)', re.I), ShapeId.Q1),
    ...
]
```
`INTENT_PATTERNS` is a list of `(compiled_pattern, ShapeId)` tuples.
`re.compile(r'...', re.I)` compiles the regex with `re.IGNORECASE` so
`"Find Jazz tracks"` and `"find jazz tracks"` both match.
`r'...'` is a raw string — backslashes are literal characters, not escape
sequences, which is essential for regex metacharacters like `\b` (word boundary)
and `\d` (digit).
Patterns are ordered from most specific to most general; `detect_shape` returns
the **first** match, so order matters.

```python
def detect_shape(question):
    for pattern, shape_id in INTENT_PATTERNS:
        if pattern.search(question):
            return shape_id
    return None
```
`pattern.search(question)` looks for a match **anywhere** in the question
string (not just at the start — that would require `pattern.match()`).
`return shape_id` exits the function immediately on the first match.
`return None` at the end is explicit — the caller must check for `None`
and must **not** silently treat it as "empty results."

```python
def extract_slots(question, shape_id):
    slots = {}
    if shape_id == ShapeId.Q4:
        m = re.search(r'(Jazz|Rock|Blues|Bebop|Swing|Hip.?Hop|Classical|Country)', question, re.I)
        if m:
            slots['genre'] = m.group(1).capitalize()
```
`slots = {}` initializes an empty dict — only the keys relevant to the
detected shape will be populated.
`re.search(r'...')` finds the genre name anywhere in the question.
`m.group(1)` returns the first capturing group — the genre name.
`.capitalize()` converts the first character to uppercase and the rest to
lowercase: `'jazz'` → `'Jazz'`, `'JAZZ'` → `'Jazz'`.
This **slot canonicalization** step is critical — the KG stores genres in
Title Case and Neo4j property matching is case-sensitive.

```python
def compile_to_cypher(shape_id):
    return TEMPLATES[shape_id]
```
Dict lookup is O(1). The template string is returned with `$params` intact —
the slot values are never embedded here.

```python
def run_query(question):
    shape_id = detect_shape(question)
    if shape_id is None:
        raise UnsupportedQueryError(question)
    ...
    with driver.session() as session:
        records = [dict(r) for r in session.run(cypher, **slots)]
    return records
```
`raise UnsupportedQueryError(question)` is the **fail-loud** pattern.
The function never returns `[]` for an unrecognized question — that would
destroy the diagnostic signal (see Stage 5).
`session.run(cypher, **slots)` passes the Cypher template and the parameters
dict as **two separate arguments**. The driver escapes the parameter values
before sending them to Neo4j.
`**slots` unpacks `{'genre': 'Jazz'}` into `session.run(cypher, genre='Jazz')`.
`[dict(r) for r in ...]` converts each Neo4j `Record` object into a plain
Python dict for easy use outside the session context.

### 4b. Running the Happy Path

#### Line-by-line explanation

```python
results = run_query('Find Jazz tracks')
```
Drives all four pipeline stages:
1. `detect_shape` → `ShapeId.Q4`
2. `extract_slots` → `{'genre': 'Jazz'}`
3. `compile_to_cypher` → Q4 template with `[:SUBCLASS_OF*0..]`
4. `session.run` → records tagged `:Bebop`, `:Swing`, etc.

### 4c. Hierarchical vs Flat Comparison

#### Line-by-line explanation

```python
cypher_hier = '''
    MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: $genre})
    MATCH (t:Track)-[:HAS_GENRE]->(g)
    RETURN DISTINCT t.title AS title, g.name AS genre ORDER BY title
'''
cypher_flat = '''
    MATCH (t:Track)-[:HAS_GENRE]->(:Genre {name: $genre})
    RETURN DISTINCT t.title AS title ORDER BY title
'''
```
The hierarchical query uses two `MATCH` clauses.
The first clause finds all genres `g` that are sub-genres of the root (zero or
more hops via `SUBCLASS_OF`).
The second clause finds tracks tagged with any of those genres.
The flat query uses a single `MATCH` and only finds tracks tagged *directly*
with the root genre — sub-genre tracks are silently absent.

```python
subgenres = {row['genre'] for row in hier if row.get('genre') != 'Jazz'}
```
A set comprehension that collects all genre names from the hierarchical results
that are *not* the root genre — these are the sub-genres that the flat query
missed.

### Complete code block

```python
import re

class UnsupportedQueryError(Exception):
    def __init__(self, question):
        supported = ', '.join(s.name for s in ShapeId)
        super().__init__(
            f"No matching shape for: '{question}'. "
            f"Supported shapes: {supported}"
        )
        self.question = question

INTENT_PATTERNS = [
    (re.compile(r'find.+by instrument|played on.+(saxophone|trumpet|piano|guitar)', re.I), ShapeId.Q1),
    (re.compile(r'tracks.{0,20}(by|from)\s+[A-Z]', re.I),                                ShapeId.Q2),
    (re.compile(r'on album|album.{0,20}tracks', re.I),                                    ShapeId.Q3),
    (re.compile(r'(find|list|show).{0,30}(jazz|rock|blues|bebop|swing|hip.?hop|classical|country).{0,20}tracks', re.I), ShapeId.Q4),
    (re.compile(r'discography|albums.{0,20}(by|from)', re.I),                             ShapeId.Q5),
    (re.compile(r'artists.{0,20}on album|who (played|performed).{0,20}on', re.I),         ShapeId.Q6),
    (re.compile(r'artists.{0,20}(play|plays|who play).+(saxophone|trumpet|piano|guitar|bass|drums)', re.I), ShapeId.Q7),
    (re.compile(r'artists.{0,20}(associated|genre)', re.I),                               ShapeId.Q8),
    (re.compile(r'bpm.{0,20}between|tempo.{0,20}range', re.I),                           ShapeId.Q9),
    (re.compile(r'key of|in (the key|[A-G] (major|minor))', re.I),                       ShapeId.Q10),
    (re.compile(r'(from|released|between).{0,10}\d{4}.{0,10}(and|to).{0,10}\d{4}', re.I), ShapeId.Q11),
    (re.compile(r'collaborat\w+|appeared.{0,20}same track', re.I),                       ShapeId.Q12),
    (re.compile(r'similar artists?|artists?.{0,20}like', re.I),                          ShapeId.Q13),
    (re.compile(r'subgenres?|genre hierarchy|sub.?genres? of', re.I),                    ShapeId.Q14),
    (re.compile(r'most popular|top \d+|highest play', re.I),                             ShapeId.Q15),
]

def detect_shape(question):
    for pattern, shape_id in INTENT_PATTERNS:
        if pattern.search(question):
            return shape_id
    return None

def extract_slots(question, shape_id):
    slots = {}
    if shape_id == ShapeId.Q4:
        m = re.search(r'(Jazz|Rock|Blues|Bebop|Swing|Hip.?Hop|Classical|Country)', question, re.I)
        if m:
            slots['genre'] = m.group(1).capitalize()
    elif shape_id == ShapeId.Q1:
        m = re.search(r'(saxophone|trumpet|piano|guitar|bass|drums|violin)', question, re.I)
        if m:
            slots['instrument'] = m.group(1).capitalize()
    elif shape_id == ShapeId.Q2:
        m = re.search(r'(?:by|from)\s+([A-Z][a-zA-Z\s]+)', question)
        if m:
            slots['artist'] = m.group(1).strip()
    elif shape_id == ShapeId.Q15:
        m = re.search(r'top\s+(\d+)', question, re.I)
        slots['limit'] = int(m.group(1)) if m else 10
    return slots

def compile_to_cypher(shape_id):
    return TEMPLATES[shape_id]

def run_query(question):
    shape_id = detect_shape(question)
    if shape_id is None:
        raise UnsupportedQueryError(question)
    print(f'  detect_shape   → {shape_id}')
    slots  = extract_slots(question, shape_id)
    print(f'  extract_slots  → {slots}')
    cypher = compile_to_cypher(shape_id)
    print(f'  compile        → template retrieved (parameterized)')
    with driver.session() as session:
        records = [dict(r) for r in session.run(cypher, **slots)]
    print(f'  session.run    → {len(records)} records returned')
    return records

# ── Happy path ────────────────────────────────────────────────────────────────
print('=' * 58)
print('Simulating: python cli.py "Find Jazz tracks"')
print('=' * 58)

try:
    results = run_query('Find Jazz tracks')
    print(f'\n{len(results)} tracks returned:')
    for row in results[:10]:
        print(f"  {row.get('title', '?'):<35} genre: {row.get('genre', '?')}")
    if len(results) > 10:
        print(f'  ... and {len(results) - 10} more.')
except UnsupportedQueryError as e:
    print(f'ERROR: {e}')

# ── Hierarchical vs flat ──────────────────────────────────────────────────────
cypher_hier = '''
    MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: $genre})
    MATCH (t:Track)-[:HAS_GENRE]->(g)
    RETURN DISTINCT t.title AS title, g.name AS genre ORDER BY title
'''
cypher_flat = '''
    MATCH (t:Track)-[:HAS_GENRE]->(:Genre {name: $genre})
    RETURN DISTINCT t.title AS title ORDER BY title
'''

with driver.session() as session:
    hier = [dict(r) for r in session.run(cypher_hier, genre='Jazz')]
    flat = [dict(r) for r in session.run(cypher_flat, genre='Jazz')]

print(f'With    [:SUBCLASS_OF*0..]: {len(hier):4d} tracks')
print(f'Without traversal (flat):   {len(flat):4d} tracks')
print(f'Silently dropped by flat:   {len(hier) - len(flat):4d} tracks')
subgenres = {row['genre'] for row in hier if row.get('genre') != 'Jazz'}
print(f'Sub-genres present in KG: {sorted(subgenres)}')
```

---

## 5. Stage 4 — Parameterization & Cypher Injection

### Line-by-line explanation

```python
correct_cypher = 'MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {name: $instrument}) RETURN t.title'
correct_params = {'instrument': 'Piano'}
```
`correct_cypher` contains a `$instrument` placeholder — not a Python variable.
`correct_params` is a separate dict.
When you call `session.run(correct_cypher, **correct_params)`, the driver
serializes these as two distinct fields in the Bolt protocol message.
The Neo4j server substitutes `$instrument` with the value `'Piano'` inside
its own query engine — the string is never concatenated.

```python
wrong_cypher = f"MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {{name: '{instrument_value}'}}) RETURN t.title"
```
The f-string prefix `f"..."` causes Python to evaluate `{instrument_value}`
and splice its string representation directly into the query before the string
ever leaves Python.
The `{{` and `}}` are escaped braces — in an f-string, double braces produce
a literal `{` or `}` character.
This is the **dangerous pattern**: the variable is embedded in the Cypher
string.

```python
malicious = "'; MATCH (n) DETACH DELETE n; //"
```
Breaking this down:
- `'` closes the string literal `{name: '...'}`
- `;` ends the first Cypher statement
- `MATCH (n) DETACH DELETE n` matches every node and detaches (removes) all
  relationships before deleting the node — a full wipe of the database
- `;` ends the injected statement
- `//` is a Cypher line comment — anything after it is ignored, which
  neutralizes whatever remains of the original query

```python
injected = f"MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {{name: '{malicious}'}}) RETURN t.title"
```
With the malicious value interpolated, the full string becomes syntactically
valid Cypher with two statements: the original query and the destructive wipe.
A real Neo4j instance would execute both.

```python
safe_params = {'instrument': malicious}
```
When passed as a parameter, the malicious string is treated as a plain
data value. The Neo4j driver escapes the `'` characters inside the value.
The server's query engine sees `$instrument` as a string parameter slot and
fills it with the escaped value — the injected Cypher syntax is never parsed
as Cypher.

### Complete code block

```python
# CORRECT: parameterized template
correct_cypher = 'MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {name: $instrument}) RETURN t.title'
correct_params = {'instrument': 'Piano'}

print('CORRECT (parameterized):')
print(f'  Cypher : {correct_cypher}')
print(f'  Params : {correct_params}')
print('  Driver sends these as two independent objects to Neo4j.')
print('  The server substitutes $instrument safely.\n')

# WRONG: f-string interpolation
instrument_value = 'Piano'
wrong_cypher = f"MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {{name: '{instrument_value}'}}) RETURN t.title"

print('WRONG (f-string interpolation):')
print(f'  Cypher : {wrong_cypher}')
print('  The value is concatenated into the string — injection is now possible.\n')

# Cypher injection demonstration
malicious = "'; MATCH (n) DETACH DELETE n; //"

print(f'Malicious slot value : {malicious!r}\n')

injected = f"MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {{name: '{malicious}'}}) RETURN t.title"
print('VULNERABLE f-string result:')
print(f'  {injected}')
print('  The DETACH DELETE clause is syntactically valid — database would be wiped.\n')

safe_cypher = 'MATCH (t:Track)-[:USES_INSTRUMENT]->(:Instrument {name: $instrument}) RETURN t.title'
safe_params = {'instrument': malicious}

print('SAFE parameterized result:')
print(f'  Cypher : {safe_cypher}')
print(f'  Params : {safe_params}')
print('  Driver escapes the value. No injection. Database untouched.')
```

---

## 6. Stage 5 — Fail-Loud `UnsupportedQueryError`

### Line-by-line explanation

```python
def run_query_silent(question):
    shape_id = detect_shape(question)
    if shape_id is None:
        return []
```
This is the **anti-pattern** — do not use it.
`return []` looks harmless: it returns an empty list when no shape matches.
But a caller receiving `[]` cannot distinguish between:
- "The query was dispatched; the KG has no matching data" (empty result set)
- "The question was off-template; the pipeline never dispatched at all"

Both cases return `[]`, but they have completely different diagnostic meanings.
Autograders, monitoring systems, and human debuggers all need this distinction.

```python
try:
    run_query(off_template)
except UnsupportedQueryError as e:
    print(f'UnsupportedQueryError caught:\n  {e}\n')
```
The correct pattern: `run_query` raises `UnsupportedQueryError` the moment
`detect_shape` returns `None`.
The caller catches it and receives a message that:
1. Names the unsupported question
2. Enumerates all 15 supported shapes
This is **actionable diagnostic information** — a human can rephrase, and an
autograder can record which shape was missing.

### Complete code block

```python
# Anti-pattern — shown for contrast only
def run_query_silent(question):
    shape_id = detect_shape(question)
    if shape_id is None:
        return []   # silent failure — destroys diagnostic signal
    cypher = compile_to_cypher(shape_id)
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher)]

off_template = 'What is the meaning of life?'
print(f'Off-template question: "{off_template}"\n')

print('── CORRECT: fail loud ─────────────────────────────────────────')
try:
    run_query(off_template)
except UnsupportedQueryError as e:
    print(f'UnsupportedQueryError caught:\n  {e}\n')

print('── WRONG: silent failure ──────────────────────────────────────')
result = run_query_silent(off_template)
print(f'Result : {result}')
print('Caller receives [] — cannot tell whether the pipeline ran or was skipped.')
```

---

## 7. Stage 6 — Tier 3 Swap-in & the Allowlist

### 7a. UnsupportedCypherError, Allowlist & Mock Cache

#### Line-by-line explanation

```python
class UnsupportedCypherError(Exception):
    def __init__(self, cypher, reason=''):
        super().__init__(f'UnsupportedCypherError: {reason}\nQuery: {cypher}')
        self.cypher = cypher
```
`UnsupportedCypherError` is distinct from `UnsupportedQueryError`.
- `UnsupportedQueryError` fires at the **intent layer** (Stage 1): the user's
  question did not match any shape.
- `UnsupportedCypherError` fires at the **Cypher layer** (Stage 3): the LLM
  generated a query that violates the allowlist.
`self.cypher = cypher` preserves the rejected query for audit logging.

```python
FORBIDDEN_VERBS = {'DELETE', 'DETACH', 'CREATE', 'MERGE', 'SET', 'REMOVE', 'DROP'}
```
A Python set — O(1) membership testing.
These are Cypher DML verbs that mutate or destroy data.
`MATCH`, `RETURN`, `WHERE`, `WITH`, `ORDER BY`, and `LIMIT` are implicitly
allowed (they are read-only).

```python
def validate_query_shape(cypher):
    upper = cypher.upper()
    for verb in FORBIDDEN_VERBS:
        if re.search(rf'\b{verb}\b', upper):
            raise UnsupportedCypherError(cypher, reason=f'Forbidden verb detected: {verb}')
    if not upper.strip().startswith('MATCH'):
        raise UnsupportedCypherError(cypher, reason='Query must begin with MATCH.')
```
`cypher.upper()` normalizes to uppercase — `delete` and `DELETE` are both caught.
`rf'\b{verb}\b'` is an f-string inside a raw string.
`\b` is a regex word boundary — it prevents `CREATED_AT` from triggering the
`CREATE` check.
The `{verb}` part is substituted by Python before the regex engine sees the
pattern, so for `verb='DELETE'` the pattern becomes `r'\bDELETE\b'`.
The `startswith('MATCH')` guard ensures the query is a read-only retrieval;
all valid SPARQL-style KG queries begin with `MATCH`.

```python
MOCK_LLM_CACHE = {
    'Find Jazz tracks': "MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: 'Jazz'}) ...",
    'delete all tracks': 'MATCH (t:Track) DELETE t',     # adversarial
    'remove everything': 'MATCH (n) DETACH DELETE n',    # adversarial
}
```
The mock cache replaces a live `GraphCypherQAChain` call.
In production, `GraphCypherQAChain.run(question)` calls the LLM API and returns
generated Cypher.
For offline grading, we look up the cached response instead, making the
autograder deterministic and API-key-free.
The adversarial entries simulate a model that has been prompted maliciously or
that hallucinated destructive output.

```python
rejection_counter = {'count': 0}
```
A mutable dict rather than a plain `int`. In Python, assigning to a bare
`int` variable inside a nested function would shadow the outer variable
(unless `nonlocal` is used). A dict is mutable in-place, so
`rejection_counter['count'] += 1` updates the same object without needing
`nonlocal`.

### 7b. Prompt Builder & Chain Runner

#### Line-by-line explanation

```python
SCHEMA_PREAMBLE = (
    'You are a Cypher expert for a Music Knowledge Graph.\n'
    ...
)
```
Adjacent string literals in Python are automatically concatenated at compile
time. This is cleaner than triple-quotes for multi-line strings that include
escape sequences.
The preamble establishes the KG schema and the read-only contract for the LLM
— it tells the model what node types, relationship types, and verbs are allowed.

```python
FEW_SHOTS = (
    '\nExample 1:\n'
    'Q: Find Jazz tracks\n'
    "A: MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: 'Jazz'})\n"
    ...
)
```
Few-shot examples guide the LLM by showing input–output pairs.
They establish the expected Cypher format, including the use of
`[:SUBCLASS_OF*0..]` for hierarchical traversal.
The more representative and correct the examples, the less likely the LLM
is to hallucinate invalid Cypher.

```python
def build_prompt(question):
    return f'{SCHEMA_PREAMBLE}\n{FEW_SHOTS}\nQ: {question}\nA:'
```
The `A:` at the end is the prompt template's "completion prefix" — it signals
to the LLM that the next tokens should be the Cypher answer.

```python
def run_chain(question):
    prompt    = build_prompt(question)
    generated = MOCK_LLM_CACHE.get(question, '')
```
`dict.get(key, default)` returns `default` if the key is absent rather than
raising `KeyError`. An empty string signals "no cached response."

```python
    try:
        validate_query_shape(generated)
    except UnsupportedCypherError as e:
        rejection_counter['count'] += 1
        cnt = rejection_counter['count']
        return {'records': [], 'rejected': True, 'reason': str(e)}
```
The allowlist check runs **before** `session.run` — the database is never
touched by a rejected query.
`rejection_counter['count'] += 1` atomically increments the counter.
`'rejected': True` is the field the autograder reads to distinguish
"blocked by allowlist" from "allowed but returned wrong results."

### Complete code block

```python
import re

class UnsupportedCypherError(Exception):
    def __init__(self, cypher, reason=''):
        super().__init__(f'UnsupportedCypherError: {reason}\nQuery: {cypher}')
        self.cypher = cypher

FORBIDDEN_VERBS = {'DELETE', 'DETACH', 'CREATE', 'MERGE', 'SET', 'REMOVE', 'DROP'}

def validate_query_shape(cypher):
    upper = cypher.upper()
    for verb in FORBIDDEN_VERBS:
        if re.search(rf'\b{verb}\b', upper):
            raise UnsupportedCypherError(cypher, reason=f'Forbidden verb detected: {verb}')
    if not upper.strip().startswith('MATCH'):
        raise UnsupportedCypherError(cypher, reason='Query must begin with MATCH.')

MOCK_LLM_CACHE = {
    'Find Jazz tracks'          : "MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: 'Jazz'}) MATCH (t:Track)-[:HAS_GENRE]->(g) RETURN DISTINCT t.title AS title",
    'List tracks by Miles Davis': "MATCH (a:Artist {name: 'Miles Davis'})-[:PERFORMED]->(t:Track) RETURN t.title AS title",
    'delete all tracks'         : 'MATCH (t:Track) DELETE t',
    'remove everything'         : 'MATCH (n) DETACH DELETE n',
}

rejection_counter = {'count': 0}

# ── Prompt builder & chain ────────────────────────────────────────────────────

SCHEMA_PREAMBLE = (
    'You are a Cypher expert for a Music Knowledge Graph.\n'
    'Nodes: Track, Artist, Album, Genre, Instrument\n'
    'Relationships: PERFORMED, RELEASED, CONTAINS, HAS_GENRE, '
    'USES_INSTRUMENT, PLAYS, ASSOCIATED_WITH, SUBCLASS_OF\n'
    'Rules: Only MATCH...RETURN. Never DELETE, CREATE, MERGE, SET, or REMOVE.'
)

FEW_SHOTS = (
    '\nExample 1:\n'
    'Q: Find Jazz tracks\n'
    "A: MATCH (g:Genre)-[:SUBCLASS_OF*0..]->(root:Genre {name: 'Jazz'})\n"
    '   MATCH (t:Track)-[:HAS_GENRE]->(g)\n'
    '   RETURN DISTINCT t.title AS title, g.name AS genre\n'
    '\nExample 2:\n'
    'Q: List albums by Miles Davis\n'
    "A: MATCH (a:Artist {name: 'Miles Davis'})-[:RELEASED]->(al:Album)\n"
    '   RETURN al.name AS album, al.year AS year ORDER BY al.year\n'
)

def build_prompt(question):
    return f'{SCHEMA_PREAMBLE}\n{FEW_SHOTS}\nQ: {question}\nA:'

def run_chain(question):
    prompt    = build_prompt(question)
    generated = MOCK_LLM_CACHE.get(question, '')
    if not generated:
        return {'records': [], 'rejected': False, 'reason': 'no cached response'}

    print(f'  [LLM generated] : {generated}')

    try:
        validate_query_shape(generated)
    except UnsupportedCypherError as e:
        rejection_counter['count'] += 1
        cnt = rejection_counter['count']
        print(f'  [REJECTED]      : count={cnt}')
        return {'records': [], 'rejected': True, 'reason': str(e)}

    with driver.session() as session:
        records = [dict(r) for r in session.run(generated)]

    print(f'  [session.run]   : {len(records)} records returned')
    return {'records': records, 'rejected': False, 'reason': ''}

# ── Adversarial test suite ────────────────────────────────────────────────────

tests = [
    ('Find Jazz tracks',  False),
    ('delete all tracks', True),
    ('remove everything', True),
]

print('Tier 3 Adversarial Mutation Demo\n' + '=' * 52)

for question, expect_rejected in tests:
    print(f'\nQuestion: "{question}"')
    result = run_chain(question)
    status   = 'REJECTED' if result['rejected'] else 'ACCEPTED'
    expected = 'REJECTED' if expect_rejected    else 'ACCEPTED'
    ok       = 'PASS' if result['rejected'] == expect_rejected else 'FAIL'
    print(f'  Status   : {status}  (expected: {expected})  [{ok}]')
    if result['records']:
        print(f'  Records  : {len(result["records"])}')

print(f'\nTotal allowlist rejections: {rejection_counter["count"]}')
print('Database untouched — all destructive queries blocked before session.run.')
```

---

## 8. Stage 7 — Pivot to the Recipe Domain

### Line-by-line explanation

```python
class RecipeShapeId(Enum):
    Q1  = auto()  # Recipes that use a specific ingredient (direct match)
    Q2  = auto()  # Recipes by cuisine — HIERARCHICAL over :Cuisine
    ...
```
Identical structure to `ShapeId`. The bounded recipe schema (Recipe, Ingredient,
Cuisine, Tag, DietaryLabel) admits the same 15-shape enumeration.
The shape numbering is a naming convention — `Q2` in the recipe domain is the
cuisine hierarchy query, not the "tracks by artist" query from the music domain.

```python
    RecipeShapeId.Q2: '''
        MATCH (c:Cuisine)-[:SUBCLASS_OF*0..]->(root:Cuisine {name: $cuisine})
        MATCH (r:Recipe)-[:BELONGS_TO_CUISINE]->(c)
        RETURN DISTINCT r.name AS recipe, c.name AS cuisine ORDER BY recipe
    ''',
```
Structurally identical to Music `Q4`. The only changes are:
- `:Genre` → `:Cuisine`
- `[:HAS_GENRE]` → `[:BELONGS_TO_CUISINE]`
- `$genre` → `$cuisine`
The hierarchical traversal pattern `[:SUBCLASS_OF*0..]` is the same.

```python
    RecipeShapeId.Q4: '''
        MATCH (i:Ingredient)-[:SUBCLASS_OF*0..]->(root:Ingredient {name: $ingredient})
        MATCH (r:Recipe)-[:USES_INGREDIENT]->(i)
        ...
    ''',
```
A second hierarchical traversal, this time over `:Ingredient`.
This catches `Pasta → Spaghetti`, `Pasta → Rigatoni`, etc., when a user asks
for "recipes using pasta."

```python
def canonicalize(raw):
    return raw.strip().title()
```
`str.strip()` removes leading and trailing whitespace.
`str.title()` capitalizes the first letter of each word and lowercases the
rest: `'ITALIAN'` → `'Italian'`, `'italian'` → `'Italian'`.
This is the slot canonicalization that the assignment requires — failing to
canonicalize means your Cypher queries return zero rows even when the data
exists, because Neo4j property matching is case-sensitive.

### The Four Failure Diagnosis Stages

When a query returns zero rows or wrong results, trace the failure through each stage:

| Stage | What to check | Common bug |
|-------|--------------|------------|
| 1 — Intent | `detect_shape(question)` | Returns `None` or wrong `ShapeId` |
| 2 — Slots | `extract_slots(question, shape_id)` | `'italian'` instead of `'Italian'` |
| 3 — Compile | `compile_to_cypher(shape_id)` | `Q1` (flat) used instead of `Q2` (hierarchical) |
| 4 — Driver | `session.run(cypher, **slots)` | Parameter name mismatch (`$cuisine` vs `$genre`) |

Your `learner_notes.md` must trace one real failure through all four stages,
naming which stage produced the wrong intermediate value.

### Complete code block

```python
from enum import Enum, auto

class RecipeShapeId(Enum):
    Q1  = auto()  # Recipes that use a specific ingredient (direct match)
    Q2  = auto()  # Recipes by cuisine — HIERARCHICAL over :Cuisine
    Q3  = auto()  # Recipes by dietary label (vegan, gluten-free, etc.)
    Q4  = auto()  # Recipes by ingredient category — HIERARCHICAL over :Ingredient
    Q5  = auto()  # Ingredients listed in a specific recipe
    Q6  = auto()  # Recipes within a cooking-time range
    Q7  = auto()  # Recipes tagged with a keyword (comfort-food, summer, etc.)
    Q8  = auto()  # Cuisines that use a specific ingredient
    Q9  = auto()  # Recipes within a calorie range
    Q10 = auto()  # Recipes that share ingredients with a given recipe
    Q11 = auto()  # Most popular recipes by user rating
    Q12 = auto()  # Recipes by difficulty level
    Q13 = auto()  # Recipes by chef or author
    Q14 = auto()  # Cuisine hierarchy — subgenres of a cuisine
    Q15 = auto()  # Ingredient-substitution recipes

RECIPE_TEMPLATES = {
    RecipeShapeId.Q2: '''
        MATCH (c:Cuisine)-[:SUBCLASS_OF*0..]->(root:Cuisine {name: $cuisine})
        MATCH (r:Recipe)-[:BELONGS_TO_CUISINE]->(c)
        RETURN DISTINCT r.name AS recipe, c.name AS cuisine ORDER BY recipe
    ''',
    RecipeShapeId.Q4: '''
        MATCH (i:Ingredient)-[:SUBCLASS_OF*0..]->(root:Ingredient {name: $ingredient})
        MATCH (r:Recipe)-[:USES_INGREDIENT]->(i)
        RETURN DISTINCT r.name AS recipe, i.name AS ingredient ORDER BY recipe
    ''',
    RecipeShapeId.Q1: '''
        MATCH (r:Recipe)-[:USES_INGREDIENT]->(:Ingredient {name: $ingredient})
        RETURN r.name AS recipe, r.cooking_time AS cooking_time ORDER BY r.cooking_time
    ''',
    RecipeShapeId.Q11: '''
        MATCH (r:Recipe)
        RETURN r.name AS recipe, r.rating AS rating
        ORDER BY r.rating DESC LIMIT $limit
    ''',
}

print(f'Recipe domain: {len(RecipeShapeId)} shapes, {len(RECIPE_TEMPLATES)} templates shown.')

# ── Slot canonicalization ─────────────────────────────────────────────────────
def canonicalize(raw):
    return raw.strip().title()   # 'italian' → 'Italian'

print('\nCanonicalization examples:')
for raw in ['italian', 'ITALIAN', 'Italian', ' french ', 'CHINESE']:
    print(f'  {raw!r:15} → {canonicalize(raw)!r}')

# ── Failure diagnosis framework ───────────────────────────────────────────────
print('\n── Failure Diagnosis Framework (for learner_notes.md) ──────────')
print('\nStage 1 — Intent:')
print('  Probe: detect_shape(question)')
print('\nStage 2 — Slots:')
print('  Probe: extract_slots(question, shape_id)')
print("  Common bug: 'italian' extracted instead of 'Italian'")
print('\nStage 3 — Compile:')
print('  Probe: compile_to_cypher(shape_id)')
print('  Common bug: Q1 (flat) used instead of Q2 (hierarchical)')
print('\nStage 4 — Driver:')
print('  Probe: session.run(cypher, **slots)')
print('  Common bug: $cuisine vs $genre parameter name mismatch')
print('\nTier 3: autograder reads challenge/mock_llm_cache.json — no live LLM needed.')
print('Set USE_MY_LINKER=1 to swap in your own linker.')
```

---

## 9. Cleanup

### Line-by-line explanation

```python
driver.close()
```
Closes all open connections in the driver's internal pool and terminates the
background threads the driver uses for connection health checks.
Always call this before the Python process exits to avoid resource leaks.

```python
result = subprocess.run(
    ['docker', 'compose', 'stop', 'music-neo4j'],
    capture_output=True,
    text=True
)
```
`docker compose stop` sends `SIGTERM` to the container process for a graceful
shutdown — Neo4j flushes its write-ahead log before exiting, preserving data
integrity.
`docker compose stop` differs from `docker compose down`:
- `stop` keeps the container layer (you can restart quickly with `docker compose start`)
- `down` removes the container layer (next `up` recreates it from the image)

```python
if result.returncode == 0:
    print('music-neo4j container stopped successfully.')
else:
    print(f'Warning — Docker stop returned non-zero exit code:\n{result.stderr.strip()}')
```
`returncode == 0` means the command exited successfully.
Any other value means an error occurred; `result.stderr` contains the Docker
error message.

### Complete code block

```python
import subprocess

# Close the Neo4j driver — releases all sockets and background threads
driver.close()
print('Neo4j driver closed — all connections released.')

# Stop the Docker container gracefully (data is preserved on the Docker volume)
result = subprocess.run(
    ['docker', 'compose', 'stop', 'music-neo4j'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print('music-neo4j container stopped successfully.')
else:
    print(f'Warning — Docker stop returned non-zero exit code:\n{result.stderr.strip()}')

print('Lab teardown complete. All resources released.')
```

---

## Summary

| Stage | Key concept | Code pattern |
|-------|------------|--------------|
| 1 | Bounded templates | `ShapeId` enum + `TEMPLATES` dict |
| 2 | Hierarchical traversal | `[:SUBCLASS_OF*0..]` in Q4/Q2 templates |
| 3 | Safe parameterization | `session.run(cypher, **slots)` |
| 4 | Cypher injection | f-string vs `$param` side-by-side |
| 5 | Fail-loud error | `raise UnsupportedQueryError` not `return []` |
| 6 | Allowlist defense | `validate_query_shape` before `session.run` |
| 7 | Slot canonicalization | `.capitalize()` / `.title()` to match KG casing |
