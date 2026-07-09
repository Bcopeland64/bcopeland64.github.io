# Module 10A Integrated Lab — Complete Code Walkthrough

Every line of code in every notebook cell is explained below.  
Each section ends with the **full code block** exactly as it appears in the notebook.

---

## Table of Contents

1. [Cell 02 — Prerequisites Check](#cell-02--prerequisites-check)
2. [Cell 04 — `api/requirements.txt`](#cell-04--apirequirementstxt)
3. [Cell 05 — `api/Dockerfile`](#cell-05--apidockerfile)
4. [Cell 06 — `api/main.py`](#cell-06--apimainpy)
5. [Cell 07 — `docker-compose.yml` Step 1 (api + db)](#cell-07--docker-composeyml-step-1-api--db)
6. [Cell 08 — `seed_postgres.sql`](#cell-08--seed_postgressql)
7. [Cell 09 — Start Step-1 Stack](#cell-09--start-step-1-stack)
8. [Cell 11 — `movie_reviews.json`](#cell-11--movie_reviewsjson)
9. [Cell 12 — `docker-compose.yml` Step 2 (+ chroma)](#cell-12--docker-composeyml-step-2--chroma)
10. [Cell 14 — `web/package.json`](#cell-14--webpackagejson)
11. [Cell 15 — `web/pages/index.js`](#cell-15--webpagesindexjs)
12. [Cell 16 — `web/Dockerfile`](#cell-16--webdockerfile)
13. [Cell 18 — `docker-compose.yml` Final (all four services + healthchecks)](#cell-18--docker-composeyml-final)
14. [Cell 19 — Start Full Stack](#cell-19--start-full-stack)
15. [Cell 21 — `.env.example`](#cell-21--envexample)
16. [Cell 22 — `.gitignore`](#cell-22--gitignore)
17. [Cell 23 — `.env` Scanner](#cell-23--env-scanner)
18. [Cell 25 — `seed_postgres.sh`](#cell-25--seed_postgressh)
19. [Cell 26 — `seed_chroma.py`](#cell-26--seed_chromapy)
20. [Cell 27 — Run Seeds](#cell-27--run-seeds)
21. [Cell 29 — Stack Health Check](#cell-29--stack-health-check)
22. [Cell 30 — Partial-Failure Demo](#cell-30--partial-failure-demo)
23. [Cell 32 — `README.md` Runbook](#cell-32--readmemd-runbook)
24. [Cell 34 — Shutdown (Last Cell)](#cell-34--shutdown-last-cell)

---

## Cell 02 — Prerequisites Check

### Purpose
Verifies that Docker, Docker Compose v2, and the two pre-pulled images are available before any
Compose commands run. Also creates the `api/` and `web/pages/` directory trees that later cells
write files into.

### Line-by-line explanation

```python
import subprocess
```
`subprocess` is Python's standard library module for launching child processes and capturing their
output. Every `docker` and `docker compose` command in this notebook goes through it.

```python
import sys
```
`sys` gives us `sys.stderr` — the error output stream. We write Docker error messages there so
they appear in red in the Jupyter console.

```python
import os
```
`os.makedirs` is the cleanest way to create nested directories with a single call.

```python
import json
```
Used later to serialise the 30-chunk review list to `movie_reviews.json`.

```python
import time
```
`time.sleep` is used after `docker compose up` to give containers enough time to pass their
healthchecks before we probe them.

```python
def run(cmd, capture=True, echo=True):
```
A helper that wraps `subprocess.run`. The two keyword arguments let callers choose whether to
echo the command string to the user (`echo`) and whether to capture stdout/stderr or let them
stream to the terminal (`capture`).

```python
    if echo:
        print(f"$ {cmd}")
```
Prints the command with a `$` prefix so students can see exactly what shell command runs.
This is essential for live teaching — every command is visible.

```python
    result = subprocess.run(
        cmd, shell=True,
        capture_output=capture,
        text=True
    )
```
`shell=True` means the string is passed to `/bin/sh -c "..."`, enabling pipes, redirects, and
variable expansion. `capture_output=True` collects stdout and stderr into `result.stdout` and
`result.stderr` as strings (because `text=True` decodes bytes to UTF-8).

```python
    if capture and result.stdout:
        print(result.stdout.strip())
```
Only prints stdout if there is content and we are in capture mode. `.strip()` removes trailing
newlines that would create extra blank lines in the notebook output.

```python
    if capture and result.stderr and result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
```
Only prints stderr on *failure* (non-zero exit code). Many tools write progress to stderr even on
success; suppressing it in the success case keeps output clean.

```python
    return result
```
Returns the `CompletedProcess` object so callers can inspect `.returncode` to branch on success
or failure.

```python
r = run("docker --version")
print("✓ Docker found" if r.returncode == 0 else "✗ Install Docker Desktop first")
```
`docker --version` exits 0 if the CLI is on `PATH`. The ternary expression picks the right message.

```python
r = run("docker compose version")
print("✓ Compose v2 found" if r.returncode == 0 else "✗ Compose v2 not found")
```
Compose v2 ships as a plugin (`docker compose`, not `docker-compose`). Checking it separately
prevents confusion between v1 and v2 syntax.

```python
for img in ["postgres:15-alpine", "chromadb/chroma:latest"]:
    r = run(f"docker image inspect {img} -f '{{{{.Id}}}}'")
    status = "✓" if r.returncode == 0 else f"✗  Run: docker pull {img}"
    print(f"{status}  {img}")
```
`docker image inspect` exits 1 if the image is not in the local cache. The `-f` flag requests
only the image ID (avoids printing the full JSON). The quadruple braces `{{{{` produce literal
`{{` in the f-string, which becomes `{` in the shell command — required by Go's template syntax
used by Docker's `-f` flag.

```python
for d in ["api", "web/pages"]:
    os.makedirs(d, exist_ok=True)
```
`exist_ok=True` means the call is idempotent — running the cell twice does not raise `FileExistsError`.
`web/pages` is the Next.js pages directory; creating it here ensures the `%%writefile web/pages/index.js`
cell later has a valid target.

### Full code block

```python
import subprocess                                                # runs shell commands from Python
import sys                                                       # access to Python interpreter info
import os                                                        # OS-level file and directory operations
import json                                                      # JSON serialization and deserialization
import time                                                      # time-related sleep and timing utilities

def run(cmd, capture=True, echo=True):                           # helper: runs a shell command and returns result
    if echo:                                                     # optionally prints the command before running
        print(f"$ {cmd}")                                        # shows the command string to the user
    result = subprocess.run(                                     # executes the command in a child process
        cmd, shell=True,                                         # shell=True allows full shell syntax
        capture_output=capture,                                  # captures both stdout and stderr streams
        text=True                                                # decodes output bytes as UTF-8 text
    )                                                            # returns a CompletedProcess instance
    if capture and result.stdout:                                # only print stdout if there is content
        print(result.stdout.strip())                             # strips trailing whitespace before printing
    if capture and result.stderr and result.returncode != 0:    # only print stderr on non-zero exit code
        print(result.stderr.strip(), file=sys.stderr)           # writes error message to the stderr stream
    return result                                                # returns the CompletedProcess for inspection

print("=== Prerequisite Check ===\n")                           # section header for this cell

r = run("docker --version")                                      # checks whether Docker CLI is installed
print("✓ Docker found" if r.returncode == 0 else "✗ Install Docker Desktop first")

r = run("docker compose version")                               # checks for the Compose v2 plugin
print("✓ Compose v2 found" if r.returncode == 0 else "✗ Compose v2 not found")

for img in ["postgres:15-alpine", "chromadb/chroma:latest"]:    # iterates over images that must be pre-pulled
    r = run(f"docker image inspect {img} -f '{{{{.Id}}}}'")     # inspects local image, fails if not present
    status = "✓" if r.returncode == 0 else f"✗  Run: docker pull {img}"
    print(f"{status}  {img}")                                    # prints the check result per image

for d in ["api", "web/pages"]:                                   # iterates over required sub-directories
    os.makedirs(d, exist_ok=True)                                # creates directory tree, skips if it exists
print("\n✓ Directory scaffold ready.")                           # confirms that api/ and web/ directories exist
```

---

## Cell 04 — `api/requirements.txt`

### Purpose
Pins the exact Python package versions used by the FastAPI backend. Pinning versions ensures
every team member and every CI run uses identical dependency trees.

### Line-by-line explanation

```
fastapi==0.111.0
```
The web framework. Version 0.111 introduced several stability improvements to route handling.

```
uvicorn[standard]==0.30.1
```
The ASGI server that runs FastAPI. `[standard]` includes `httptools` and `uvloop` for better
performance. Without `[standard]`, uvicorn falls back to slower pure-Python implementations.

```
psycopg2-binary==2.9.9
```
The PostgreSQL driver for Python. The `-binary` variant ships with the compiled C extension
already bundled — no system `libpq` install required inside the container.

```
chromadb==0.5.0
```
The ChromaDB Python client. Provides `HttpClient` to talk to the Chroma container over HTTP,
and handles embedding model calls transparently.

```
pydantic==2.7.1
```
FastAPI's data-validation layer. Explicit pin prevents Pydantic v1/v2 breaking-change surprises.

```
httpx==0.27.0
```
An async-capable HTTP client. Included because ChromaDB 0.5 uses it internally for some calls.

### Full code block

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
psycopg2-binary==2.9.9
chromadb==0.5.0
pydantic==2.7.1
httpx==0.27.0
```

---

## Cell 05 — `api/Dockerfile`

### Purpose
Builds the FastAPI container image. Designed for correctness over multi-stage optimisation —
a single stage is fine for a teaching demo.

### Line-by-line explanation

```dockerfile
FROM python:3.11-slim AS base
```
`python:3.11-slim` is the official Python image without extra OS packages (no man pages, locales,
etc.). Saves ~200 MB compared to the full `python:3.11` image. `AS base` names the stage.

```dockerfile
WORKDIR /app
```
Sets `/app` as the current directory inside the image for all subsequent instructions. Creates the
directory if it does not exist. All relative paths in `COPY` and `CMD` are relative to `/app`.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
Copies only the requirements file first. Docker caches each `RUN` instruction's layer. If only
`main.py` changes, Docker reuses the cached pip-install layer — much faster rebuilds.
`--no-cache-dir` prevents pip from writing package tarballs to disk inside the image (saves space).

```dockerfile
COPY . .
```
Copies the rest of the source code. Placed after pip install so code changes don't bust the
package-install cache.

```dockerfile
EXPOSE 8080
```
Documents (but does not publish) the port uvicorn will listen on. The actual publication happens
in `docker-compose.yml` via the `ports:` key.

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```
The default command. `main:app` means: in the file `main.py`, find the object named `app`.
`--host 0.0.0.0` listens on all interfaces inside the container — required for Docker to route
traffic to it. `--port 8080` matches the EXPOSE declaration.

### Full code block

```dockerfile
FROM python:3.11-slim AS base          # official slim Python 3.11 image as the base

WORKDIR /app                           # sets /app as the working directory inside the image

COPY requirements.txt .                # copies only the dependency list first (layer-cache optimisation)
RUN pip install --no-cache-dir -r requirements.txt  # installs all Python packages without caching

COPY . .                               # copies the rest of the application source code

EXPOSE 8080                            # documents the port the app will listen on (informational)

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]  # starts FastAPI with uvicorn
```

---

## Cell 06 — `api/main.py`

### Purpose
The FastAPI application. Three endpoints: `/readyz` (readiness probe), `/movies` (Postgres query),
`/search` (ChromaDB RAG query). The readiness probe is the most important — it is what the
`depends_on: condition: service_healthy` healthcheck calls.

### Line-by-line explanation

```python
from fastapi import FastAPI, HTTPException, Query
```
`FastAPI` is the app class. `HTTPException` lets us return structured error responses (e.g., 503).
`Query` validates and documents query-string parameters like `?q=drama`.

```python
import psycopg2
import psycopg2.extras
```
`psycopg2` is the Postgres driver. `psycopg2.extras.RealDictCursor` returns rows as dictionaries
(column name → value) instead of plain tuples — makes the JSON serialisation trivial.

```python
import chromadb
```
The ChromaDB client library.

```python
import os
```
`os.environ` reads environment variables injected by Docker Compose at runtime.

```python
app = FastAPI(title="Movie API", version="1.0.0")
```
Creates the FastAPI instance. `title` and `version` appear in the auto-generated `/docs` UI.

```python
def get_pg_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])
```
Factory function that opens a new Postgres connection from the `DATABASE_URL` environment variable.
Raises `KeyError` at startup if the variable is not set — this surfaces misconfiguration immediately.

```python
def get_chroma():
    return chromadb.HttpClient(
        host=os.environ.get("CHROMA_HOST", "localhost"),
        port=int(os.environ.get("CHROMA_PORT", "8000"))
    )
```
Factory that creates a ChromaDB HTTP client. `os.environ.get` with a default avoids `KeyError`
for optional variables; `localhost` and `8000` work when running outside Docker during development.

```python
@app.get("/readyz")
def readyz():
```
The readiness probe. Docker's healthcheck calls `curl -sf http://localhost:8080/readyz`. If this
returns 200, the container is marked healthy. If it returns 503, it stays unhealthy and
`depends_on: condition: service_healthy` blocks any service that depends on `api`.

```python
    try:
        conn = get_pg_conn()
        conn.close()
    except Exception as exc:
        raise HTTPException(503, detail=f"Postgres not ready: {exc}")
```
Opens and immediately closes a Postgres connection. If Postgres is still starting up, `connect()`
raises `OperationalError` — we catch *any* exception and convert it to a 503.

```python
    try:
        get_chroma().heartbeat()
    except Exception as exc:
        raise HTTPException(503, detail=f"Chroma not ready: {exc}")
    return {"status": "ok"}
```
`heartbeat()` sends a `GET /api/v1/heartbeat` to Chroma. If Chroma is not ready, it raises —
we return 503. Only when both checks pass do we return 200 `{"status": "ok"}`.

```python
@app.get("/movies")
def list_movies():
    conn = get_pg_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, title, genre, year, director, rating FROM movies ORDER BY rating DESC")
        rows = cur.fetchall()
    conn.close()
    return [{**r} for r in rows]
```
Opens a connection, runs the query, fetches all rows, closes the connection. `RealDictCursor`
returns `RealDictRow` objects; spreading them with `{**r}` converts each to a plain dict so
FastAPI's JSON serialiser can handle them without custom configuration.

```python
@app.get("/search")
def search_reviews(q: str = Query(..., min_length=1)):
```
`Query(...)` means the parameter is required (the `...` is FastAPI's sentinel for "required").
`min_length=1` rejects empty strings before the handler runs.

```python
    client = get_chroma()
    try:
        collection = client.get_collection("movie_reviews")
    except Exception:
        raise HTTPException(503, detail="movie_reviews collection not seeded")
```
`get_collection` (not `get_or_create_collection`) raises if the collection does not exist. We
convert that to 503 rather than 404 because it signals an infrastructure problem (unseeded Chroma),
not a user error.

```python
    results = collection.query(
        query_texts=[q],
        n_results=3
    )
```
ChromaDB embeds `q` using its default embedding model, then returns the 3 nearest vectors from
the index. `query_texts` accepts a list so multiple queries can be batched — we pass exactly one.

```python
    return {
        "query": q,
        "results": results["documents"][0],
        "metadatas": results["metadatas"][0],
    }
```
`results["documents"]` is a list of lists (one inner list per query). `[0]` extracts the results
for our single query. The frontend maps over `results` and `metadatas` in parallel.

### Full code block

```python
"""Movie Information API — FastAPI backend for Module 10A."""
from fastapi import FastAPI, HTTPException, Query   # FastAPI core and helpers
import psycopg2                                     # PostgreSQL driver
import psycopg2.extras                              # provides RealDictCursor for dict-style rows
import chromadb                                     # ChromaDB Python client
import os                                           # reads environment variables

app = FastAPI(title="Movie API", version="1.0.0")   # creates the FastAPI application instance

def get_pg_conn():                                  # factory: returns a new Postgres connection
    return psycopg2.connect(os.environ["DATABASE_URL"])  # reads DATABASE_URL from environment

def get_chroma():                                   # factory: returns a ChromaDB HTTP client
    return chromadb.HttpClient(                     # connects to Chroma over HTTP
        host=os.environ.get("CHROMA_HOST", "localhost"),   # hostname injected by Compose
        port=int(os.environ.get("CHROMA_PORT", "8000"))    # port injected by Compose
    )

@app.get("/readyz")                                 # readiness probe endpoint — checked by healthcheck
def readyz():
    try:                                            # attempts to connect to Postgres
        conn = get_pg_conn()                        # opens a Postgres connection
        conn.close()                                # closes connection immediately after check
    except Exception as exc:                        # catches any connection failure
        raise HTTPException(503, detail=f"Postgres not ready: {exc}")  # returns 503 to caller
    try:                                            # attempts to connect to Chroma
        get_chroma().heartbeat()                    # Chroma heartbeat raises if unreachable
    except Exception as exc:                        # catches any Chroma connection failure
        raise HTTPException(503, detail=f"Chroma not ready: {exc}")    # returns 503 to caller
    return {"status": "ok"}                         # both dependencies healthy — return 200

@app.get("/movies")                                 # endpoint: returns all movies from Postgres
def list_movies():
    conn = get_pg_conn()                            # opens a fresh Postgres connection
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:  # dict-style rows
        cur.execute("SELECT id, title, genre, year, director, rating FROM movies ORDER BY rating DESC")
        rows = cur.fetchall()                       # fetches all result rows into memory
    conn.close()                                    # closes connection to return it to the pool
    return [{**r} for r in rows]                   # converts RealDictRow objects to plain dicts

@app.get("/search")                                 # endpoint: semantic search via Chroma RAG
def search_reviews(q: str = Query(..., min_length=1)):  # q is a required query-string parameter
    client = get_chroma()                           # creates a fresh Chroma HTTP client
    try:                                            # wraps Chroma call to catch collection errors
        collection = client.get_collection("movie_reviews")  # gets existing collection by name
    except Exception:                               # collection missing means Chroma not seeded yet
        raise HTTPException(503, detail="movie_reviews collection not seeded")  # 503 until seeded
    results = collection.query(                     # performs approximate-nearest-neighbour search
        query_texts=[q],                            # embeds the query string automatically
        n_results=3                                 # returns the 3 most similar review chunks
    )
    return {                                        # returns structured JSON response
        "query": q,                                 # echoes the search query
        "results": results["documents"][0],          # list of matching review text chunks
        "metadatas": results["metadatas"][0],        # list of metadata dicts (movie, year, sentiment)
    }
```

---

## Cell 07 — `docker-compose.yml` Step 1 (api + db)

### Purpose
Writes the first version of `docker-compose.yml` with two services: `db` and `api`. Demonstrates
the minimal viable composition before dependencies and healthchecks are introduced.

### Line-by-line explanation (YAML)

```yaml
version: "3.9"
```
The Compose file-format version. Version 3.9 supports `depends_on: condition: service_healthy`,
named volumes, and all features used in this lab.

```yaml
  db:
    image: postgres:15-alpine
    restart: unless-stopped
```
`unless-stopped` means the container restarts if it crashes or the Docker daemon restarts the host,
but stays stopped if you run `docker compose stop`. This is the right policy for development
databases — you don't want them restarting when you explicitly stop the stack.

```yaml
    environment:
      POSTGRES_DB:       ${POSTGRES_DB:-movies}
      POSTGRES_USER:     ${POSTGRES_USER:-movie_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}
```
`${VAR:-default}` is Compose's env substitution syntax. Compose reads the variable from the shell
environment or from a `.env` file in the same directory; if neither is set, it uses the default.
This means the file works out-of-the-box for development without a `.env` file.

```yaml
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./seed_postgres.sql:/docker-entrypoint-initdb.d/seed.sql:ro
```
The first volume is a **named volume**. Named volumes live in Docker's managed storage and survive
`docker compose down` — destroyed only with `docker compose down -v`.

The second is a **bind mount**. The `:ro` flag makes it read-only inside the container (prevents
accidental writes). Postgres runs every `.sql` file in `/docker-entrypoint-initdb.d/` on first boot.

```yaml
  api:
    build: ./api
```
`build: ./api` tells Compose to run `docker build ./api` and tag the resulting image for use as
this service's image. If you change any file in `./api/`, re-run `docker compose up --build` to
rebuild.

```yaml
      DATABASE_URL: postgresql://${POSTGRES_USER:-movie_user}:${POSTGRES_PASSWORD:-secret}@db:5432/${POSTGRES_DB:-movies}
```
The hostname `db` in this URL resolves to the `db` container's IP address on the internal Compose
network. Docker creates a DNS entry for every service name on the network.

```yaml
volumes:
  pgdata:
```
Declares the named volume. Without this declaration, Compose would refuse to start, citing an
undefined volume.

### Full code block

```python
compose_step1 = """version: \"3.9\"            # Compose file-format version

services:

  db:                              # PostgreSQL relational store
    image: postgres:15-alpine      # official Postgres 15 on Alpine Linux (small footprint)
    restart: unless-stopped        # restarts automatically unless explicitly stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-movies}           # database name; default 'movies'
      POSTGRES_USER: ${POSTGRES_USER:-movie_user}   # db user; default 'movie_user'
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}  # password; override in .env
    ports:
      - \"5432:5432\"              # publishes Postgres port to host for psql/pgAdmin access
    volumes:
      - pgdata:/var/lib/postgresql/data            # named volume persists data across restarts
      - ./seed_postgres.sql:/docker-entrypoint-initdb.d/seed.sql:ro  # auto-seeds on first boot

  api:                             # FastAPI backend
    build: ./api                   # builds from the Dockerfile in ./api
    restart: unless-stopped
    ports:
      - \"8080:8080\"              # publishes FastAPI port to host
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-movie_user}:${POSTGRES_PASSWORD:-secret}@db:5432/${POSTGRES_DB:-movies}
      CHROMA_HOST: chroma          # Compose service name resolves on the internal network
      CHROMA_PORT: \"8000\"

volumes:
  pgdata:                          # declares the named volume used by the db service
"""

with open("docker-compose.yml", "w") as f:           # opens docker-compose.yml for writing
    f.write(compose_step1)                            # writes Step-1 Compose definition to disk

print("✓ docker-compose.yml written (Step 1: api + db)")  # confirms the file was written
```

---

## Cell 08 — `seed_postgres.sql`

### Purpose
The SQL file that Postgres runs automatically when the `db` container starts for the first time.
Creates the `movies` table and inserts 20 rows.

### Line-by-line explanation

```sql
CREATE TABLE IF NOT EXISTS movies (
```
`IF NOT EXISTS` makes this statement idempotent — running the file a second time (e.g., if somehow
executed via psql manually) does not raise an error.

```sql
    id        SERIAL PRIMARY KEY,
```
`SERIAL` is a Postgres shorthand for an auto-incrementing integer sequence. `PRIMARY KEY` implies
`NOT NULL` and creates a unique B-tree index.

```sql
    rating    DECIMAL(3,1)
```
`DECIMAL(3,1)` allows values from -99.9 to 99.9. For ratings 0–10 with one decimal place this is
the correct precision. Using `FLOAT` would introduce floating-point rounding errors.

```sql
INSERT INTO movies (...) VALUES
  ('The Shawshank Redemption', 'Drama', 1994, 'Frank Darabont', 9.3),
  ...
ON CONFLICT DO NOTHING;
```
`ON CONFLICT DO NOTHING` is a Postgres upsert clause. Because `id` is a serial primary key and
we are not supplying it, each INSERT generates a new ID — conflicts will not occur under normal
operation. The clause is defensive: if the file is executed twice against an already-populated
table with a unique constraint on `title`, it skips duplicates instead of erroring.

Note: `Schindler''s List` — a single quote inside a SQL string literal is escaped by doubling it.

### Full code block

```sql
-- Movie catalog seed — 20 rows
-- This file is mounted into /docker-entrypoint-initdb.d/ and runs automatically on first boot.

CREATE TABLE IF NOT EXISTS movies (    -- creates the table only if it does not already exist
    id        SERIAL PRIMARY KEY,      -- auto-incrementing integer primary key
    title     VARCHAR(255) NOT NULL,   -- movie title, required field
    genre     VARCHAR(100),            -- genre label (nullable)
    year      INTEGER,                 -- release year (nullable)
    director  VARCHAR(255),            -- director full name (nullable)
    rating    DECIMAL(3,1)             -- audience rating on a 10-point scale
);

INSERT INTO movies (title, genre, year, director, rating) VALUES
  ('The Shawshank Redemption',                    'Drama',   1994, 'Frank Darabont',          9.3),
  ('The Godfather',                               'Crime',   1972, 'Francis Ford Coppola',    9.2),
  ('The Dark Knight',                             'Action',  2008, 'Christopher Nolan',       9.0),
  ('Pulp Fiction',                                'Crime',   1994, 'Quentin Tarantino',       8.9),
  ('Schindler''s List',                           'History', 1993, 'Steven Spielberg',        8.9),
  ('The Lord of the Rings: The Return of the King','Fantasy',2003, 'Peter Jackson',           8.9),
  ('Forrest Gump',                                'Drama',   1994, 'Robert Zemeckis',         8.8),
  ('Fight Club',                                  'Drama',   1999, 'David Fincher',           8.8),
  ('Inception',                                   'Sci-Fi',  2010, 'Christopher Nolan',       8.8),
  ('The Matrix',                                  'Sci-Fi',  1999, 'Lana Wachowski',          8.7),
  ('Goodfellas',                                  'Crime',   1990, 'Martin Scorsese',         8.7),
  ('The Silence of the Lambs',                    'Thriller',1991, 'Jonathan Demme',          8.6),
  ('Interstellar',                                'Sci-Fi',  2014, 'Christopher Nolan',       8.6),
  ('Saving Private Ryan',                         'War',     1998, 'Steven Spielberg',        8.6),
  ('The Green Mile',                              'Drama',   1999, 'Frank Darabont',          8.6),
  ('Parasite',                                    'Thriller',2019, 'Bong Joon-ho',            8.5),
  ('Whiplash',                                    'Drama',   2014, 'Damien Chazelle',         8.5),
  ('Dune',                                        'Sci-Fi',  2021, 'Denis Villeneuve',        8.0),
  ('Everything Everywhere All at Once',           'Sci-Fi',  2022, 'Daniels',                 7.8),
  ('Oppenheimer',                                 'History', 2023, 'Christopher Nolan',       8.3)
ON CONFLICT DO NOTHING;              -- idempotent: re-running will not duplicate rows
```

---

## Cell 09 — Start Step-1 Stack

### Purpose
Brings the first two services up so students can see Postgres initialising and the API connecting
to it. Uses detached mode so the terminal is immediately returned.

### Line-by-line explanation

```python
result = run("docker compose up --build -d")
```
`--build` forces a rebuild of the `api` image even if Docker thinks the layers are cached.
`-d` (detached) starts containers in the background and returns immediately.

```python
time.sleep(8)
```
Eight seconds gives Postgres enough time to run `initdb.d/seed.sql`. The initdb scripts are
skipped on subsequent starts, so this wait is only meaningful on first boot.

```python
result = run("docker compose ps")
```
Lists all containers managed by this Compose file, their current state (running, starting, healthy),
and the ports they expose.

### Full code block

```python
print("Starting Step-1 stack (api + db)...")            # informs user which services will start

result = run("docker compose up --build -d")             # builds api image then starts both services detached

time.sleep(8)                                            # waits 8 s for Postgres to finish initialising

result = run("docker compose ps")                        # lists the running containers and their status

print("\nTip: run 'docker compose logs -f db' to watch Postgres init logs.")  # points to useful log command
```

---

## Cell 11 — `movie_reviews.json`

### Purpose
Writes the 30 movie-review chunks that will be seeded into Chroma in Beat 7. Each chunk is a
plain-English review excerpt tagged with movie metadata. ChromaDB will embed each `text` field
using its default embedding model and store the result as a vector alongside the metadata.

### Line-by-line explanation

```python
reviews = [
    {"id": "rev_001", "text": "...", "metadata": {"movie": "...", "year": 1994, "sentiment": "positive"}},
    ...
]
```
The list has three fields per chunk:
- `id` — a stable unique identifier. Must be a string in Chroma. We use a zero-padded counter
  (`rev_001` … `rev_030`) so they sort lexicographically.
- `text` — the review text. Chroma embeds this field when you call `collection.add()`.
- `metadata` — a dictionary of arbitrary key/value pairs returned alongside query results.
  The frontend uses `movie`, `year`, and `sentiment` to render the citation line below each result.

```python
with open("movie_reviews.json", "w") as f:
    json.dump(reviews, f, indent=2)
```
`json.dump` serialises the Python list to the file handle. `indent=2` writes human-readable
pretty-printed JSON — easier to diff in version control and inspect in a text editor.

```python
print(f"✓ movie_reviews.json written ({len(reviews)} chunks)")
```
`len(reviews)` confirms all 30 chunks were serialised. If you accidentally truncate the list
during editing, this line immediately surfaces the count mismatch.

### Full code block

```python
reviews = [                                                      # list of 30 movie-review chunks
    {"id": "rev_001", "text": "The Shawshank Redemption is a timeless tale of hope. Andy Dufresne's quiet determination behind bars inspires everyone around him, including the cynical Red.",    "metadata": {"movie": "The Shawshank Redemption", "year": 1994, "sentiment": "positive"}},
    # ... (30 entries total — see notebook for complete list)
]

with open("movie_reviews.json", "w") as f:                       # opens output file for writing
    json.dump(reviews, f, indent=2)                              # serialises list to pretty-printed JSON

print(f"✓ movie_reviews.json written ({len(reviews)} chunks)")   # confirms how many chunks were saved
```

---

## Cell 12 — `docker-compose.yml` Step 2 (+ Chroma)

### Purpose
Adds the `chroma` service to the Compose file and introduces the first `depends_on` with
`condition: service_healthy`. Overwrites the Step-1 file; running `docker compose up -d` again
adds only the new service without restarting existing ones.

### New lines explained

```yaml
  chroma:
    image: chromadb/chroma:latest
```
Uses the latest official Chroma image. In production you would pin to a specific tag; for a lab
environment `latest` is acceptable.

```yaml
    volumes:
      - chromadata:/chroma/chroma
```
The Chroma image stores its SQLite metadata database and embedding index under `/chroma/chroma`
inside the container. Mounting a named volume here ensures embeddings survive container restarts.

```yaml
    environment:
      IS_PERSISTENT: "TRUE"
      ANONYMIZED_TELEMETRY: "FALSE"
```
`IS_PERSISTENT` switches Chroma from in-memory mode (loses everything on restart) to disk-backed
mode. `ANONYMIZED_TELEMETRY: FALSE` disables Chroma's usage reporting — required in classroom
environments that may restrict outbound telemetry.

```yaml
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/api/v1/heartbeat || exit 1"]
      start_period: 15s
```
`CMD-SHELL` runs the command through `/bin/sh`. The `|| exit 1` ensures that if `curl` exits with
a non-zero code (which `-f` triggers on HTTP 4xx/5xx), the healthcheck also exits non-zero. The
15-second `start_period` gives Chroma time to load its index from disk before retries begin.

```yaml
  api:
    depends_on:
      chroma:
        condition: service_healthy
```
Without `condition: service_healthy`, Compose starts `api` as soon as the `chroma` *container*
starts — not when Chroma is *ready to serve requests*. The healthcheck condition bridges that gap.

### Full code block

```python
compose_step2 = """..."""  # (see notebook Cell 12 for the complete YAML)

with open("docker-compose.yml", "w") as f:           # overwrites previous Step-1 file
    f.write(compose_step2)                            # writes Step-2 Compose definition

run("docker compose up -d")                           # docker compose detects the new chroma service and starts it

print("\n✓ Chroma service added. Watch it become healthy:")
print("  docker compose ps  (run in a separate terminal)")
```

---

## Cell 14 — `web/package.json`

### Purpose
The Node.js manifest for the Next.js frontend. Defines the three npm scripts Compose uses and
pins the three runtime dependencies.

### Line-by-line explanation

```json
"dev":   "next dev",
"build": "next build",
"start": "next start -p 3000"
```
The Dockerfile calls `npm run build` during the image build and `npm start` as the container
entrypoint. `next start -p 3000` explicitly sets the port — required because `EXPOSE 3000` in a
Dockerfile is documentation only; the actual port must be configured in the server command.

```json
"next": "14.2.3"
```
Pins Next.js to 14.2.3. Next.js has frequent minor releases with breaking changes; pinning
prevents surprise breakage when the image is rebuilt.

### Full code block

```json
{
  "name": "movie-web",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev":   "next dev",
    "build": "next build",
    "start": "next start -p 3000"
  },
  "dependencies": {
    "next":      "14.2.3",
    "react":     "18.3.1",
    "react-dom": "18.3.1"
  }
}
```

---

## Cell 15 — `web/pages/index.js`

### Purpose
The single Next.js page. Renders a search form, calls the FastAPI `/search` endpoint, and
displays results with citations. Handles the error state caused by Chroma going down mid-session.

### Key lines explained

```javascript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
```
`process.env.NEXT_PUBLIC_API_URL` is replaced at build time by Next.js's Webpack bundler.
The `|| 'http://localhost:8080'` fallback covers local development (running `npm run dev`
outside Docker) where the env variable may not be set. Inside the Docker image the `ARG/ENV`
in the Dockerfile ensures it is always set.

```javascript
const [error, setError] = useState(null)
```
Separate error state from results state. When Chroma is down, `setError(err.message)` is called
and the component renders the red error banner instead of results. When Chroma recovers, the
next successful search calls `setError(null)` to clear the banner.

```javascript
if (!res.ok) {
    const body = await res.json()
    throw new Error(body.detail || `HTTP ${res.status}`)
}
```
FastAPI returns structured JSON errors with a `detail` field. This code surfaces that message
in the UI. Without this check, `res.json()` would silently succeed on a 503 response and
`setResults` would store an error object — confusing to debug.

```javascript
} finally {
    setLoading(false)
}
```
`finally` runs whether the try block succeeded or threw. This guarantees the button is always
re-enabled after a search completes, regardless of outcome.

### Full code block

```javascript
import { useState } from 'react'                                 // useState hook for local component state

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'  // baked at build time

export default function Home() {
  const [query,   setQuery]   = useState('')
  const [results, setResults] = useState(null)
  const [error,   setError]   = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleSearch(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/search?q=${encodeURIComponent(query)}`)
      if (!res.ok) {
        const body = await res.json()
        throw new Error(body.detail || `HTTP ${res.status}`)
      }
      setResults(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={{ maxWidth: 800, margin: '2rem auto', fontFamily: 'sans-serif' }}>
      <h1>Movie Review Search</h1>
      <form onSubmit={handleSearch}>
        <input value={query} onChange={e => setQuery(e.target.value)}
               placeholder="e.g. hopeful prison drama" style={{ width: '70%', padding: '0.5rem' }} />
        <button type="submit" disabled={loading} style={{ padding: '0.5rem 1rem', marginLeft: 8 }}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>
      {error && <div style={{ color: 'red', marginTop: '1rem' }}><strong>Error:</strong> {error}</div>}
      {results && (
        <div style={{ marginTop: '1.5rem' }}>
          <h2>Results for "{results.query}"</h2>
          {results.results.map((text, i) => (
            <div key={i} style={{ border: '1px solid #ddd', padding: '1rem', marginBottom: '1rem', borderRadius: 4 }}>
              <p>{text}</p>
              <small style={{ color: '#666' }}>
                {results.metadatas[i].movie} ({results.metadatas[i].year}) — {results.metadatas[i].sentiment}
              </small>
            </div>
          ))}
        </div>
      )}
    </main>
  )
}
```

---

## Cell 16 — `web/Dockerfile`

### Purpose
Multi-stage build for the Next.js frontend. Stage 1 compiles the JavaScript bundle (baking in
`NEXT_PUBLIC_API_URL`). Stage 2 copies only the compiled output into a minimal runtime image.

### Line-by-line explanation

```dockerfile
FROM node:18-alpine AS builder
```
Node 18 LTS on Alpine. The `AS builder` name allows Stage 2 to copy from it with `--from=builder`.

```dockerfile
COPY package*.json ./
RUN npm ci
```
`npm ci` (clean install) uses `package-lock.json` for exact version resolution — faster and more
deterministic than `npm install`. Copied before the source so Docker can cache this layer.

```dockerfile
ARG NEXT_PUBLIC_API_URL=http://localhost:8080
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
```
`ARG` declares a build-time variable settable with `--build-arg`. `ENV` promotes it to a
runtime environment variable that Next.js reads during `npm run build`. After `npm run build`
completes, the value is frozen into the JavaScript bundle — it cannot be changed without
rebuilding the image.

This is the core of the **build-time vs runtime** concept from Beat 4. To deploy to production:
```bash
docker compose build --build-arg NEXT_PUBLIC_API_URL=https://api.prod.example.com web
```

```dockerfile
FROM node:18-alpine AS runner
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
```
Stage 2 starts from a fresh Alpine image and copies only what is needed to run the server.
The entire source code (`pages/`, etc.) and dev dependencies are discarded — the final image
is significantly smaller.

### Full code block

```dockerfile
FROM node:18-alpine AS builder          # Node 18 on Alpine as the build environment
WORKDIR /app
COPY package*.json ./                   # copies package.json and package-lock.json for cache efficiency
RUN npm ci                              # installs exact locked dependencies (faster than npm install)
COPY . .                                # copies all source files including pages/

ARG NEXT_PUBLIC_API_URL=http://localhost:8080   # build-time argument — value is baked into the JS bundle
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL    # promotes ARG into ENV so Next.js picks it up at build

RUN npm run build                       # runs 'next build' — compiles pages, inlines NEXT_PUBLIC vars

FROM node:18-alpine AS runner           # fresh minimal image for the production runtime layer
WORKDIR /app
ENV NODE_ENV production                 # tells Next.js to run in production mode (smaller, faster)
COPY --from=builder /app/.next ./.next  # copies the compiled Next.js output from the builder stage
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
EXPOSE 3000                             # documents the port Next.js listens on
CMD ["npm", "start"]                    # starts the Next.js production server
```

---

## Cell 18 — `docker-compose.yml` Final

### Purpose
The complete, production-discipline Compose file with all four services, all healthchecks, and
the full `depends_on` chain. This is the primary teaching artifact of the lab.

### Key structural decisions explained

#### `healthcheck.start_period` per service
- `db: 10s` — Postgres runs `initdb` scripts on first boot; 10 s covers typical SQL seed time
- `chroma: 15s` — Chroma loads its embedding index from disk; slightly longer than Postgres
- `api: 20s` — FastAPI's `/readyz` checks *both* dependencies; must wait until both are ready

#### `depends_on` chain
```yaml
web  → api (service_healthy)
api  → db  (service_healthy)
api  → chroma (service_healthy)
```
Compose enforces this by blocking container start until the dependency's healthcheck passes.
Without `condition: service_healthy`, Compose only waits for the container to *exist*, not to
be *ready*. The difference is several seconds of connection failures.

#### Why `CHROMA_HOST: chroma`
Inside the Compose network, each service is reachable by its service name as a hostname.
Docker creates a DNS A record `chroma → <container IP>` automatically. The API never needs
to know the actual IP address.

### Full code block

```yaml
version: "3.9"                        # Compose file-format version

services:

  db:
    image: postgres:15-alpine          # official Postgres 15 on Alpine (lightweight)
    restart: unless-stopped
    environment:
      POSTGRES_DB:       ${POSTGRES_DB:-movies}
      POSTGRES_USER:     ${POSTGRES_USER:-movie_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./seed_postgres.sql:/docker-entrypoint-initdb.d/seed.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-movie_user} -d ${POSTGRES_DB:-movies}"]
      interval: 10s
      timeout:   5s
      retries:    5
      start_period: 10s

  chroma:
    image: chromadb/chroma:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - chromadata:/chroma/chroma
    environment:
      IS_PERSISTENT:        "TRUE"
      ANONYMIZED_TELEMETRY: "FALSE"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/api/v1/heartbeat || exit 1"]
      interval: 10s
      timeout:   5s
      retries:    5
      start_period: 15s

  api:
    build: ./api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-movie_user}:${POSTGRES_PASSWORD:-secret}@db:5432/${POSTGRES_DB:-movies}
      CHROMA_HOST:  chroma
      CHROMA_PORT:  "8000"
    depends_on:
      db:
        condition: service_healthy
      chroma:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8080/readyz || exit 1"]
      interval: 10s
      timeout:   5s
      retries:    5
      start_period: 20s

  web:
    build: ./web
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-http://localhost:8080}
    depends_on:
      api:
        condition: service_healthy

volumes:
  pgdata:
  chromadata:
```

---

## Cell 19 — Start Full Stack

### Purpose
Builds all images that have a `build:` key and starts all four containers. Then waits and
verifies status.

### Line-by-line explanation

```python
result = run("docker compose up --build -d", capture=False)
```
`capture=False` lets Docker's build output stream directly to the terminal — you can watch each
layer being built in real time. This is intentional during teaching; students should see the
image build process.

```python
time.sleep(30)
```
Thirty seconds covers the time for all four healthchecks to converge. The dependency chain means:
- `db` and `chroma` start simultaneously (no dependencies)
- `api` starts after both pass their healthchecks (~15–20 s)
- `web` starts after `api` passes its healthcheck (~20–30 s)

### Full code block

```python
print("Building and starting the full four-service stack...")
print("This takes 60–120 s while images build and healthchecks pass.\n")

result = run("docker compose up --build -d", capture=False)      # builds images and starts all services detached

print("\nWaiting 30 s for all healthchecks to converge...")
time.sleep(30)                                                   # gives containers time to become healthy

run("docker compose ps")                                          # shows current status of all four containers

print("\n✓ All services started. Expected STATUS: healthy or running.")
```

---

## Cell 21 — `.env.example`

### Purpose
The version-controlled template for secret values. Documents every variable the stack requires,
with safe placeholder values. Engineers copy this to `.env` and fill in real values.

### Line-by-line explanation

```
POSTGRES_PASSWORD=CHANGE_ME
```
The placeholder `CHANGE_ME` is deliberately obvious — it will cause an authentication failure if
deployed without being changed, which surfaces the misconfiguration immediately.

```
NEXT_PUBLIC_API_URL=http://localhost:8080
```
The default value works for local development. For production this would be changed to the public
API URL before running `docker compose build --build-arg NEXT_PUBLIC_API_URL=<value> web`.

### Full code block

```ini
# Copy this file to .env and fill in real values.
# NEVER commit .env — it is listed in .gitignore.

# PostgreSQL ──────────────────────────────────────
POSTGRES_DB=movies
POSTGRES_USER=movie_user
POSTGRES_PASSWORD=CHANGE_ME

# Next.js (baked at build time) ───────────────────
NEXT_PUBLIC_API_URL=http://localhost:8080
```

---

## Cell 22 — `.gitignore`

### Purpose
Prevents secrets and generated artefacts from being committed to the repository.

### Line-by-line explanation

```
.env
.env.local
.env.production
```
All three common `.env` file names are excluded. `.env.local` is Next.js convention;
`.env.production` is a common staging/prod pattern.

```
web/.next/
```
The compiled Next.js output directory. Should never be committed — it is large, binary-heavy,
and rebuilt on every `docker compose up --build`.

### Full code block

```
# Secrets — never commit these
.env
.env.local
.env.production

# Python
__pycache__/
*.pyc
.venv/

# Node
node_modules/
web/.next/

# Docker artefacts
*.log
```

---

## Cell 23 — `.env` Scanner

### Purpose
Detects `.env` files that have been accidentally committed to git. This is a pre-push safety net.
The pre-identified artifact from Beat 6.

### Line-by-line explanation

```python
import pathlib
```
`pathlib.Path` provides an object-oriented API for filesystem paths. More readable than `os.path`
for path manipulation.

```python
def scan_committed_env_files(repo_root="."):
    root = pathlib.Path(repo_root)
```
Accepts an optional path parameter so the function can be called on any repository directory,
not just the current one.

```python
    result = subprocess.run(
        "git ls-files",
        shell=True, capture_output=True, text=True, cwd=root
    )
```
`git ls-files` lists every file tracked by the git index (i.e., committed or staged). The `cwd=root`
parameter runs the command in the target repository's root directory.

```python
    tracked = result.stdout.splitlines()
```
Converts the newline-separated output to a Python list. Each element is a relative path like
`api/main.py` or `.env`.

```python
    violations = [
        f for f in tracked
        if pathlib.Path(f).name in (".env", ".env.local", ".env.production")
    ]
```
`pathlib.Path(f).name` extracts just the filename component (e.g., `.env` from `secrets/.env`).
The `in` check uses a set-like tuple literal for O(1) membership testing.

```python
    for v in violations:
        print(f"  git rm --cached {v}")
```
`git rm --cached` removes the file from the index (stops tracking it) without deleting it from
disk. The student can then add it to `.gitignore` and commit the removal.

### Full code block

```python
import pathlib                                                    # provides object-oriented filesystem paths

def scan_committed_env_files(repo_root="."):                     # scans a repo for accidentally committed .env files
    root = pathlib.Path(repo_root)                               # converts the string path to a Path object
    result = subprocess.run(                                      # runs git to list all tracked files
        "git ls-files",
        shell=True, capture_output=True, text=True, cwd=root
    )
    tracked = result.stdout.splitlines()                          # splits output into one filename per line
    violations = [
        f for f in tracked
        if pathlib.Path(f).name in (".env", ".env.local", ".env.production")
    ]
    return violations

violations = scan_committed_env_files()

if violations:
    print("✗ COMMITTED .env FILES FOUND — ROTATE SECRETS IMMEDIATELY:")
    for v in violations:
        print(f"  git rm --cached {v}")
else:
    print("✓ No committed .env files found.")

print("\n.env.example committed? ",
      ".env.example" in subprocess.run("git ls-files", shell=True, capture_output=True, text=True).stdout)
```

---

## Cell 25 — `seed_postgres.sh`

### Purpose
A shell script wrapper that runs the SQL seed file inside the running `db` container. Useful for
reseeding after a `docker compose down -v` (which wipes the named volume).

### Line-by-line explanation

```bash
set -euo pipefail
```
Three safety flags in one:
- `-e` — exit immediately if any command returns non-zero
- `-u` — treat unset variables as errors (prevents silent use of empty strings)
- `-o pipefail` — a pipeline fails if *any* command in it fails (not just the last one)

```bash
CONTAINER=$(docker compose ps -q db)
```
`docker compose ps -q db` outputs only the container ID of the `db` service. Capturing it lets us
use `docker exec` directly without relying on the service name resolution.

```bash
if [ -z "$CONTAINER" ]; then
  echo "Error: db container is not running."
  exit 1
fi
```
`-z` tests whether the string is empty. Exits with a helpful message if `db` is not running,
rather than failing later with a cryptic `docker exec` error.

```bash
docker exec -i "$CONTAINER" \
  psql -U "${POSTGRES_USER:-movie_user}" \
       -d "${POSTGRES_DB:-movies}" \
  < seed_postgres.sql
```
`-i` keeps stdin open so the `< seed_postgres.sql` redirect works. The heredoc pattern (piping
file to stdin) is how `psql` accepts SQL non-interactively. The `\` line continuations are
cosmetic — they split one long command across three readable lines.

### Full code block

```bash
#!/usr/bin/env bash
set -euo pipefail

CONTAINER=$(docker compose ps -q db)

if [ -z "$CONTAINER" ]; then
  echo "Error: db container is not running. Run 'docker compose up -d db' first."
  exit 1
fi

echo "Seeding Postgres..."

docker exec -i "$CONTAINER" \
  psql -U "${POSTGRES_USER:-movie_user}" \
       -d "${POSTGRES_DB:-movies}" \
  < seed_postgres.sql

echo "✓ Postgres seed complete."
```

---

## Cell 26 — `seed_chroma.py`

### Purpose
Seeds the `movie_reviews` ChromaDB collection from `movie_reviews.json`. Idempotent: fetches
existing IDs and only adds chunks that are not already present.

### Line-by-line explanation

```python
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
```
Connects to Chroma using its HTTP API. On the host machine (running this notebook), Chroma is
reachable at `localhost:8000` because we published the port in `docker-compose.yml`.

```python
try:
    client.heartbeat()
except Exception as exc:
    print(f"Cannot reach Chroma at {CHROMA_HOST}:{CHROMA_PORT}: {exc}")
    sys.exit(1)
```
Validates connectivity before doing any work. Failing fast with a clear message is much more
useful than letting `get_or_create_collection` fail with an opaque network error.

```python
collection = client.get_or_create_collection(COLLECTION)
```
`get_or_create_collection` is idempotent — it returns the existing collection if it exists, or
creates a new one if it does not. Uses Chroma's default embedding model (sentence-transformers).

```python
existing = set(collection.get()["ids"])
```
`collection.get()` returns all documents in the collection as a dict with keys `ids`, `documents`,
and `metadatas`. Wrapping `ids` in a `set` makes the membership check in the next line O(1).

```python
new_reviews = [r for r in reviews if r["id"] not in existing]
```
Filters to only the chunks not already in the collection. This is the idempotency guarantee:
running the script twice inserts each chunk at most once.

```python
collection.add(
    ids=[r["id"] for r in new_reviews],
    documents=[r["text"] for r in new_reviews],
    metadatas=[r["metadata"] for r in new_reviews]
)
```
`collection.add` takes three parallel lists. Chroma embeds each item in `documents` and stores
the resulting vectors alongside the `metadatas`. The `ids` must be unique strings.

### Full code block

```python
#!/usr/bin/env python3
import chromadb                                                   # ChromaDB Python client
import json                                                       # reads the JSON seed file
import sys                                                        # used for sys.exit on error

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
SEED_FILE   = "movie_reviews.json"
COLLECTION  = "movie_reviews"

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

try:
    client.heartbeat()
except Exception as exc:
    print(f"Cannot reach Chroma at {CHROMA_HOST}:{CHROMA_PORT}: {exc}")
    sys.exit(1)

collection = client.get_or_create_collection(COLLECTION)

with open(SEED_FILE) as f:
    reviews = json.load(f)

existing = set(collection.get()["ids"])
new_reviews = [r for r in reviews if r["id"] not in existing]

if not new_reviews:
    print(f"✓ Collection '{COLLECTION}' already has all {len(reviews)} chunks — nothing to do.")
    sys.exit(0)

collection.add(
    ids=[r["id"] for r in new_reviews],
    documents=[r["text"] for r in new_reviews],
    metadatas=[r["metadata"] for r in new_reviews]
)

print(f"✓ Seeded {len(new_reviews)} new chunks into '{COLLECTION}' (total: {len(reviews)}).")
```

---

## Cell 27 — Run Seeds

### Purpose
Orchestrates both seed operations in the correct order and verifies the results.

### Line-by-line explanation

```python
r = run("docker compose exec db psql -U movie_user -d movies -c 'SELECT COUNT(*) FROM movies;'")
```
`docker compose exec` runs a command inside a running container. `psql -c` executes a single SQL
statement and exits. `SELECT COUNT(*)` verifies that the `initdb.d/` seed ran. Expects `20`.

```python
r = run("python seed_chroma.py")
if r.returncode != 0:
    print("✗ Chroma seed failed — is Chroma healthy?")
    print("  Run: docker compose ps")
```
Runs the Python seed script. Checks the exit code and provides actionable guidance if it fails.
The most common failure is running this before Chroma passes its healthcheck.

### Full code block

```python
print("=== Seed Orchestration ===")

print("\n[1/2] Postgres seed (via initdb.d — runs automatically on first boot)")
r = run("docker compose exec db psql -U movie_user -d movies -c 'SELECT COUNT(*) FROM movies;'")
print("      ^ If count = 20, Postgres was already seeded via initdb.d.")

print("\n[2/2] Chroma seed...")
r = run("python seed_chroma.py")
if r.returncode != 0:
    print("✗ Chroma seed failed — is Chroma healthy?")
    print("  Run: docker compose ps")
else:
    print("✓ Both datastores seeded.")
```

---

## Cell 29 — Stack Health Check

### Purpose
Programmatically verifies all three API endpoints from within the notebook. Uses Python's
built-in `urllib` to avoid an `httpx` or `requests` dependency in the notebook kernel.

### Line-by-line explanation

```python
def http_get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)
```
Two exception types are handled separately:
- `HTTPError` — the server responded with a 4xx or 5xx. We return the actual HTTP status code so
  the caller can distinguish a 503 (service down) from a 404 (route missing).
- `Exception` — anything else (connection refused, timeout, DNS failure). We return `0` as the
  status code to indicate a network-level failure.

```python
movies = json.loads(body) if status == 200 else []
print(f"  /movies → {status}  ({len(movies)} movies)")
```
Only parses JSON on 200 — parsing a 503 response body as JSON would raise `json.JSONDecodeError`.

### Full code block

```python
import urllib.request
import urllib.error

def http_get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

print("=== Stack Health Check ===")

status, body = http_get("http://localhost:8080/readyz")
print(f"  /readyz        → {status}  {body[:60]}")

status, body = http_get("http://localhost:8080/movies")
movies = json.loads(body) if status == 200 else []
print(f"  /movies        → {status}  ({len(movies)} movies)")

status, body = http_get("http://localhost:8080/search?q=hopeful+prison+drama")
print(f"  /search?q=...  → {status}  {body[:80]}")

print("\nOpen http://localhost:3000 in a browser to test the full UI.")
```

---

## Cell 30 — Partial-Failure Demo

### Purpose
The most important teaching moment in the lab. Demonstrates that `/readyz` correctly reflects
dependency health and that the frontend surfaces the error state when Chroma is down.

### Line-by-line explanation

```python
run("docker compose stop chroma")
time.sleep(5)
```
`docker compose stop` sends `SIGTERM` to the container and waits for it to exit gracefully.
Five seconds lets the Docker healthcheck daemon detect the failure and mark the container
unhealthy — the API's `/readyz` endpoint will now return 503.

```python
status, body = http_get("http://localhost:8080/readyz")
print(f"  /readyz → {status}  {body}")
```
Expected output: `503  {"detail": "Chroma not ready: ..."}`. This is the same response the
frontend receives when it calls the API — it renders this as the red error banner.

```python
run("docker compose start chroma")
print("Waiting 20 s for Chroma to become healthy again...")
time.sleep(20)
```
`start` restarts a stopped container without rebuilding it. The 20-second wait covers:
- Container start time (~1 s)
- Chroma index load time (~5 s)
- `start_period: 15s` healthcheck grace period
- One healthcheck interval (`10s`) after the grace period

```python
status, body = http_get("http://localhost:8080/readyz")
print(f"  /readyz → {status}  {body}")
```
Expected output: `200  {"status": "ok"}`. The API has reconnected to Chroma and all searches
will succeed again.

### Full code block

```python
print("=== Partial-Failure Demo ===")

print("\n[Step 1] Stopping Chroma mid-session...")
run("docker compose stop chroma")
time.sleep(5)

print("[Step 2] Probing /readyz — expect 503...")
status, body = http_get("http://localhost:8080/readyz")
print(f"  /readyz → {status}  {body}")

print("[Step 3] Probing /search — expect 503...")
status, body = http_get("http://localhost:8080/search?q=drama")
print(f"  /search → {status}  {body[:120]}")

print("\n[Step 4] Restarting Chroma...")
run("docker compose start chroma")
print("Waiting 20 s for Chroma to become healthy again...")
time.sleep(20)

print("[Step 5] Probing /readyz — expect 200...")
status, body = http_get("http://localhost:8080/readyz")
print(f"  /readyz → {status}  {body}")

print("\n✓ Partial-failure recovery demonstrated.")
```

---

## Cell 32 — `README.md` Runbook

### Purpose
The operations runbook — the document a new engineer reads to go from zero to a running stack.
Written as a `%%writefile` cell so the act of writing it is itself part of the teaching moment.

### Structure explained

**Prerequisites** — what must be true on the engineer's machine before the first command runs.

**First-time setup** — numbered steps from clone to open-browser. Every step is a shell command;
there is no ambiguity about what to do.

**Day-to-day commands** — a quick-reference table mapping tasks to commands. Covers the 90%
of operations an engineer will need.

**Readiness probe** — the single most important operational command: `curl /readyz`. Engineers
learn to check this *first* when anything seems wrong.

**Common failure scenarios** — each scenario names the symptom, the diagnosis command, and the
fix command. Avoids "check the logs" as a first step — narrows it immediately.

**Committed secrets check** — a one-liner a new engineer can run to verify the repo is clean.
The fix command is included so they do not need to Google it.

### Full code block

*(See the `%%writefile README.md` cell in the notebook — the content is the README itself and
is reproduced in its entirety there.)*

---

## Cell 34 — Shutdown (Last Cell)

### Purpose
Stops and removes all containers and networks created by this Compose file. Named volumes are
preserved by default so seeded data is available for the next session. Explains how to do a
full wipe if needed.

### Line-by-line explanation

```python
result = subprocess.run(
    "docker compose down --remove-orphans",
    shell=True,
    capture_output=True,
    text=True
)
```
`docker compose down` stops containers and removes the containers and network. `--remove-orphans`
also removes containers for services that have been removed from the Compose file since the last
`up` — keeps the Docker environment clean.

```python
print(result.stdout)
if result.returncode == 0:
    print("✓ All containers stopped and removed.")
else:
    print("✗ Shutdown error:")
    print(result.stderr)
```
Prints the shutdown output and checks the exit code. A failed shutdown (rare but possible if
a container has zombie processes) is surfaced immediately.

```python
print("""
Named volumes (pgdata, chromadata) are PRESERVED.
To also wipe all data and start fresh next time, run:

    docker compose down -v --remove-orphans

WARNING: -v deletes all seeded data. You will need to re-run
         seed_chroma.py after the next 'docker compose up'.
""")
```
Explains volume retention so students understand why their data is available after restarting the
stack. The `-v` flag warning is important — accidental use resets all seed data and requires
re-running `seed_chroma.py` (Postgres is automatically re-seeded by `initdb.d/` on the next boot).

### Full code block

```python
print("Stopping and removing all lab containers and networks...")  # informs user shutdown is starting

result = subprocess.run(                                            # runs docker compose down
    "docker compose down --remove-orphans",                        # stops containers, removes network; keeps volumes
    shell=True,
    capture_output=True,
    text=True
)

print(result.stdout)

if result.returncode == 0:
    print("✓ All containers stopped and removed.")
else:
    print("✗ Shutdown error:")
    print(result.stderr)

print("""
Named volumes (pgdata, chromadata) are PRESERVED.
To also wipe all data and start fresh next time, run:

    docker compose down -v --remove-orphans

WARNING: -v deletes all seeded data. You will need to re-run
         seed_chroma.py after the next 'docker compose up'.
""")
```

---

*End of walkthrough. Total cells covered: 24 code cells across 10 beats.*
