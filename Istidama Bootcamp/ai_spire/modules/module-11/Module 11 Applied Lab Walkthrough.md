# Module 11 Applied Lab — Complete Code Walkthrough

Every line of code in every notebook cell is explained below.  
Each section ends with the **full code block** exactly as it appears in the notebook.

---

## Table of Contents

1. [Cell 02 — Prerequisites Check](#cell-02--prerequisites-check)
2. [Cell 03 — `requirements.txt` v1 (skeleton)](#cell-03--requirementstxt-v1-skeleton)
3. [Cell 04 — `app.py` v1 (route only)](#cell-04--apppy-v1-route-only)
4. [Cell 05 — Start Server + Smoke Test](#cell-05--start-server--smoke-test)
5. [Cell 07 — `requirements.txt` v2 (+ prometheus_client)](#cell-07--requirementstxt-v2--prometheus_client)
6. [Cell 08 — `app.py` v2 (metric declarations)](#cell-08--apppy-v2-metric-declarations)
7. [Cell 09 — Duplicate-Registration Error Demo](#cell-09--duplicate-registration-error-demo)
8. [Cell 11 — `app.py` v3 (+ `/metrics` mount)](#cell-11--apppy-v3---metrics-mount)
9. [Cell 12 — Restart + Read `/metrics`](#cell-12--restart--read-metrics)
10. [Cell 14 — `app.py` v4 (+ request-ID middleware)](#cell-14--apppy-v4--request-id-middleware)
11. [Cell 15 — curl -i Request-ID Demo](#cell-15--curl--i-request-id-demo)
12. [Cell 17 — `app.py` v5 (+ structured logging)](#cell-17--apppy-v5--structured-logging)
13. [Cell 18 — Logging Demo](#cell-18--logging-demo)
14. [Cell 20 — `app.py` v6 (full — + metrics middleware)](#cell-20--apppy-v6-full---metrics-middleware)
15. [Cell 21 — Metrics Demo](#cell-21--metrics-demo)
16. [Cell 22 — Cardinality Pitfall Demo](#cell-22--cardinality-pitfall-demo)
17. [Cell 24 — `read_metrics.py`](#cell-24--read_metricspy)
18. [Cell 25 — Run `read_metrics.py`](#cell-25--run-read_metricspy)
19. [Cell 26 — Shutdown](#cell-26--shutdown)

---

## Cell 02 — Prerequisites Check

### Purpose
Verifies that all required Python packages are installed before any code runs.
Also creates the `tests/` directory that will hold the pytest suite written during the session.

### Line-by-line explanation

```python
import subprocess
```
Python's standard library module for launching child processes and capturing their output.
Every `python -m pip` and `uvicorn` call in this notebook runs through it.

```python
import sys
```
Gives us `sys.executable` — the exact Python binary that is running this notebook.
Using `sys.executable` guarantees we install into the same environment the notebook kernel uses,
not some other Python on the `PATH`.

```python
import os
```
Used for `os.getcwd()` (current working directory) and `os.makedirs()`.

```python
import time
```
`time.sleep()` is used after starting `uvicorn` to give it enough time to bind the port before we
send HTTP requests.

```python
import json
```
Used later in the logging demo to parse the JSON log line emitted by the server.

```python
LAB_DIR = os.getcwd()
```
Records the directory where the notebook lives. Later cells use it as the working directory when
launching `uvicorn` so relative paths like `app:app` resolve correctly.

```python
os.makedirs("tests", exist_ok=True)
```
Creates `tests/` alongside `app.py`. `exist_ok=True` means this is safe to run more than once —
it will not raise an error if the directory already exists.

```python
required = ["fastapi", "uvicorn", "prometheus_client", "httpx"]
```
A simple list of the four packages we depend on. Checking them individually gives clearer output
than letting a single install fail with a cryptic message.

```python
for pkg in required:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", pkg],
        capture_output=True,
    )
    status = "✓" if result.returncode == 0 else "✗ MISSING"
    print(f"  {status}  {pkg}")
```
`pip show <pkg>` exits with code 0 if the package is installed, non-zero otherwise.
`capture_output=True` suppresses the `pip show` output — we only care about the exit code.
The ternary builds a one-character status indicator.

```python
subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "fastapi==0.111.0", "uvicorn[standard]==0.30.1",
     "prometheus_client==0.20.0", "httpx==0.27.0",
     "--quiet"],
    check=True,
)
```
Installs all four dependencies in one call. Pinned versions ensure every learner runs identical
code. `--quiet` suppresses the download progress bar. `check=True` raises `CalledProcessError` if
`pip` exits non-zero, making failures visible immediately.

### Full code block

```python
import subprocess                                                # runs shell commands from Python
import sys                                                       # access to interpreter info
import os                                                        # OS-level file/directory operations
import time                                                      # wall-clock timing utilities
import json                                                      # JSON serialization

LAB_DIR = os.getcwd()                                           # current working directory = lab root

os.makedirs("tests", exist_ok=True)                             # create tests/ directory (idempotent)

required = ["fastapi", "uvicorn", "prometheus_client", "httpx"] # packages needed for this lab

print("Checking installed packages…")
for pkg in required:                                            # iterate over each required package
    result = subprocess.run(                                    # attempt to locate the package
        [sys.executable, "-m", "pip", "show", pkg],            # pip show exits 0 if installed
        capture_output=True,                                    # suppress output during check
    )
    status = "✓" if result.returncode == 0 else "✗ MISSING"   # ✓ found  ✗ needs install
    print(f"  {status}  {pkg}")

print("\nInstalling / upgrading packages…")
subprocess.run(                                                 # install all dependencies at once
    [sys.executable, "-m", "pip", "install",
     "fastapi==0.111.0",
     "uvicorn[standard]==0.30.1",
     "prometheus_client==0.20.0",
     "httpx==0.27.0",
     "--quiet"],                                                # suppress verbose pip output
    check=True,                                                 # raise error if install fails
)
print("All packages ready.")                                    # confirm successful install
```

---

## Cell 03 — `requirements.txt` v1 (skeleton)

### Purpose
Creates the requirements file with just the web server dependencies — before we add
`prometheus_client`. The `%%writefile` Jupyter magic writes the cell body directly to disk.

### Line-by-line explanation

```
fastapi==0.111.0
```
ASGI web framework. Provides the `FastAPI` class, route decorators (`@app.get`), and automatic
OpenAPI documentation at `/docs`.

```
uvicorn[standard]==0.30.1
```
The ASGI server that runs `app.py`. The `[standard]` extra adds `watchfiles` (used by `--reload`)
and `websockets`.

### Full code block

```
%%writefile requirements.txt
fastapi==0.111.0            # ASGI web framework with automatic OpenAPI docs
uvicorn[standard]==0.30.1   # ASGI server; [standard] adds websocket + watchfiles reload
```

---

## Cell 04 — `app.py` v1 (route only)

### Purpose
The skeleton application: a `FastAPI` instance, a five-city in-memory dict, and a single `GET /weather` route.
No Prometheus, no middleware. This is the target the session instruments.

### Line-by-line explanation

```python
from fastapi import FastAPI
```
Imports the `FastAPI` class. One instance of this class *is* the application.

```python
from fastapi.responses import JSONResponse
```
Imported for the 404 path. Using `JSONResponse` with an explicit `status_code` is clearer than
raising an `HTTPException` when we want full control over the response body.

```python
app = FastAPI(title="Weather API")
```
Creates the ASGI application. The `title` appears in the auto-generated `/docs` Swagger UI.

```python
FORECASTS = {
    "Boston":    "Partly cloudy, 18 °C",
    ...
}
```
A module-level dict acts as the in-memory data store. The keys are city names exactly as they
will arrive in the `?city=` query parameter. No database, no external API — keeps the demo
focused on instrumentation, not I/O.

```python
@app.get("/weather")
def get_weather(city: str):
```
Registers a `GET /weather` handler. FastAPI reads the function signature: `city: str` with no
default value means it is a *required* query parameter. A missing `?city=` will return a 422
automatically before this function runs.

```python
    forecast = FORECASTS.get(city)
```
`dict.get()` returns `None` for an unknown key instead of raising `KeyError`.

```python
    if forecast is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No forecast for '{city}'"},
        )
```
Returns a structured 404. The `content` dict is serialised to JSON automatically by `JSONResponse`.

```python
    return {"city": city, "forecast": forecast}
```
FastAPI serialises a plain `dict` to a 200 JSON response. The schema is inferred automatically.

### Full code block

```python
%%writefile app.py
"""Weather API — Beat 1 skeleton: route only, no instrumentation."""
from fastapi import FastAPI                                      # FastAPI application class
from fastapi.responses import JSONResponse                       # helper for explicit JSON responses

app = FastAPI(title="Weather API")                              # create the ASGI application instance

# ── In-memory data store ──────────────────────────────────────────────────────
FORECASTS = {                                                    # dict: city name → forecast string
    "Boston":    "Partly cloudy, 18 °C",                        # northeast US city
    "Dubai":     "Sunny, 41 °C",                                # UAE desert city
    "London":    "Overcast, 12 °C",                             # UK capital
    "New York":  "Thunderstorms, 22 °C",                        # northeast US metro
    "Tokyo":     "Clear, 26 °C",                                # Japanese capital
}

# ── Route ─────────────────────────────────────────────────────────────────────
@app.get("/weather")                                             # registers a GET handler at /weather
def get_weather(city: str):                                      # city is a required query param (?city=…)
    forecast = FORECASTS.get(city)                              # look up city; None if not found
    if forecast is None:                                        # city absent from the data store
        return JSONResponse(                                    # return a 404 JSON error body
            status_code=404,
            content={"error": f"No forecast for '{city}'"},
        )
    return {"city": city, "forecast": forecast}                 # happy path: city + forecast as JSON
```

---

## Cell 05 — Start Server + Smoke Test

### Purpose
Starts `uvicorn` as a background process and makes two test requests (one 200, one 404) to
confirm the skeleton is working before we add any instrumentation.

### Line-by-line explanation

```python
subprocess.run(["pkill", "-f", "uvicorn app:app"], capture_output=True)
```
Kills any previously running `uvicorn app:app` process. `-f` matches the full command string, not
just the process name. `capture_output=True` silences the "no process found" error if nothing
was running. This line makes the cell idempotent — safe to re-run.

```python
time.sleep(1)
```
Gives the OS time to release port 8000 after the previous process exits before we try to bind it
again.

```python
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "app:app", "--port", "8000", "--log-level", "warning"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
```
`Popen` (as opposed to `run`) is non-blocking — it starts the process and returns immediately,
letting the notebook cell continue. `stdout=subprocess.PIPE` captures the server's output so we
can read it later (Beat 5). `stderr=subprocess.STDOUT` merges stderr into the same stream.
`--log-level warning` suppresses uvicorn's normal INFO boot messages so they don't pollute cell
output.

```python
time.sleep(2)
```
Gives uvicorn 2 seconds to bind the socket and load the app module. The request below will
`ConnectionRefusedError` if we don't wait.

```python
resp = httpx.get("http://localhost:8000/weather?city=Boston", timeout=5)
```
`httpx` is a modern `requests`-compatible HTTP client with async support. `timeout=5` means the
call raises `httpx.TimeoutException` after 5 seconds — fast feedback if the server isn't running.

```python
resp404 = httpx.get("http://localhost:8000/weather?city=Atlantis", timeout=5)
```
Exercises the 404 branch. "Atlantis" is not in `FORECASTS`.

### Full code block

```python
import subprocess, time, httpx, os

# ── Kill any running server from a previous cell ──────────────────────────────
subprocess.run(["pkill", "-f", "uvicorn app:app"],              # send SIGTERM to any uvicorn process
               capture_output=True)
time.sleep(1)                                                   # brief pause to free port 8000

# ── Start uvicorn in the background ──────────────────────────────────────────
server = subprocess.Popen(                                      # Popen = non-blocking process launch
    ["python", "-m", "uvicorn", "app:app",                     # run uvicorn serving our app module
     "--port", "8000",                                          # bind to port 8000
     "--log-level", "warning"],                                 # suppress info-level boot chatter
    stdout=subprocess.PIPE,                                     # capture stdout
    stderr=subprocess.STDOUT,                                   # merge stderr → stdout stream
)
time.sleep(2)                                                   # wait for uvicorn to finish starting

# ── Smoke test ────────────────────────────────────────────────────────────────
resp = httpx.get("http://localhost:8000/weather?city=Boston",   # hit the weather endpoint
                  timeout=5)                                    # fail fast if server is down
print(f"Status : {resp.status_code}")                          # should print 200
print(f"Body   : {resp.json()}")                               # should print city + forecast dict

resp404 = httpx.get("http://localhost:8000/weather?city=Atlantis",  # unknown city → 404
                     timeout=5)
print(f"\n404 test — Status : {resp404.status_code}")          # should print 404
print(f"404 test — Body   : {resp404.json()}")                 # should print error dict
```

---

## Cell 07 — `requirements.txt` v2 (+ prometheus_client)

### Purpose
Adds `prometheus_client` to the requirements file. This pin matches the version used in the
integration lab so the parser API is identical.

### Full code block

```
%%writefile requirements.txt
fastapi==0.111.0            # ASGI web framework with automatic OpenAPI docs
uvicorn[standard]==0.30.1   # ASGI server; [standard] adds websocket + watchfiles reload
prometheus_client==0.20.0   # Prometheus instrumentation library for Python
```

---

## Cell 08 — `app.py` v2 (metric declarations)

### Purpose
Adds three metric objects at module scope. This beat's entire lesson is *where* these lines go:
outside any function, at the top level of the module, executed exactly once when Python imports
`app.py`.

### Line-by-line explanation

```python
from prometheus_client import Counter, Histogram, Gauge
```
Imports the three metric type classes. `Counter`, `Histogram`, and `Gauge` each create an object
that registers itself in the default `CollectorRegistry` on construction.

```python
REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total HTTP requests",
    ["path", "status"],
)
```
`Counter` is a monotonically increasing number. It can only go up (and reset to zero when the
process restarts). The string `"requests_total"` is the metric name that appears in the
`/metrics` output. The list `["path", "status"]` defines two label *dimensions*. Every unique
combination of label values becomes a separate time series in Prometheus.

`path` will be values like `"/weather"` or `"/metrics"` — bounded (we know all our routes).
`status` will be `"200"`, `"404"`, `"500"` — bounded (HTTP status codes are a closed set).

```python
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency in seconds",
    ["path"],
)
```
`Histogram` records the *distribution* of a measured value across pre-defined buckets (default:
`.005`, `.01`, `.025`, `.05`, `.1`, `.25`, `.5`, `1`, `2.5`, `5`, `10` seconds). For each
observation it increments the count of every bucket whose upper bound is ≥ the observed value.
This enables quantile queries like `histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))`.

Only `path` is a label here — not `status`. Adding `status` would multiply the number of bucket
series by the number of distinct status codes, which is wasteful for a timing metric.

```python
INFLIGHT_REQUESTS = Gauge(
    "inflight_requests",
    "Number of requests currently in-flight",
)
```
`Gauge` is a value that can go up *and* down. We increment it at the start of each request and
decrement it at the end. At any moment it tells you how many requests are currently being
processed. No labels — one global count is the right granularity for concurrent load.

### Full code block

```python
%%writefile app.py
"""Weather API — Beat 2: metric declarations added at module scope."""
from fastapi import FastAPI                                      # FastAPI application class
from fastapi.responses import JSONResponse                       # helper for explicit JSON responses
from prometheus_client import Counter, Histogram, Gauge          # the three metric type classes we need

app = FastAPI(title="Weather API")                              # create the ASGI application instance

# ── In-memory data store ──────────────────────────────────────────────────────
FORECASTS = {                                                    # dict: city name → forecast string
    "Boston":    "Partly cloudy, 18 °C",
    "Dubai":     "Sunny, 41 °C",
    "London":    "Overcast, 12 °C",
    "New York":  "Thunderstorms, 22 °C",
    "Tokyo":     "Clear, 26 °C",
}

# ── Metric declarations (module scope — created once at import time) ──────────
REQUESTS_TOTAL = Counter(                                        # Counter: monotonically increasing
    "requests_total",                                            # metric name (appears in /metrics)
    "Total HTTP requests",                                       # description → # HELP line
    ["path", "status"],                                          # label names (bounded: paths + codes)
)

REQUEST_LATENCY = Histogram(                                     # Histogram: bucketed duration samples
    "request_latency_seconds",                                   # metric name
    "HTTP request latency in seconds",                           # description
    ["path"],                                                    # one label: path (not status — too many buckets)
)

INFLIGHT_REQUESTS = Gauge(                                       # Gauge: value goes up and down
    "inflight_requests",                                         # metric name
    "Number of requests currently in-flight",                    # description
)                                                                # no labels — one global count is enough

# ── Route ─────────────────────────────────────────────────────────────────────
@app.get("/weather")                                             # GET /weather handler
def get_weather(city: str):                                      # city query parameter
    forecast = FORECASTS.get(city)
    if forecast is None:
        return JSONResponse(status_code=404, content={"error": f"No forecast for '{city}'"})
    return {"city": city, "forecast": forecast}
```

---

## Cell 09 — Duplicate-Registration Error Demo

### Purpose
Shows the error you get when a metric is declared inside a function and that function is called
twice. The room sees the real exception message, not just a description of it.

### Line-by-line explanation

```python
def bad_handler():
    c = Counter("requests_total", "oops", ["path", "status"])
```
On the *first* call, `Counter(...)` succeeds and registers `requests_total` in the default
registry. On the *second* call, it tries to register the same name again with the same labels
and raises `ValueError: Duplicated timeseries in CollectorRegistry`.

```python
try:
    bad_handler()   # first call registers
    bad_handler()   # second call collides → ValueError
except ValueError as e:
    print(f"ValueError caught: {e}")
```
We catch the error so the cell doesn't terminate with a red traceback — the lesson is the
message, not the failure.

> **Why this matters in production:** Every HTTP request triggers the handler, not just every
> restart. The error surfaces on the *second* request, which means your service crashes in
> production after exactly one successful request.

### Full code block

```python
# ── Demonstrate the duplicate-registration error ──────────────────────────────
# Run this cell to see what happens when you declare a metric inside a function.
# Then read the error, and notice we already declared REQUESTS_TOTAL at module scope above.

from prometheus_client import Counter

def bad_handler():
    c = Counter("requests_total", "oops", ["path", "status"])  # re-registering an existing name
    c.labels(path="/weather", status="200").inc()

try:
    bad_handler()                                               # first call registers
    bad_handler()                                               # second call collides → ValueError
except ValueError as e:
    print(f"ValueError caught: {e}")                           # shows: Duplicated timeseries in CollectorRegistry
print("\n→ Always declare metrics at module scope, not inside request handlers.")
```

---

## Cell 11 — `app.py` v3 (+ `/metrics` mount)

### Purpose
Wires the Prometheus exporter endpoint. After this cell, `GET /metrics` returns the OpenMetrics
text format — even though no middleware is populating the counter or histogram yet.

### Line-by-line explanation

```python
from prometheus_client import (
    Counter, Histogram, Gauge,
    make_asgi_app,
)
```
`make_asgi_app()` is added to the import. It is a factory function that returns a small ASGI
application wrapping the default `CollectorRegistry`.

```python
metrics_app = make_asgi_app()
```
Builds the ASGI sub-application. At this point it is just an object — not yet served anywhere.

```python
app.mount("/metrics", metrics_app)
```
`FastAPI.mount()` splices an ASGI sub-application into the router at the given prefix. Any
request whose path starts with `/metrics` is dispatched to `metrics_app` rather than the normal
FastAPI route table. The `make_asgi_app()` sub-app handles `GET /metrics` and returns the full
registry serialised as OpenMetrics text.

> **Key insight:** `/metrics` must be mounted *after* the `FastAPI` instance is created but
> *before* any request is served. Placement in the file matters — module-level code runs top to
> bottom at import time.

### Full code block

```python
%%writefile app.py
"""Weather API — Beat 3: /metrics endpoint mounted."""
from fastapi import FastAPI                                      # FastAPI application class
from fastapi.responses import JSONResponse                       # helper for explicit JSON responses
from prometheus_client import (                                  # import metric types + ASGI factory
    Counter,                                                     # monotonically increasing counter
    Histogram,                                                   # bucketed duration distribution
    Gauge,                                                       # point-in-time mutable value
    make_asgi_app,                                               # factory: Prometheus registry → ASGI app
)

app = FastAPI(title="Weather API")                              # create the ASGI application instance

FORECASTS = {
    "Boston":    "Partly cloudy, 18 °C",
    "Dubai":     "Sunny, 41 °C",
    "London":    "Overcast, 12 °C",
    "New York":  "Thunderstorms, 22 °C",
    "Tokyo":     "Clear, 26 °C",
}

REQUESTS_TOTAL = Counter(
    "requests_total", "Total HTTP requests", ["path", "status"],
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "HTTP request latency in seconds", ["path"],
)
INFLIGHT_REQUESTS = Gauge(
    "inflight_requests", "Number of requests currently in-flight",
)

# ── Mount /metrics endpoint ───────────────────────────────────────────────────
metrics_app = make_asgi_app()                                    # build Prometheus ASGI sub-application
app.mount("/metrics", metrics_app)                               # attach exporter at /metrics path

# ── Route ─────────────────────────────────────────────────────────────────────
@app.get("/weather")
def get_weather(city: str):
    forecast = FORECASTS.get(city)
    if forecast is None:
        return JSONResponse(status_code=404, content={"error": f"No forecast for '{city}'"})
    return {"city": city, "forecast": forecast}
```

---

## Cell 12 — Restart + Read `/metrics`

### Purpose
Restarts the server with the new `app.py` and fetches `/metrics` to confirm the contract surface
(HELP + TYPE lines) is visible before any middleware starts populating values.

### Line-by-line explanation

```python
for line in metrics_text.splitlines():
    if line.startswith("#") or line.startswith("inflight"):
        print(line)
```
Filters the full OpenMetrics output to just the comment lines (`# HELP`, `# TYPE`) and the
`inflight_requests` gauge sample (which initialises to `0.0`). The counter and histogram have no
sample lines yet — they only appear once at least one label combination has been observed.

### Full code block

```python
import subprocess, time, httpx

subprocess.run(["pkill", "-f", "uvicorn app:app"], capture_output=True)  # stop previous server
time.sleep(1)

server = subprocess.Popen(                                      # start fresh server process
    ["python", "-m", "uvicorn", "app:app", "--port", "8000",
     "--log-level", "warning"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
time.sleep(2)                                                   # wait for startup

# ── Read /metrics output ──────────────────────────────────────────────────────
metrics_text = httpx.get("http://localhost:8000/metrics",       # fetch the Prometheus exporter output
                          timeout=5).text
# Print just the # HELP / # TYPE lines + the inflight_requests sample
for line in metrics_text.splitlines():                          # iterate over each output line
    if line.startswith("#") or line.startswith("inflight"):     # keep HELP/TYPE lines + gauge sample
        print(line)
```

---

## Cell 14 — `app.py` v4 (+ request-ID middleware)

### Purpose
Adds the request-ID `ContextVar` and the first middleware function. Every request now gets a
UUID stamped on it and returned in `X-Request-ID`.

### Line-by-line explanation

```python
from contextvars import ContextVar
```
`ContextVar` is Python's async-safe per-task storage. Unlike thread-local variables, a
`ContextVar` is isolated to a single `asyncio` task — which corresponds to a single HTTP request
in an ASGI server. Two concurrent requests cannot read each other's `request_id`.

```python
import uuid
```
Standard library module for generating Universally Unique Identifiers.

```python
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
```
Declares the `ContextVar` at module scope. The type annotation `ContextVar[str]` tells type
checkers what `.get()` returns. The `default=""` means `.get()` returns an empty string outside
a request context (e.g. in background tasks or tests that do not set it).

```python
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
```
`@app.middleware("http")` registers this coroutine as an HTTP middleware function. FastAPI calls
it for every incoming request. The `call_next` argument is a callable that forwards the request
to the next layer (another middleware or the route handler).

```python
    rid = uuid.uuid4().hex
```
`uuid4()` generates a random 128-bit UUID. `.hex` returns it as a 32-character lowercase
hexadecimal string with no hyphens, e.g. `a3f8c1e9d2b7405694e1fbc073d20081`.

```python
    request_id_var.set(rid)
```
Binds `rid` to the current async task's copy of `request_id_var`. Any code running inside this
request — including the route handler and other middleware — can call `request_id_var.get()` to
retrieve it.

```python
    response = await call_next(request)
```
Suspends this coroutine and hands control to the next middleware or route handler. The `await`
is necessary because the inner layers are also async.

```python
    response.headers["X-Request-ID"] = rid
```
Attaches the ID to the *outgoing* response header after the inner layers have finished. This
means the client gets the same ID that was stored in the logs.

### Full code block

```python
%%writefile app.py
"""Weather API — Beat 4: request-ID ContextVar + middleware added."""
from fastapi import FastAPI, Request                             # Request gives us headers, url, etc.
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from contextvars import ContextVar                               # async-safe per-task storage
import uuid                                                      # generates universally unique IDs

app = FastAPI(title="Weather API")

FORECASTS = {
    "Boston":    "Partly cloudy, 18 °C",
    "Dubai":     "Sunny, 41 °C",
    "London":    "Overcast, 12 °C",
    "New York":  "Thunderstorms, 22 °C",
    "Tokyo":     "Clear, 26 °C",
}

REQUESTS_TOTAL = Counter("requests_total", "Total HTTP requests", ["path", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "HTTP request latency in seconds", ["path"])
INFLIGHT_REQUESTS = Gauge("inflight_requests", "Number of requests currently in-flight")

# ── Request-ID context storage ────────────────────────────────────────────────
request_id_var: ContextVar[str] = ContextVar(                   # declare module-level ContextVar
    "request_id", default=""                                    # default empty string when no request
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")                                          # registers an HTTP middleware function
async def request_id_middleware(request: Request, call_next):   # called for every incoming request
    rid = uuid.uuid4().hex                                      # generate 32-char hex unique ID
    request_id_var.set(rid)                                     # store in ContextVar for this async task
    response = await call_next(request)                         # forward request to next layer
    response.headers["X-Request-ID"] = rid                      # attach ID to outgoing response header
    return response                                             # return completed response to client

# ── Route ─────────────────────────────────────────────────────────────────────
@app.get("/weather")
def get_weather(city: str):
    forecast = FORECASTS.get(city)
    if forecast is None:
        return JSONResponse(status_code=404, content={"error": f"No forecast for '{city}'"})
    return {"city": city, "forecast": forecast}
```

---

## Cell 15 — curl -i Request-ID Demo

### Purpose
Shows the `X-Request-ID` header in actual HTTP responses and demonstrates that each request
produces a different ID.

### Line-by-line explanation

```python
for name, value in resp.headers.items():
    if "request-id" in name.lower() or "content" in name.lower():
        print(f"  {name}: {value}")
```
Filters the response headers to just the ones we care about: `x-request-id` and `content-type`.
Header names in `httpx` are lower-cased automatically.

```python
for _ in range(2):
    r = httpx.get("http://localhost:8000/weather?city=Dubai", timeout=5)
    print(f"  {r.headers.get('x-request-id')}")
```
Makes two more requests to the same endpoint and prints their IDs. The point: same endpoint, same
city, but a different 32-char hex ID every time — proving the UUID is generated fresh per request.

### Full code block

```python
import subprocess, time, httpx

subprocess.run(["pkill", "-f", "uvicorn app:app"], capture_output=True)
time.sleep(1)
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "app:app", "--port", "8000", "--log-level", "warning"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
time.sleep(2)

# ── Show the X-Request-ID response header ─────────────────────────────────────
resp = httpx.get("http://localhost:8000/weather?city=Tokyo", timeout=5)  # make a request
print("Response headers:")
for name, value in resp.headers.items():                        # iterate over all response headers
    if "request-id" in name.lower() or "content" in name.lower():  # filter to relevant headers
        print(f"  {name}: {value}")

print(f"\nRequest ID: {resp.headers.get('x-request-id')}")    # pull the X-Request-ID header value
print("\nMake two more requests — each gets a unique ID:")
for _ in range(2):
    r = httpx.get("http://localhost:8000/weather?city=Dubai", timeout=5)
    print(f"  {r.headers.get('x-request-id')}")                # each hex string should be different
```

---

## Cell 17 — `app.py` v5 (+ structured logging)

### Purpose
Adds a JSON log formatter and a logging middleware. After this beat, every request emits one
machine-readable JSON line to stderr.

### Line-by-line explanation

```python
import logging
```
Python's standard library logging framework. We subclass its `Formatter` rather than write our
own output function so the result integrates with any standard log handler (file, syslog, etc.).

```python
import json
```
`json.dumps()` serialises the log record dict to a single-line JSON string.

```python
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
```
Subclasses `logging.Formatter` and overrides `format()`. The `LogRecord` argument carries the
log level, message, and any extra fields injected by the caller.

```python
        return json.dumps({
            "ts":         self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":      record.levelname,
            "request_id": request_id_var.get(),
            "path":       getattr(record, "path", ""),
            "status":     getattr(record, "status", ""),
            "latency_ms": getattr(record, "latency_ms", ""),
            "msg":        record.getMessage(),
        })
```
`self.formatTime(...)` formats the timestamp using `time.strftime`. `request_id_var.get()`
reads the current request's ID from the `ContextVar` — it is still set at the moment
`format()` is called because the logging call happens before the middleware returns.
`getattr(record, "path", "")` reads a field injected via the `extra={}` dict in `logger.info()`;
the default `""` prevents `AttributeError` on log calls that do not inject those fields.

```python
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("weather")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
```
- `StreamHandler` writes to `stderr` by default.
- `setFormatter` wires our JSON formatter to the handler.
- `getLogger("weather")` creates (or retrieves) a named logger. Named loggers are isolated from
  the root logger.
- `logger.propagate = False` prevents the record from travelling up to the root logger and being
  printed a second time in the default format.

```python
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("request", extra={
        "path":       request.url.path,
        "status":     response.status_code,
        "latency_ms": round(latency_ms, 2),
    })
    return response
```
`time.perf_counter()` returns a float in seconds with nanosecond resolution — more accurate than
`time.time()` for sub-millisecond measurements. The `extra={}` dict merges its keys into the
`LogRecord` object, where the formatter reads them with `getattr`. Multiplying by `1000` converts
seconds to milliseconds.

### Full code block

```python
%%writefile app.py
"""Weather API — Beat 5: structured JSON logging middleware added."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from contextvars import ContextVar
import uuid                                                      # generates universally unique IDs
import logging                                                   # Python standard logging framework
import json                                                      # JSON serialization
import time                                                      # wall-clock timing

app = FastAPI(title="Weather API")

FORECASTS = {
    "Boston":    "Partly cloudy, 18 °C",
    "Dubai":     "Sunny, 41 °C",
    "London":    "Overcast, 12 °C",
    "New York":  "Thunderstorms, 22 °C",
    "Tokyo":     "Clear, 26 °C",
}

REQUESTS_TOTAL = Counter("requests_total", "Total HTTP requests", ["path", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "HTTP request latency in seconds", ["path"])
INFLIGHT_REQUESTS = Gauge("inflight_requests", "Number of requests currently in-flight")

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# ── Structured JSON log formatter ─────────────────────────────────────────────
class JSONFormatter(logging.Formatter):                          # subclass standard Formatter
    def format(self, record: logging.LogRecord) -> str:         # override to emit JSON instead of text
        return json.dumps({                                      # serialize all fields to one JSON line
            "ts":         self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),  # ISO 8601 timestamp
            "level":      record.levelname,                     # DEBUG / INFO / WARNING / ERROR
            "request_id": request_id_var.get(),                 # pull ID from ContextVar (async-safe)
            "path":       getattr(record, "path", ""),          # URL path — injected via extra={}
            "status":     getattr(record, "status", ""),        # HTTP status — injected via extra={}
            "latency_ms": getattr(record, "latency_ms", ""),    # latency in ms — injected via extra={}
            "msg":        record.getMessage(),                  # the string passed to logger.info(…)
        })

handler = logging.StreamHandler()                               # writes log records to stderr
handler.setFormatter(JSONFormatter())                           # attach our JSON formatter to it
logger = logging.getLogger("weather")                           # named logger for this application
logger.addHandler(handler)                                      # wire the handler to the logger
logger.setLevel(logging.INFO)                                   # INFO and above; suppress DEBUG noise
logger.propagate = False                                        # prevent duplicate lines in root logger

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = uuid.uuid4().hex
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

@app.middleware("http")                                          # second middleware registration
async def logging_middleware(request: Request, call_next):      # emits one JSON log line per request
    start = time.perf_counter()                                 # capture start time for latency calc
    response = await call_next(request)                         # pass request through inner layers
    latency_ms = (time.perf_counter() - start) * 1000          # convert elapsed seconds → ms
    logger.info(                                                # emit at INFO level
        "request",                                              # the msg field in the JSON line
        extra={                                                 # extra dict injects fields into LogRecord
            "path":       request.url.path,                    # URL path (e.g. /weather)
            "status":     response.status_code,                # HTTP status code integer
            "latency_ms": round(latency_ms, 2),                # latency rounded to 2 decimal places
        },
    )
    return response                                             # pass response back up the chain

# ── Route ─────────────────────────────────────────────────────────────────────
@app.get("/weather")
def get_weather(city: str):
    forecast = FORECASTS.get(city)
    if forecast is None:
        return JSONResponse(status_code=404, content={"error": f"No forecast for '{city}'"})
    return {"city": city, "forecast": forecast}
```

---

## Cell 18 — Logging Demo

### Purpose
Restarts the server and reads one JSON log line from its stdout pipe to show the structured
format live.

### Line-by-line explanation

```python
raw_line = server.stdout.readline().decode("utf-8", errors="replace").strip()
```
`readline()` blocks until there is a line in the pipe buffer. The server emits a JSON log line
on every request, so after `httpx.get(...)` triggers a request, this readline gets the log
output. `.decode(...)` converts bytes to a string; `errors="replace"` avoids crashes on
unexpected characters. `.strip()` removes the trailing newline.

```python
log_entry = json.loads(raw_line)
for k, v in log_entry.items():
    print(f"  {k:12s}: {v}")
```
Parses the JSON and pretty-prints each field with aligned column widths. The `:12s` format
specifier left-pads the key to 12 characters.

### Full code block

```python
import subprocess, time, httpx, json

subprocess.run(["pkill", "-f", "uvicorn app:app"], capture_output=True)
time.sleep(1)
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "app:app", "--port", "8000", "--log-level", "warning"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
time.sleep(2)

# ── Make a request and read the log line from server stderr ──────────────────
resp = httpx.get("http://localhost:8000/weather?city=London", timeout=5)  # trigger logging
time.sleep(0.3)                                                 # give server time to flush log line

# Read one line of output from the server process stdout (merged stderr)
raw_line = server.stdout.readline().decode("utf-8", errors="replace").strip()  # read server output

try:
    log_entry = json.loads(raw_line)                            # parse JSON log line
    print("Parsed log entry:")
    for k, v in log_entry.items():                             # pretty-print each field
        print(f"  {k:12s}: {v}")
except json.JSONDecodeError:
    print(f"Raw output: {raw_line}")                           # fallback if uvicorn emitted non-JSON
    print("(Tip: the JSON line may be on a subsequent read — run the cell again)")
```

---

## Cell 20 — `app.py` v6 (full — + metrics middleware)

### Purpose
Adds the metrics middleware, completing the full instrumented application. This is the final
version of `app.py`.

### Line-by-line explanation (metrics_middleware only — all other pieces covered above)

```python
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    INFLIGHT_REQUESTS.inc()
```
First thing: increment the in-flight gauge. This must happen *before* `await call_next` so the
gauge accurately reflects that this request is now being processed.

```python
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
```
Brackets the route handler execution. `perf_counter()` has sub-microsecond resolution.

```python
    INFLIGHT_REQUESTS.dec()
```
Decrements *after* the response is ready. If the route handler raises an exception, the gauge
would not be decremented — in production you would wrap this in a `try/finally` block.

```python
    path   = request.url.path
    status = str(response.status_code)
    REQUESTS_TOTAL.labels(path=path, status=status).inc()
```
`.labels(...)` returns a child counter for this specific label combination. `.inc()` increments
it by 1. Prometheus creates the child time series on first use.

```python
    REQUEST_LATENCY.labels(path=path).observe(latency)
```
`.observe(value)` records one measurement. The histogram distributes it across the pre-defined
buckets automatically.

### Middleware registration order

FastAPI processes middleware in reverse registration order (last-registered = outermost wrapper).
The three `@app.middleware("http")` decorators run in this order:

```
Client → logging_middleware → metrics_middleware → request_id_middleware → route handler
```

`request_id_middleware` must be innermost (registered first) so it sets `request_id_var` before
either of the other two tries to read it.

### Full code block

```python
%%writefile app.py
"""Weather API — Beat 6: metrics middleware added (full instrumented version)."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from contextvars import ContextVar
import uuid
import logging
import json
import time                                                      # perf_counter for sub-millisecond timing

app = FastAPI(title="Weather API")

FORECASTS = {
    "Boston":    "Partly cloudy, 18 °C",
    "Dubai":     "Sunny, 41 °C",
    "London":    "Overcast, 12 °C",
    "New York":  "Thunderstorms, 22 °C",
    "Tokyo":     "Clear, 26 °C",
}

REQUESTS_TOTAL = Counter("requests_total", "Total HTTP requests", ["path", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "HTTP request latency in seconds", ["path"])
INFLIGHT_REQUESTS = Gauge("inflight_requests", "Number of requests currently in-flight")

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts":         self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":      record.levelname,
            "request_id": request_id_var.get(),
            "path":       getattr(record, "path", ""),
            "status":     getattr(record, "status", ""),
            "latency_ms": getattr(record, "latency_ms", ""),
            "msg":        record.getMessage(),
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("weather")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Middleware (registered in reverse — last registered = outermost) ──────────
@app.middleware("http")                                          # innermost: sets request_id first
async def request_id_middleware(request: Request, call_next):
    rid = uuid.uuid4().hex                                      # 32-char hex unique request ID
    request_id_var.set(rid)                                     # bind ID to this async task
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

@app.middleware("http")                                         # middle: times + increments metrics
async def metrics_middleware(request: Request, call_next):      # wraps the route handler
    INFLIGHT_REQUESTS.inc()                                     # +1 before processing starts
    start = time.perf_counter()                                 # record high-resolution start time
    response = await call_next(request)                         # run route handler
    latency = time.perf_counter() - start                      # elapsed seconds (float)
    INFLIGHT_REQUESTS.dec()                                     # -1 after response is ready
    path   = request.url.path                                   # e.g. "/weather"
    status = str(response.status_code)                          # convert int → string for label
    REQUESTS_TOTAL.labels(path=path, status=status).inc()       # increment counter for this path+status combo
    REQUEST_LATENCY.labels(path=path).observe(latency)          # record duration in histogram buckets
    return response

@app.middleware("http")                                         # outermost: logs after full round-trip
async def logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("request", extra={
        "path":       request.url.path,
        "status":     response.status_code,
        "latency_ms": round(latency_ms, 2),
    })
    return response

# ── Route ─────────────────────────────────────────────────────────────────────
@app.get("/weather")                                             # GET /weather?city=… handler
def get_weather(city: str):                                      # city is a required query parameter
    forecast = FORECASTS.get(city)                              # look up city in FORECASTS dict
    if forecast is None:                                        # city not in our data store
        return JSONResponse(                                    # return 404 JSON error
            status_code=404,
            content={"error": f"No forecast for '{city}'"},
        )
    return {"city": city, "forecast": forecast}                 # success: return city + forecast
```

---

## Cell 21 — Metrics Demo

### Purpose
Hits all five cities and reads back the `/metrics` output to confirm the counter is climbing and
the histogram is recording observations.

### Line-by-line explanation

```python
for city in cities:
    r = httpx.get(f"http://localhost:8000/weather?city={city.replace(' ', '%20')}", timeout=5)
```
`"New York"` contains a space; `%20` is the URL-encoded form. `replace(' ', '%20')` avoids a
malformed URL for that city.

```python
for line in metrics_text.splitlines():
    if line.startswith("requests_total") and not line.startswith("#"):
        print(line)
```
Filters out the HELP/TYPE comment lines and prints only the counter sample lines. Each line
shows the label set and the current count, e.g.:
`requests_total{path="/weather",status="200"} 5.0`

### Full code block

```python
import subprocess, time, httpx

subprocess.run(["pkill", "-f", "uvicorn app:app"], capture_output=True)
time.sleep(1)
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "app:app", "--port", "8000", "--log-level", "warning"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
time.sleep(2)

# ── Hit /weather 5 times ──────────────────────────────────────────────────────
cities = ["Boston", "Dubai", "London", "New York", "Tokyo"]
for city in cities:                                             # make one request per city
    r = httpx.get(f"http://localhost:8000/weather?city={city.replace(' ', '%20')}",
                  timeout=5)
    print(f"{city:12s} → {r.status_code}")                     # confirm each returns 200

# ── Read /metrics and show requests_total counter ────────────────────────────
print("\n── /metrics sample ──")
metrics_text = httpx.get("http://localhost:8000/metrics", timeout=5).text
for line in metrics_text.splitlines():                          # scan every line of the exporter output
    if line.startswith("requests_total") and not line.startswith("#"):  # filter to counter samples
        print(line)                                             # each line: labels + current count
    if line.startswith("request_latency_seconds_count"):        # histogram count lines
        print(line)
    if line.startswith("inflight_requests") and not line.startswith("#"):
        print(line)                                             # in-flight gauge (should be 0 at rest)
```

---

## Cell 22 — Cardinality Pitfall Demo

### Purpose
Demonstrates concretely why using an unbounded query parameter (city name) as a label is
dangerous. The room sees the current (correct) cardinality, then reads the projection of what
would happen with a `city` label.

### Key concept

A Prometheus counter with labels `["path", "status"]` creates at most:
`number_of_paths × number_of_status_codes` time series.

For our one-endpoint service: `1 path × 3 status codes = 3 time series maximum`.

Adding `"city"` as a third label changes the upper bound to:
`1 × 3 × (number of distinct city strings ever sent)`.

In a real geocoded weather API that accepts any city string from users, this is unbounded.
Prometheus stores all time series in memory. Enough series → OOM → scrape failures → silent
monitoring blackout.

### Full code block

```python
# ── Cardinality pitfall demo ──────────────────────────────────────────────────
# We temporarily redefine REQUESTS_TOTAL with a 'city' label to show the explosion.
# This runs against the live server's /metrics — we parse the output, not re-run the server.
#
# The pattern: each unique (path, status, city) triple becomes its own time series.
# With 195 countries and thousands of cities, Prometheus runs out of memory.

print("Current requests_total time series in /metrics (correct — 2 labels):")
metrics_text = httpx.get("http://localhost:8000/metrics", timeout=5).text
counter_lines = [l for l in metrics_text.splitlines()          # collect all counter sample lines
                 if l.startswith("requests_total{")]            # filter to just counter samples
for line in counter_lines:
    print(f"  {line}")                                         # each line is one time series

print(f"\nTotal time series for requests_total: {len(counter_lines)}")  # small: 1-2 series

pitfall_msg = (
    "\n" +
    "-" * 53 + "\n" +
    "If we added city as a label:\n" +
    "  REQUESTS_TOTAL.labels(path=path, status=status, city=city).inc()\n\n" +
    "After hitting 5 cities:\n" +
    '  requests_total{path="/weather",status="200",city="Boston"}   1.0\n' +
    '  requests_total{path="/weather",status="200",city="Dubai"}    1.0\n' +
    '  requests_total{path="/weather",status="200",city="London"}   1.0\n' +
    '  requests_total{path="/weather",status="200",city="New York"} 1.0\n' +
    '  requests_total{path="/weather",status="200",city="Tokyo"}    1.0\n\n' +
    "In production with a real geocoded weather API:\n" +
    "  -> 195+ countries x thousands of cities x N status codes\n" +
    "  -> millions of time series -> Prometheus OOM\n\n" +
    "Rule: only use labels with a bounded, known set of values.\n" +
    "-" * 53
)
print(pitfall_msg)                                             # display the cardinality warning
```

---

## Cell 24 — `read_metrics.py`

### Purpose
A standalone script that fetches `/metrics` and parses it back into Python objects using
`prometheus_client`'s own text parser. The same pattern is used in the Module 10A integration's
`lib/metrics_reader.py`.

### Line-by-line explanation

```python
from prometheus_client.parser import text_string_to_metric_families
```
`text_string_to_metric_families` is a generator that parses OpenMetrics / Prometheus text format
and yields `MetricFamily` objects. Each `MetricFamily` has:
- `.name` — e.g. `"requests_total"`
- `.samples` — a list of `Sample(name, labels, value, timestamp, exemplar)` namedtuples

```python
def fetch_metric(name: str) -> list:
    text = httpx.get(f"{BASE_URL}/metrics", timeout=5).text
    for family in text_string_to_metric_families(text):
        if family.name == name:
            return family.samples
    return []
```
Fetches the full `/metrics` page on every call and iterates families until it finds the one
matching `name`. Returns the samples list, or `[]` if the metric is not registered.

> **Why re-fetch each call?** For a demo script, simplicity beats efficiency. In production you
> would fetch once and extract all metrics in one pass.

```python
for s in samples:
    labels_str = ", ".join(f"{k}={v}" for k, v in s.labels.items())
    print(f"  {{{labels_str}}}  →  {s.value:.0f}")
```
`s.labels` is a dict of label name → value. The `join` builds a human-readable key=value string.
`s.value` is a `float`; `:.0f` formats it as an integer (no decimal) since counters accumulate
whole requests.

### Full code block

```python
%%writefile read_metrics.py
"""Beat 7 — parse /metrics from a running weather service."""
import httpx                                                     # HTTP client for fetching /metrics
from prometheus_client.parser import text_string_to_metric_families  # OpenMetrics text parser

BASE_URL = "http://localhost:8000"                              # weather service base URL

def fetch_metric(name: str) -> list:                            # returns list of Sample objects
    """Fetch /metrics and extract all samples for a given metric name."""
    text = httpx.get(f"{BASE_URL}/metrics", timeout=5).text     # GET the full /metrics page
    for family in text_string_to_metric_families(text):         # iterate parsed metric families
        if family.name == name:                                 # find the family we want
            return family.samples                               # return all its samples
    return []                                                   # metric not found → empty list

def main():
    # ── requests_total counter ────────────────────────────────────────────────
    samples = fetch_metric("requests_total")                    # pull all requests_total samples
    print("requests_total samples:")
    for s in samples:                                           # each s is a Sample namedtuple
        labels_str = ", ".join(f"{k}={v}" for k, v in s.labels.items())  # format label dict
        print(f"  {{{labels_str}}}  →  {s.value:.0f}")         # label set + current count

    # ── request_latency_seconds histogram ────────────────────────────────────
    hist_count = fetch_metric("request_latency_seconds_count")  # histogram observation count
    print("\nrequest_latency_seconds_count samples:")
    for s in hist_count:
        labels_str = ", ".join(f"{k}={v}" for k, v in s.labels.items())
        print(f"  {{{labels_str}}}  →  {s.value:.0f} observations")

    # ── inflight_requests gauge ───────────────────────────────────────────────
    gauge_samples = fetch_metric("inflight_requests")           # pull gauge samples
    print("\ninflight_requests:")
    for s in gauge_samples:
        print(f"  value = {s.value}")                          # should be 0.0 (no active requests)

if __name__ == "__main__":
    main()                                                      # entry point when run directly
```

---

## Cell 25 — Run `read_metrics.py`

### Purpose
Executes `read_metrics.py` as a subprocess and prints its output — showing the parsed metric
values from the live server.

### Line-by-line explanation

```python
result = subprocess.run(
    [sys.executable, "read_metrics.py"],
    capture_output=True,
    text=True,
)
```
`sys.executable` ensures we use the same Python binary as the notebook. `capture_output=True`
collects both stdout and stderr into `result.stdout` / `result.stderr`.
`text=True` decodes the bytes to a string automatically.

```python
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```
Prints the parsed metric output. If `result.stderr` is non-empty (an exception in
`read_metrics.py`), it prints that too so errors are visible.

### Full code block

```python
import subprocess, sys

result = subprocess.run(                                        # run read_metrics.py as a subprocess
    [sys.executable, "read_metrics.py"],                        # use the same Python as the notebook
    capture_output=True,                                        # capture stdout and stderr
    text=True,                                                  # decode bytes → str automatically
)
print(result.stdout)                                            # show parsed metric output
if result.stderr:                                               # print any errors if present
    print("STDERR:", result.stderr)
```

---

## Cell 26 — Shutdown

### Purpose
Terminates the background `uvicorn` process cleanly at the end of the session.

### Full code block

```python
import subprocess

subprocess.run(["pkill", "-f", "uvicorn app:app"],              # send SIGTERM to uvicorn process
               capture_output=True)
print("Server stopped.")                                        # confirm shutdown
```

---

## Concept Summary

| Concept | Where introduced | Key rule |
|---------|-----------------|----------|
| Counter | Beat 2, Cell 08 | Only goes up; use `rate()` for req/s |
| Histogram | Beat 2, Cell 08 | Buckets are fixed at declaration; choose them before load testing |
| Gauge | Beat 2, Cell 08 | Increment before handler, decrement after (or in `finally`) |
| Module-scope metrics | Beat 2, Cell 09 | Never declare inside a function — you will hit `ValueError` on the second request |
| `/metrics` mount | Beat 3, Cell 11 | `make_asgi_app()` + `app.mount()` — two lines |
| ContextVar | Beat 4, Cell 14 | Async-safe per-task storage; not thread-local |
| Middleware order | Beat 6, Cell 20 | Last registered = outermost; producer before consumer |
| Label cardinality | Beat 6, Cell 22 | Only label with bounded, known value sets |
| OpenMetrics parser | Beat 7, Cell 24 | `text_string_to_metric_families` → `MetricFamily.samples` |
