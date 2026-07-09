# Module 10A Applied Lab — Complete Code Walkthrough

Every line of code written during the lab session is explained here.
Each section ends with the **full code block** for that beat.

---

## Table of Contents

1. [Environment Verification](#environment-verification)
2. [Beat 1 — `/extract`: Stateless NLP Wrap](#beat-1--extract-stateless-nlp-wrap)
3. [Beat 2 — Dependency Injection + Lifespan](#beat-2--dependency-injection--lifespan)
4. [Beat 3 — RAG Wrap: Retrieve → Assemble → Generate → Cite](#beat-3--rag-wrap)
5. [Beat 4 — `/healthz` and `/readyz`](#beat-4--healthz-and-readyz)
6. [Beat 5 — CORS Middleware](#beat-5--cors-middleware)
7. [Beat 6 — Next.js Page Scaffold](#beat-6--nextjs-page-scaffold)
8. [Beat 7 — Backend Dockerfile](#beat-7--backend-dockerfile)
9. [Beat 8 — Frontend Dockerfile (Multi-Stage)](#beat-8--frontend-dockerfile-multi-stage)
10. [Beat 9 — Lab-Level docker-compose.yml](#beat-9--lab-level-docker-composeyml)
11. [Shutdown Cell](#shutdown-cell)

---

## Environment Verification

This cell checks that every dependency is importable and that Docker Compose v2 and
Node.js 20 are on the PATH. It must pass before any other cell will work.

### Line-by-line explanation

```python
import subprocess
```
Lets Python spawn shell commands (`docker compose version`, `node --version`) without
leaving the notebook kernel.

```python
import importlib
```
`importlib.import_module(name)` attempts a dynamic import at runtime and raises
`ImportError` if the package is missing. This is safer than a bare `import` that
would crash the cell on the first missing package.

```python
import sys
```
`sys.version` contains the full Python version string; we split on whitespace to
extract the semver portion for display.

```python
REQUIRED_PACKAGES = {
    "fastapi":               "fastapi",
    "uvicorn":               "uvicorn",
    ...
}
```
A dict mapping the `pip install` name (shown to the user) to the `import` name (used
by `importlib`). They differ for some packages — e.g. `weaviate-client` installs as
`weaviate`, and `sentence-transformers` imports as `sentence_transformers`.

```python
for pip_name, import_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(import_name)
        print(f"  [OK]      {pip_name}")
    except ImportError:
        print(f"  [MISSING] {pip_name}  ->  pip install {pip_name}")
        all_ok = False
```
Iterates every entry. A successful import prints `[OK]`; an `ImportError` prints
`[MISSING]` with the exact install command and sets the `all_ok` flag to `False`.
The loop continues so all missing packages are reported in one pass.

```python
dc = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
```
`subprocess.run` blocks until the command exits. `capture_output=True` redirects
both stdout and stderr to strings; `text=True` decodes bytes to `str`. The list
form `["docker", "compose", "version"]` avoids shell injection.

```python
node = subprocess.run(["node", "--version"], capture_output=True, text=True)
node_ver = node.stdout.strip()
if node.returncode == 0 and node_ver.startswith("v20"):
```
Node.js version output is `v20.x.y\n`. `.strip()` removes the trailing newline;
`.startswith("v20")` checks the major version without parsing semver manually.

### Full code block

```python
import subprocess
import importlib
import sys

REQUIRED_PACKAGES = {
    "fastapi":               "fastapi",
    "uvicorn":               "uvicorn",
    "pydantic":              "pydantic",
    "neo4j":                 "neo4j",
    "spacy":                 "spacy",
    "weaviate-client":       "weaviate",
    "sentence-transformers": "sentence_transformers",
    "transformers":          "transformers",
    "requests":              "requests",
}

print(f"Python {sys.version.split()[0]}\n")

all_ok = True

for pip_name, import_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(import_name)
        print(f"  [OK]      {pip_name}")
    except ImportError:
        print(f"  [MISSING] {pip_name}  ->  pip install {pip_name}")
        all_ok = False

dc = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
if dc.returncode == 0:
    print(f"\n  [OK]      docker compose  —  {dc.stdout.strip()}")
else:
    print("\n  [MISSING] docker compose v2  ->  install Docker Desktop")
    all_ok = False

node = subprocess.run(["node", "--version"], capture_output=True, text=True)
node_ver = node.stdout.strip()
if node.returncode == 0 and node_ver.startswith("v20"):
    print(f"  [OK]      node  —  {node_ver}")
else:
    print(f"  [WARN]    node  —  {node_ver}  (expected v20.x LTS)")

print()
print("All dependencies satisfied — ready for Beat 0." if all_ok else
      "Install missing packages, then re-run this cell.")
```

---

## Beat 1 — `/extract`: Stateless NLP Wrap

**File:** `demo/main_v1.py`  
**What it builds:** A `POST /extract` endpoint that accepts raw text and returns
a typed list of named-entity spans.

### Line-by-line explanation

```python
from fastapi import FastAPI
```
`FastAPI` is the main application class. It manages routing, dependency injection,
middleware, and auto-generates OpenAPI/Swagger documentation.

```python
from pydantic import BaseModel
```
`BaseModel` is Pydantic's base class. Any class that inherits from it gets automatic
JSON parsing, type coercion, and validation. FastAPI uses it for both request bodies
and response serialisation.

```python
from typing import List
```
`List[Entity]` is a generic type hint. FastAPI and Pydantic use it to validate that
the `entities` field in the response is an array of `Entity` objects, not a scalar.

```python
import spacy
```
spaCy is the NLP library. Its `nlp` pipeline object tokenises text, assigns
part-of-speech tags, builds the dependency tree, and runs the NER model.

```python
nlp = spacy.load("en_core_web_sm")
```
**Module scope, not inside the endpoint.** `spacy.load` reads the model weights from
disk into memory — it takes 200–800 ms. Placing it here means it runs once when the
Python process starts, not on every HTTP request.

```python
app = FastAPI(title="Movie Information Service — SI Demo", version="1.0.0", ...)
```
Creates the application instance. `title` and `version` appear in the auto-generated
Swagger UI at `/docs`. The `description` field populates the OpenAPI info block.

```python
class ExtractRequest(BaseModel):
    text: str
```
The request schema. FastAPI reads the JSON body, finds `text`, validates it is a
string, and passes it as a typed `ExtractRequest` object to the handler. If `text`
is missing, FastAPI automatically returns a `422 Unprocessable Entity` before your
handler is called.

```python
class Entity(BaseModel):
    text:  str
    label: str
    start: int
    end:   int
```
One named-entity span. `text` is the surface form; `label` is the spaCy type string
(e.g. `PERSON`, `WORK_OF_ART`, `GPE`); `start`/`end` are character offsets, not
token indices, so the client can highlight the span without running spaCy itself.

```python
class ExtractResponse(BaseModel):
    entities:     List[Entity]
    entity_count: int
```
The response schema. `entity_count` duplicates `len(entities)` — it is a client
convenience field so the consumer does not need to compute `len()` in JavaScript.

```python
@app.post("/extract", response_model=ExtractResponse, ...)
```
`@app.post` registers the route. `response_model=ExtractResponse` tells FastAPI to
(a) validate the return value against the schema and (b) include the schema in
OpenAPI so Swagger renders it correctly. If the handler returns something that does
not match, FastAPI raises a 500 before the response is sent.

```python
async def extract_entities(request: ExtractRequest) -> ExtractResponse:
```
`async def` is required for all FastAPI handlers so the event loop is not blocked
during I/O. The `request: ExtractRequest` parameter signals FastAPI to parse and
validate the JSON body. The return type annotation `-> ExtractResponse` is
informational but consistent with `response_model`.

```python
    doc = nlp(request.text)
```
Runs the full spaCy pipeline on the input string. Returns a `Doc` object containing
tokens, POS tags, dependency arcs, and `doc.ents` (a tuple of entity spans).

```python
    entities: List[Entity] = [
        Entity(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char,
        )
        for ent in doc.ents
    ]
```
A list comprehension over `doc.ents`. Each `ent` is a spaCy `Span` object.
`ent.label_` (with trailing underscore) returns the string label, not the integer ID.
`ent.start_char` / `ent.end_char` are character offsets, matching our `Entity` schema.

```python
    return ExtractResponse(entities=entities, entity_count=len(entities))
```
FastAPI serialises this to JSON, validates it against `response_model`, and sends it
with `Content-Type: application/json` and `200 OK`.

### Full code block — `demo/main_v1.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import spacy

# Load once at module scope — avoids per-request cold start
nlp = spacy.load("en_core_web_sm")

app = FastAPI(
    title="Movie Information Service — SI Demo",
    version="1.0.0",
    description="Module 10A SI demonstration (movie domain).",
)

class ExtractRequest(BaseModel):
    text: str  # Raw text to analyse

class Entity(BaseModel):
    text:  str  # Surface form of the entity
    label: str  # spaCy entity type: PERSON, WORK_OF_ART, ORG, DATE, …
    start: int  # Char offset (inclusive)
    end:   int  # Char offset (exclusive)

class ExtractResponse(BaseModel):
    entities:     List[Entity]  # All recognised named-entity spans
    entity_count: int           # len(entities) — client convenience

@app.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Named-entity extraction",
    tags=["NLP"],
)
async def extract_entities(request: ExtractRequest) -> ExtractResponse:
    doc = nlp(request.text)  # Run tokeniser → POS → dep → NER
    entities: List[Entity] = [
        Entity(
            text=ent.text,
            label=ent.label_,       # String label (trailing _ not int id)
            start=ent.start_char,   # Char-level start offset
            end=ent.end_char,       # Char-level end offset (exclusive)
        )
        for ent in doc.ents
    ]
    return ExtractResponse(entities=entities, entity_count=len(entities))
```

---

## Beat 2 — Dependency Injection + Lifespan

**File:** `demo/main_v2.py`  
**What it adds:** A `lifespan` context manager that opens the Neo4j connection pool
at startup and closes it at shutdown; a `get_driver` dependency function; a
`POST /kg/query` endpoint.

### Line-by-line explanation

```python
from contextlib import asynccontextmanager
```
`asynccontextmanager` is a decorator from the standard library. It transforms an
`async` generator function (one with `yield`) into an async context manager usable
with `async with`. FastAPI accepts a lifespan as an async context manager.

```python
from fastapi import FastAPI, Depends, Request
```
`Depends` is the dependency-injection marker: `Depends(get_driver)` tells FastAPI to
call `get_driver` and inject its return value into the handler. `Request` gives
access to `request.app.state` inside dependency functions.

```python
from neo4j import AsyncGraphDatabase, AsyncDriver
```
`AsyncGraphDatabase.driver(uri, auth=...)` creates an async connection pool to Neo4j.
`AsyncDriver` is the type hint for the pool object, used in function signatures.

```python
NEO4J_URI      = "bolt://localhost:7688"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"
```
Hard-coded for the demo. In production, read these from environment variables using
`os.environ` or a library like `python-dotenv` so secrets are never in source code.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    driver: AsyncDriver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    app.state.neo4j_driver = driver
    yield
    await driver.close()
```
Everything **before** `yield` runs at application startup (after the event loop starts
but before the first request is served). `app.state` is a `SimpleNamespace` that
persists for the lifetime of the application — any attribute set here is accessible
in every request handler via `request.app.state`.

`yield` is the dividing line. The application serves traffic here. Everything
**after** `yield` runs at shutdown (after the last request is done). `driver.close()`
releases all pooled TCP connections.

```python
app = FastAPI(title="...", version="2.0.0", lifespan=lifespan)
```
Passing `lifespan=lifespan` wires the context manager into FastAPI's startup/shutdown
lifecycle. This replaces the older `@app.on_event("startup")` pattern.

```python
async def get_driver(request: Request) -> AsyncDriver:
    return request.app.state.neo4j_driver
```
A **dependency function** — any function usable with `Depends()`. FastAPI calls it
once per request that declares `driver = Depends(get_driver)`. It returns the shared
driver pool from `app.state`. There is no session opened here — callers open sessions
themselves inside `async with driver.session()`.

```python
class KGQueryRequest(BaseModel):
    cypher: str
    params: dict = {}
```
`cypher` is the raw Cypher string. `params` is an optional dict bound into the
Cypher server-side — Neo4j substitutes `$param_name` placeholders, preventing
Cypher injection. The `= {}` default means the field is optional in the JSON body.

```python
async with driver.session() as session:
    result = await session.run(body.cypher, body.params)
    records = [dict(r) async for r in result]
```
`driver.session()` opens a lightweight session (logical connection) from the pool.
`session.run` is async; it returns an `AsyncResult` cursor. The `async for` iterates
records from the cursor without loading them all into memory at once. `dict(r)`
converts a Neo4j `Record` to a plain Python dict for JSON serialisation.

```python
return {"records": records, "count": len(records)}
```
FastAPI serialises this plain dict to JSON. Because there is no `response_model=`
here, FastAPI skips validation — the return type is whatever the dict contains.

### Full code block — `demo/main_v2.py`

```python
from contextlib import asynccontextmanager
from typing import List, AsyncGenerator

from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
import spacy
from neo4j import AsyncGraphDatabase, AsyncDriver

NEO4J_URI      = "bolt://localhost:7688"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

nlp = spacy.load("en_core_web_sm")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: open the Neo4j connection pool
    driver: AsyncDriver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    app.state.neo4j_driver = driver  # Share via app.state

    yield  # Application runs here

    # SHUTDOWN: release all pooled connections
    await driver.close()

app = FastAPI(
    title="Movie Information Service — SI Demo",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Beat 1 models + endpoint (unchanged) ─────────────────────────────────────
class ExtractRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str; label: str; start: int; end: int

class ExtractResponse(BaseModel):
    entities: List[Entity]; entity_count: int

@app.post("/extract", response_model=ExtractResponse, tags=["NLP"])
async def extract_entities(request: ExtractRequest) -> ExtractResponse:
    doc = nlp(request.text)
    entities = [Entity(text=e.text, label=e.label_, start=e.start_char, end=e.end_char)
                for e in doc.ents]
    return ExtractResponse(entities=entities, entity_count=len(entities))

# ── Beat 2: dependency + KG endpoint ─────────────────────────────────────────
async def get_driver(request: Request) -> AsyncDriver:
    return request.app.state.neo4j_driver  # Return the shared pool

class KGQueryRequest(BaseModel):
    cypher: str        # Raw Cypher query
    params: dict = {}  # Optional query parameters (injected server-side)

@app.post("/kg/query", tags=["Knowledge Graph"])
async def kg_query(
    body: KGQueryRequest,
    driver: AsyncDriver = Depends(get_driver),  # Injected by FastAPI
) -> dict:
    async with driver.session() as session:
        result = await session.run(body.cypher, body.params)
        records = [dict(r) async for r in result]  # async cursor iteration
    return {"records": records, "count": len(records)}
```

---

## Beat 3 — RAG Wrap

**File:** `demo/main_v3.py`  
**What it adds:** Weaviate retrieval, numbered-context prompt assembly, flan-t5-base
generation, citation extraction, and grounding refusal.

### Line-by-line explanation

#### Constants

```python
WEAVIATE_URL = "http://localhost:8081"
RAG_CLASS    = "MovieReview"
TOP_K        = 4
```
`WEAVIATE_URL` targets the demo Weaviate container on port 8081. `RAG_CLASS` is the
Weaviate class (analogous to a database table) that stores movie-review chunks.
`TOP_K = 4` means we retrieve the four most semantically similar chunks per question.

```python
GROUNDING_SENTINEL = "I cannot answer this question from the provided sources."
```
A fixed string returned instead of the generated text when the model produces no
citation markers. It signals to the client that the answer is not grounded in the
retrieved sources.

```python
PROMPT_TEMPLATE = (
    "You are a movie expert. "
    "Use ONLY the excerpts below to answer. "
    "Cite each excerpt you use with its number in square brackets.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
```
A Python format-string with two placeholders: `{context}` (the numbered chunks) and
`{question}`. The instruction "cite each excerpt you use with its number in square
brackets" is what causes the model to emit `[1]`, `[2]` markers that we parse later.

#### Lifespan additions

```python
wv_client = weaviate.Client(url=WEAVIATE_URL)
app.state.weaviate_client = wv_client
```
Creates the Weaviate HTTP client. The v3 weaviate-client is synchronous; it wraps
the Weaviate GraphQL API. Stored on `app.state` so all handlers share one client.

```python
generator = hf_pipeline("text2text-generation", model="google/flan-t5-base")
app.state.generator = generator
```
`hf_pipeline` is a high-level Hugging Face wrapper. `text2text-generation` is the
task type for encoder-decoder (seq2seq) models like flan-t5. The model is loaded
from the local HuggingFace cache (`~/.cache/huggingface`). This is the module-scope
loading pattern applied to the generator — loading it inside the endpoint would add
20–30 s per request.

#### `assemble_prompt`

```python
def assemble_prompt(chunks: List[str], question: str) -> str:
    context_lines = [f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks)]
    context = "\n\n".join(context_lines)
    return PROMPT_TEMPLATE.format(context=context, question=question)
```
Numbers each chunk starting at 1 (not 0) because the model will emit `[1]`, `[2]`,
etc. in its answer — 1-based is more natural for humans and models. Chunks are
separated by a blank line to make the boundary visible to the model.
`str.format()` substitutes `{context}` and `{question}` into the template.

#### `extract_citations`

```python
raw_indices = re.findall(r"\[(\d+)\]", answer)
```
`re.findall` returns a list of all strings captured by the parenthesised group
`(\d+)`. For the answer `"The film uses IMAX [1] and practical effects [2]."`,
this returns `["1", "2"]`.

```python
unique_indices = list(dict.fromkeys(int(i) for i in raw_indices))
```
Converts strings to ints and deduplicates while preserving order. `dict.fromkeys`
is the idiomatic order-preserving deduplication pattern in Python (since 3.7,
dicts maintain insertion order).

```python
return [
    CitedChunk(index=i, text=chunks[i - 1])
    for i in unique_indices
    if 1 <= i <= len(chunks)
]
```
`i - 1` converts from 1-based citation index to 0-based list index. The guard
`1 <= i <= len(chunks)` discards any indices the model hallucinated (e.g. `[9]`
when only 4 chunks were retrieved).

#### Step (a) — Weaviate retrieval

```python
result = (
    wv.query
    .get(RAG_CLASS, ["chunk_text"])
    .with_near_text({"concepts": [body.question]})
    .with_limit(TOP_K)
    .do()
)
```
`.get(RAG_CLASS, ["chunk_text"])` starts a GraphQL query on the `MovieReview` class,
requesting only the `chunk_text` property. `.with_near_text({"concepts": [...]})` tells
Weaviate to embed the question with the same model used at index time, then perform
cosine similarity search. `.with_limit(TOP_K)` caps the results at 4. `.do()` executes
the HTTP request and returns the raw response dict.

```python
hits   = result["data"]["Get"][RAG_CLASS]
chunks = [h["chunk_text"] for h in hits]
```
Navigates Weaviate's nested GraphQL response: `data → Get → ClassName → list`.
Extracts the `chunk_text` string from each hit object.

#### Step (c) — Generation

```python
outputs    = gen(prompt, max_new_tokens=256, do_sample=False)
raw_answer = outputs[0]["generated_text"].strip()
```
`gen(prompt, ...)` calls the pipeline. `max_new_tokens=256` caps the output length.
`do_sample=False` uses greedy decoding (always picks the highest-probability token),
which is deterministic and reproducible. `outputs[0]["generated_text"]` extracts the
text from the pipeline's return structure.

#### Step (e) — Grounding refusal

```python
if not citations:
    return RAGResponse(answer=GROUNDING_SENTINEL, citations=[], grounded=False)
```
If `extract_citations` returned an empty list, the model generated text without
referencing any retrieved chunk. Rather than return potentially hallucinated content,
the endpoint returns the sentinel. The `grounded=False` field tells the client to
display a "no sources available" message instead of the answer.

### Full code block — `demo/main_v3.py`

```python
import re
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
import spacy
from neo4j import AsyncGraphDatabase, AsyncDriver
import weaviate
from transformers import pipeline as hf_pipeline

NEO4J_URI      = "bolt://localhost:7688"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"
WEAVIATE_URL   = "http://localhost:8081"
RAG_CLASS      = "MovieReview"
TOP_K          = 4
GROUNDING_SENTINEL = "I cannot answer this question from the provided sources."
PROMPT_TEMPLATE = (
    "You are a movie expert. Use ONLY the excerpts below to answer. "
    "Cite each excerpt you use with its number in square brackets.\n\n"
    "{context}\n\nQuestion: {question}\nAnswer:"
)

nlp = spacy.load("en_core_web_sm")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open Neo4j driver
    driver: AsyncDriver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    app.state.neo4j_driver = driver
    # Create Weaviate client
    app.state.weaviate_client = weaviate.Client(url=WEAVIATE_URL)
    # Load generator — reads from local HuggingFace cache
    app.state.generator = hf_pipeline("text2text-generation", model="google/flan-t5-base")
    yield
    await driver.close()

app = FastAPI(title="Movie Information Service — SI Demo", version="3.0.0", lifespan=lifespan)

# ── Beats 1 + 2 (abbreviated) ────────────────────────────────────────────────
class ExtractRequest(BaseModel): text: str
class Entity(BaseModel): text: str; label: str; start: int; end: int
class ExtractResponse(BaseModel): entities: List[Entity]; entity_count: int

@app.post("/extract", response_model=ExtractResponse, tags=["NLP"])
async def extract_entities(request: ExtractRequest) -> ExtractResponse:
    doc = nlp(request.text)
    entities = [Entity(text=e.text, label=e.label_, start=e.start_char, end=e.end_char) for e in doc.ents]
    return ExtractResponse(entities=entities, entity_count=len(entities))

async def get_driver(request: Request) -> AsyncDriver: return request.app.state.neo4j_driver
class KGQueryRequest(BaseModel): cypher: str; params: dict = {}

@app.post("/kg/query", tags=["Knowledge Graph"])
async def kg_query(body: KGQueryRequest, driver: AsyncDriver = Depends(get_driver)) -> dict:
    async with driver.session() as session:
        result = await session.run(body.cypher, body.params)
        records = [dict(r) async for r in result]
    return {"records": records, "count": len(records)}

# ── Beat 3 helpers ────────────────────────────────────────────────────────────
def assemble_prompt(chunks: List[str], question: str) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    return PROMPT_TEMPLATE.format(context=context, question=question)

def extract_citations(answer: str, chunks: List[str]):
    raw     = re.findall(r"\[(\d+)\]", answer)
    unique  = list(dict.fromkeys(int(i) for i in raw))
    return [CitedChunk(index=i, text=chunks[i-1]) for i in unique if 1 <= i <= len(chunks)]

class CitedChunk(BaseModel): index: int; text: str
class RAGRequest(BaseModel): question: str
class RAGResponse(BaseModel): answer: str; citations: List[CitedChunk]; grounded: bool

@app.post("/rag/answer", response_model=RAGResponse, tags=["RAG"])
async def rag_answer(body: RAGRequest, request: Request) -> RAGResponse:
    wv  = request.app.state.weaviate_client
    gen = request.app.state.generator

    # (a) Retrieve
    result = wv.query.get(RAG_CLASS, ["chunk_text"]).with_near_text({"concepts": [body.question]}).with_limit(TOP_K).do()
    chunks = [h["chunk_text"] for h in result["data"]["Get"][RAG_CLASS]]

    # (b) Assemble
    prompt = assemble_prompt(chunks, body.question)

    # (c) Generate
    raw_answer = gen(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"].strip()

    # (d) Extract citations
    citations = extract_citations(raw_answer, chunks)

    # (e) Grounding refusal
    if not citations:
        return RAGResponse(answer=GROUNDING_SENTINEL, citations=[], grounded=False)
    return RAGResponse(answer=raw_answer, citations=citations, grounded=True)
```

---

## Beat 4 — `/healthz` and `/readyz`

**File:** `demo/main_v4.py`  
**What it adds:** A liveness probe and a readiness probe with two backend checks.

### Line-by-line explanation

```python
from fastapi.responses import JSONResponse
```
`JSONResponse` lets us return an arbitrary HTTP status code alongside the JSON body.
`return {"status": "ok"}` always yields 200; `JSONResponse(status_code=503, content=...)` yields 503.

```python
import time as _time
```
Renamed to `_time` to avoid shadowing the variable name `time` that `time.time()`
returns. We store the startup timestamp in `app.state.start_time` so `/healthz` can
report uptime.

#### `/healthz`

```python
@app.get("/healthz", tags=["Ops"])
async def healthz(request: Request) -> dict:
    uptime = _time.time() - request.app.state.start_time
    return {"status": "ok", "uptime_s": round(uptime, 2)}
```
A GET endpoint because health checks are read-only idempotent queries. Returns `200`
as long as the process is running and the event loop is responsive (no backend checks).
Kubernetes `livenessProbe` calls this; a non-200 response causes the container to be
restarted.

#### `/readyz`

```python
driver: AsyncDriver = request.app.state.neo4j_driver
async with driver.session() as session:
    await session.run("RETURN 1")
checks["neo4j"] = "ok"
```
`RETURN 1` is the cheapest valid Cypher query — it does no graph traversal. If Neo4j
is reachable, it succeeds in < 5 ms. Any exception populates `checks["neo4j"]` with
the error message instead of `"ok"`.

```python
async with httpx.AsyncClient() as client:
    wv_resp = await client.get(f"{WEAVIATE_URL}/v1/.well-known/ready", timeout=2.0)
checks["weaviate"] = "ok" if wv_resp.status_code == 200 else f"http {wv_resp.status_code}"
```
Weaviate publishes a readiness endpoint at `/v1/.well-known/ready` — it returns `200`
when ready, `503` when still initialising. We use `httpx.AsyncClient` (not `requests`)
because `requests` is synchronous and would block the event loop inside an `async def`.
`timeout=2.0` prevents a hung Weaviate from holding up the probe indefinitely.

```python
all_ok = all(v == "ok" for v in checks.values())
return JSONResponse(
    status_code=200 if all_ok else 503,
    content={"status": "ready" if all_ok else "not_ready", "checks": checks},
)
```
`all()` returns `True` only when every check passed. HTTP 503 tells the load balancer
to stop routing traffic to this instance. The `checks` dict provides per-backend
detail so ops engineers can diagnose which backend is down.

### Full code block — `/healthz` and `/readyz` additions

```python
import time as _time
import httpx
from fastapi.responses import JSONResponse

# In lifespan startup, add:
app.state.start_time = _time.time()

@app.get("/healthz", tags=["Ops"])
async def healthz(request: Request) -> dict:
    uptime = _time.time() - request.app.state.start_time  # Seconds since startup
    return {"status": "ok", "uptime_s": round(uptime, 2)}

@app.get("/readyz", tags=["Ops"])
async def readyz(request: Request) -> JSONResponse:
    checks = {}

    # Probe 1: Neo4j
    try:
        async with request.app.state.neo4j_driver.session() as session:
            await session.run("RETURN 1")   # Cheapest valid Cypher query
        checks["neo4j"] = "ok"
    except Exception as exc:
        checks["neo4j"] = f"error: {exc}"

    # Probe 2: Weaviate
    try:
        async with httpx.AsyncClient() as client:
            wv_resp = await client.get(
                f"{WEAVIATE_URL}/v1/.well-known/ready",
                timeout=2.0,  # Fail fast — do not let a hung probe block requests
            )
        checks["weaviate"] = "ok" if wv_resp.status_code == 200 else f"http {wv_resp.status_code}"
    except Exception as exc:
        checks["weaviate"] = f"error: {exc}"

    # Decision
    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )
```

---

## Beat 5 — CORS Middleware

**File:** `demo/main_final.py`  
**What it adds:** `CORSMiddleware` so the Next.js browser app can call the API.

### Line-by-line explanation

```python
from fastapi.middleware.cors import CORSMiddleware
```
FastAPI bundles the CORS middleware from Starlette. It intercepts incoming requests,
checks the `Origin` header, and adds `Access-Control-*` headers to the response.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
`add_middleware` must be called **before** route definitions (or at least before the
first request). `allow_origins` is an allowlist — the browser checks that its `Origin`
header is in this list. `allow_credentials=True` allows cookies and `Authorization`
headers. `allow_methods=["*"]` and `allow_headers=["*"]` permit all methods and
headers during development — in production, restrict these to what the client actually
sends.

**Why not `allow_origins=["*"]`?** When `allow_credentials=True`, the CORS spec
forbids the wildcard origin — the spec requires an explicit origin when cookies are
permitted. Using `"*"` with credentials would be silently ignored by the browser.

### Full code block — CORS addition

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="...", lifespan=lifespan)

# Must be called before route definitions
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Lab Next.js dev server
        "http://localhost:3001",   # SI demo Next.js dev server
    ],
    allow_credentials=True,    # Allow Authorization / Cookie headers
    allow_methods=["*"],       # GET, POST, OPTIONS, etc.
    allow_headers=["*"],       # Content-Type, Authorization, etc.
)

# ... all route definitions follow ...
```

---

## Beat 6 — Next.js Page Scaffold

**File:** `demo/frontend/pages/extract.tsx`  
**What it builds:** A React page at URL `/extract` that calls the FastAPI `/extract`
endpoint, renders the response in a table, and handles loading and error states.

### Line-by-line explanation

```typescript
import { useState } from "react";
```
`useState` is a React hook that declares a mutable variable. When its setter is
called, React re-renders the component with the new value. This is the React
equivalent of a Python variable whose assignment triggers a UI refresh.

```typescript
import type { NextPage } from "next";
```
`NextPage` is a TypeScript type — it tells the compiler that this component satisfies
the Next.js page contract (default export, no special props).

```typescript
interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
}
interface ExtractResponse {
  entities: Entity[];
  entity_count: number;
}
```
TypeScript interfaces mirror the Python Pydantic models exactly. If the backend
changes a field name, TypeScript will produce a compile-time error when you try to
read the old name from the response — the interface acts as a client-side schema
contract.

```typescript
const [inputText, setInputText]   = useState<string>("");
const [result,    setResult]      = useState<ExtractResponse | null>(null);
const [loading,   setLoading]     = useState<boolean>(false);
const [error,     setError]       = useState<string | null>(null);
```
Four state variables:
- `inputText` — bound two-way to the textarea
- `result` — `null` until the API responds; triggers the results table to render
- `loading` — `true` while the fetch is in flight; disables the button
- `error` — `null` unless the request fails; triggers the error paragraph

```typescript
const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";
```
`process.env.NEXT_PUBLIC_API_URL` reads from `.env.local`. The `??` (nullish
coalescing) operator falls back to the hard-coded default if the variable is not set.
The `NEXT_PUBLIC_` prefix tells Next.js to embed this value in the client bundle —
without the prefix, it would be `undefined` in the browser.

```typescript
const response = await fetch(`${apiUrl}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: inputText }),
});
```
`fetch` is the browser's built-in HTTP client. The equivalent `requests` call would be
`requests.post(f"{api_url}/extract", json={"text": input_text})`. The
`Content-Type: application/json` header tells FastAPI to parse the body as JSON.

```typescript
if (!response.ok) {
    const err = await response.json();
    throw new Error(JSON.stringify(err.detail ?? err));
}
```
`response.ok` is `true` for 200–299. FastAPI returns a 422 with `{"detail": [...]}` for
validation errors — this branch extracts and throws that detail so the `catch` block
can display it.

```typescript
const data: ExtractResponse = await response.json();
setResult(data);
```
`response.json()` parses the response body. The TypeScript generic `: ExtractResponse`
asserts the shape (it is a cast, not runtime validation — use Zod for runtime validation
in production). `setResult(data)` triggers a re-render that shows the results table.

```typescript
onChange={(e) => setInputText(e.target.value)}
```
A controlled input pattern: the textarea's value is always `inputText` (React controls
it), and every keystroke calls `setInputText` to update the state. This keeps React as
the single source of truth for the field value.

### Full code block — `demo/frontend/pages/extract.tsx`

```typescript
import { useState } from "react";
import type { NextPage } from "next";

interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
}

interface ExtractResponse {
  entities: Entity[];
  entity_count: number;
}

const ExtractPage: NextPage = () => {
  const [inputText, setInputText] = useState<string>("");
  const [result,    setResult]    = useState<ExtractResponse | null>(null);
  const [loading,   setLoading]   = useState<boolean>(false);
  const [error,     setError]     = useState<string | null>(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";
    try {
      const response = await fetch(`${apiUrl}/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(JSON.stringify(err.detail ?? err));
      }
      const data: ExtractResponse = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ fontFamily: "sans-serif", maxWidth: 720, margin: "2rem auto" }}>
      <h1>Named Entity Extraction — Movie Domain</h1>
      <textarea
        rows={5}
        style={{ width: "100%", fontSize: "1rem", marginBottom: "0.5rem" }}
        placeholder="Paste a movie description…"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />
      <button onClick={handleSubmit} disabled={loading || inputText.trim().length === 0}>
        {loading ? "Extracting…" : "Extract Entities"}
      </button>
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}
      {result && (
        <>
          <p>Found <strong>{result.entity_count}</strong> entities.</p>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "#f0f0f0" }}>
                <th>Text</th><th>Label</th><th>Start</th><th>End</th>
              </tr>
            </thead>
            <tbody>
              {result.entities.map((ent, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid #ddd" }}>
                  <td>{ent.text}</td>
                  <td><code>{ent.label}</code></td>
                  <td>{ent.start}</td>
                  <td>{ent.end}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </main>
  );
};

export default ExtractPage;
```

### Full code block — `demo/frontend/.env.local`

```bash
# NEXT_PUBLIC_ prefix: embedded in browser bundle — safe for URLs, never secrets
NEXT_PUBLIC_API_URL=http://localhost:8001
```

---

## Beat 7 — Backend Dockerfile

**File:** `demo/Dockerfile.backend`  
**Pattern:** Single-stage Python image with layer-cache optimisation.

### Line-by-line explanation

```dockerfile
FROM python:3.11-slim
```
The official Python image on Debian Slim (~130 MB). The `slim` variant omits
development headers, man pages, and build tools that are not needed at runtime.
Pinning the minor version (`3.11`) ensures builds are reproducible across environments.

```dockerfile
WORKDIR /app
```
All subsequent `COPY`, `RUN`, and `CMD` instructions use `/app` as the working
directory. If `/app` does not exist, Docker creates it automatically.

```dockerfile
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
```
**Layer-cache pattern:** `requirements.txt` is copied and dependencies are installed
*before* the application source is copied. Docker caches each `RUN` layer. If
`requirements.txt` has not changed since the last build, Docker skips `pip install`
entirely and restores the cached layer — saving 2–5 minutes per build cycle.
`--no-cache-dir` prevents pip from storing the download cache inside the layer,
keeping the image smaller.

```dockerfile
RUN python -m spacy download en_core_web_sm
```
Downloads the spaCy model into the image so the container does not need internet
access at runtime. This layer is also cached — it only re-runs if the preceding pip
layer changes.

```dockerfile
COPY . .
```
Copies application source code. This layer is cache-busted on every code change,
but because it comes *after* the pip layer, the pip cache remains valid.

```dockerfile
USER nobody
```
Drops from root to the `nobody` user. Running as root inside a container is a
security risk — if the container is compromised, root inside the container maps to
root outside (on older kernels without user namespaces).

```dockerfile
EXPOSE 8000
CMD ["uvicorn", "main_final:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```
`EXPOSE` is documentation — it does not publish the port. `--host 0.0.0.0` binds to
all interfaces inside the container (required for Docker port mapping; without it,
uvicorn would only accept connections from localhost inside the container).

### Full code block — `demo/Dockerfile.backend`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Layer-cache optimisation: install deps before copying source
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Source is copied after pip so code changes don't bust the pip cache layer
COPY . .

# Drop privileges
USER nobody

EXPOSE 8000
CMD ["uvicorn", "main_final:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

---

## Beat 8 — Frontend Dockerfile (Multi-Stage)

**File:** `demo/Dockerfile.frontend`  
**Pattern:** Two-stage Node build — builder (~700 MB) discarded; runner (~250 MB) shipped.

### Line-by-line explanation

```dockerfile
FROM node:20-alpine AS builder
```
`AS builder` names this stage. Stage 2 references it with `--from=builder`. Alpine
Linux is the smallest Node base image — ~40 MB compressed vs ~350 MB for Debian slim.

```dockerfile
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
```
Same layer-cache pattern as the backend: copy the lockfile first, install, then copy
source. `npm ci` (clean install) is faster than `npm install` because it reads the
lockfile exactly without resolution. `--omit=dev` skips linting and type-checking
tools not needed for the build.

```dockerfile
COPY . .
RUN npm run build
```
`npm run build` calls `next build`, which compiles TypeScript, bundles assets, and
(when `output: "standalone"` is set in `next.config.js`) produces a self-contained
`/.next/standalone/` directory with a `server.js` entrypoint.

```dockerfile
FROM node:20-alpine AS runner
```
A fresh Alpine image. None of the `node_modules`, source files, or build tools from
the `builder` stage are present — Docker starts from scratch for this stage.

```dockerfile
RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid  1001 nextjs
```
Creates a dedicated system user/group for the Next.js process. Using a specific UID/GID
(1001) makes permissions predictable across container restarts and allows volume mounts
with correct ownership.

```dockerfile
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
```
`--from=builder` copies files from the named builder stage. Only three directories
are needed at runtime: the standalone server bundle, the static assets, and the
public directory. Everything else (source, `node_modules`, TypeScript) is discarded.
`--chown=nextjs:nodejs` sets ownership so the non-root user can read the files.

```dockerfile
CMD ["node", "server.js"]
```
`server.js` is generated by `next build` when `output: "standalone"` is configured.
It is a minimal HTTP server that does not require Next.js to be installed — only
Node.js is needed in the runner stage, which is why the image is ~250 MB instead of ~700 MB.

### Full code block — `demo/Dockerfile.frontend`

```dockerfile
# Stage 1: builder — compiles the Next.js app
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY . .
RUN npm run build   # Requires output: "standalone" in next.config.js

# Stage 2: runner — only the compiled output, no build tools
FROM node:20-alpine AS runner

WORKDIR /app

RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid  1001 nextjs

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
```

---

## Beat 9 — Lab-Level `docker-compose.yml`

**File:** `demo/docker-compose.yml`  
**What it does:** Defines the `api` and `web` services, wires their port mappings,
injects environment variables, and ensures `web` starts only when `api` is healthy.

### Line-by-line explanation

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8001:8000"
```
`context: .` sets the Docker build context to the current directory (`demo/`).
`ports: "8001:8000"` maps host port 8001 → container port 8000. The demo uses 8001
to avoid collision with the Lab's `api` service on 8000.

```yaml
    environment:
      - NEO4J_URI=bolt://host.docker.internal:7688
      - WEAVIATE_URL=http://host.docker.internal:8081
```
`host.docker.internal` is a Docker-provided hostname that resolves to the host
machine's IP from inside a container. It allows the `api` container to reach Neo4j
and Weaviate that are running on the host (not in Docker). On Linux, you may need to
add `--add-host=host.docker.internal:host-gateway` to the service if this hostname
is not automatically available.

```yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/readyz"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 40s
```
Docker runs this command inside the container every `interval` seconds.
`curl -f` exits non-zero for HTTP error responses (4xx, 5xx). `start_period` gives
the container 40 s to start before the health check counts failures — this accounts
for model loading time. After 3 consecutive failures, Docker marks the container
`unhealthy`.

```yaml
  web:
    depends_on:
      api:
        condition: service_healthy
```
`condition: service_healthy` tells Compose to wait until `api` passes its
healthcheck before starting `web`. Without this, `web` might start and make
requests to `api` before the model has finished loading.

```yaml
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8001
```
`localhost` is the **host machine** from the browser's perspective. Even though `web`
and `api` are on the same Docker network, the browser is not — it runs on the host.
The Next.js server-side rendering could use `http://api:8000` (Docker-internal DNS),
but client-side fetch calls must use `http://localhost:8001`.

```yaml
networks:
  default:
    name: m10-demo-net
```
Giving the network an explicit name prevents Compose from generating a random name.
This makes `docker network inspect m10-demo-net` usable for debugging.

### Full code block — `demo/docker-compose.yml`

```yaml
services:

  api:
    build:
      context: .
      dockerfile: Dockerfile.backend
    image: movie-api:local
    ports:
      - "8001:8000"
    environment:
      - NEO4J_URI=bolt://host.docker.internal:7688
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - WEAVIATE_URL=http://host.docker.internal:8081
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/readyz"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 40s

  web:
    build:
      context: ./frontend
      dockerfile: ../Dockerfile.frontend
    image: movie-web:local
    ports:
      - "3001:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8001
    depends_on:
      api:
        condition: service_healthy

networks:
  default:
    name: m10-demo-net
```

---

## Shutdown Cell

**Purpose:** Release all resources — uvicorn processes, Docker Compose stack, and
kernel memory — at the end of the session or whenever a clean slate is needed.

### Line-by-line explanation

```python
pkill_result = subprocess.run(
    ["pkill", "-f", "uvicorn"],
    capture_output=True, text=True
)
```
`pkill -f pattern` sends `SIGTERM` to every process whose full command-line string
matches the pattern. `-f uvicorn` matches all uvicorn processes started by this
notebook. Return code `0` = processes were killed; `1` = no matching processes found
(not an error).

```python
dc_result = subprocess.run(
    ["docker", "compose", "down", "--remove-orphans"],
    cwd=str(compose_dir),
    capture_output=True, text=True
)
```
`docker compose down` stops and removes the containers defined in `docker-compose.yml`.
`--remove-orphans` also removes any containers left over from a previous Compose
configuration that are no longer defined. `cwd=` runs the command from the directory
that contains `docker-compose.yml`.

```python
get_ipython().run_line_magic("reset", "-f")
```
IPython magic equivalent of `%reset -f`. It drops all user-defined variables from
the kernel's namespace without prompting — this frees the memory held by the spaCy
model and the flan-t5-base generator (each is 500 MB – 3 GB). Wrapped in `try/except
NameError` so it silently skips when run outside Jupyter (e.g. in a plain Python
script).

### Full code block — Shutdown Cell

```python
import subprocess
import pathlib
import time

print("=" * 60)
print("SHUTDOWN: freeing all demo resources")
print("=" * 60)

# Kill uvicorn processes started during the lab
pkill_result = subprocess.run(
    ["pkill", "-f", "uvicorn"],
    capture_output=True, text=True
)
if pkill_result.returncode in (0, 1):
    print("  [OK]  uvicorn processes stopped (or none were running)")
else:
    print(f"  [WARN] pkill returned {pkill_result.returncode}: {pkill_result.stderr.strip()}")

time.sleep(1)  # Allow OS to reclaim the port

# Stop the demo Docker Compose stack
compose_dir = pathlib.Path("demo")
if (compose_dir / "docker-compose.yml").exists():
    dc_result = subprocess.run(
        ["docker", "compose", "down", "--remove-orphans"],
        cwd=str(compose_dir),
        capture_output=True, text=True
    )
    if dc_result.returncode == 0:
        print("  [OK]  docker compose stack stopped")
    else:
        print(f"  [WARN] docker compose down: {dc_result.stderr.strip()}")
else:
    print("  [SKIP] docker-compose.yml not found — nothing to stop")

# Release kernel memory held by large model objects
try:
    get_ipython().run_line_magic("reset", "-f")  # type: ignore[name-defined]
    print("  [OK]  kernel variables cleared")
except NameError:
    pass  # Running outside Jupyter — skip silently

print()
print("=" * 60)
print("SHUTDOWN COMPLETE — kernel memory released, containers stopped.")
print("Re-run 'Environment Verification' cell to start a fresh session.")
print("=" * 60)
```

---

*End of walkthrough. All source files live under `demo/` in the lab directory.*
