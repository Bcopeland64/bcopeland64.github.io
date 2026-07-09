# Support Instructor Demo — Code Walkthrough

This document walks every line of [support_instructor_demo.ipynb](support_instructor_demo.ipynb). Each section starts with a line-by-line explanation, then shows the full code block at the end so you can copy-paste it cleanly.

The demo retrieves cooking recipes three ways — BM25, dense, hybrid — to mirror the Module 8A lab on a different domain. Read alongside the notebook.

---

## 0. Setup — imports and Weaviate client

### Why this section exists
We need a configured Python environment connected to a running Weaviate container before any retrieval can happen. The container must be on port **8090** (not the default 8080) so learners running the lab on 8080 do not collide with the demo machine.

### Line-by-line

- `import json` — Python's built-in JSON library. We use `json.loads(line)` to parse each line of the JSONL file into a Python dict, and `json.dumps(doc)` to convert a dict back into a JSON string for writing. JSONL (JSON Lines) is a format where each line is a self-contained JSON object — it's streaming-friendly because you can process one line at a time without loading the whole file.
- `from pathlib import Path` — `Path` objects represent filesystem paths as proper objects rather than plain strings. Instead of error-prone string concatenation like `"data/" + filename + ".jsonl"`, you write `Path("data") / filename`. The key method we use is `CORPUS_PATH.exists()` (check if the file is already on disk) and `CORPUS_PATH.open()` (open it for reading or writing). This is the modern, preferred way to handle paths in Python 3.
- `import weaviate` — the official Weaviate Python client, version 4. Weaviate is the vector database we're using to store documents and run BM25, dense, and hybrid searches. The v4 client has a noticeably different API from v3 — most calls are method-chained on collection handles rather than passed as raw dicts.
- `from weaviate.classes.config import Configure, DataType, Property` — three helpers for defining the database schema. `Configure` is a factory that builds vectorizer settings (we'll use `Configure.Vectorizer.none()` to say "we supply our own vectors"). `DataType` is an enum with options like `DataType.TEXT`, `DataType.INT`, `DataType.NUMBER` — tells Weaviate how to index each field. `Property` wraps a field name and its `DataType` into a single schema declaration.
- `from weaviate.classes.query import MetadataQuery` — a small config object that tells Weaviate which relevance metadata to return alongside each result. Without it, fields like `obj.metadata.score` and `obj.metadata.distance` come back as `None`. We pass `MetadataQuery(score=True)` for BM25/hybrid and `MetadataQuery(distance=True)` for pure-vector search.
- `from sentence_transformers import SentenceTransformer` — loads the Sentence Transformers library, which wraps a pretrained transformer model and exposes a simple `.encode(text)` method that returns a fixed-length embedding vector. We use it to convert both documents (at ingest) and queries (at search time) into the same 384-dimensional vector space.
- `from datasets import load_dataset` — Hugging Face's `datasets` library. It can fetch, cache, and stream datasets from the Hub or local files. Once a dataset is downloaded it is stored in `~/.cache/huggingface/datasets/` so subsequent runs are instant even offline.
- `WEAVIATE_HTTP_PORT = 8090` — the REST/HTTP port the Weaviate container is listening on. In the `docker run` command this is the left side of the `-p 8090:8080` mapping (host port 8090 → container port 8080). We use 8090 instead of the default 8080 so the demo machine's container doesn't collide with each learner's container if they're running locally.
- `WEAVIATE_GRPC_PORT = 50061` — the gRPC port. The Weaviate v4 client uses gRPC (a high-performance binary RPC protocol) for bulk batch inserts because it is significantly faster than REST for that workload. The default gRPC port is 50051; we use 50061 for the same collision-avoidance reason as above.
- `CLASS_NAME = "Post"` — the name of our Weaviate collection (Weaviate calls these "classes"). We intentionally match the lab's class name so that the same retrieval helper functions — `bm25_search`, `dense_search`, `hybrid_search` — run against the demo corpus unchanged. Learners can directly compare their lab code to this demo.
- `CORPUS_PATH = Path("demo_corpus.jsonl")` — the local file that holds the 200 reshaped recipes. We generate it once on first run and reuse it on subsequent runs, which keeps the demo fast. Using a `Path` object (not a raw string) lets us call `.exists()`, `.open()`, etc. cleanly throughout.
- `EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"` — the Hugging Face model ID for our sentence embedder. It produces 384-dimensional vectors and is fast on CPU (~80 MB download, cached after first use). Using a named constant means if we ever swap models, we change exactly one line and every `.encode()` call updates automatically.
- `client = weaviate.connect_to_local(port=WEAVIATE_HTTP_PORT, grpc_port=WEAVIATE_GRPC_PORT)` — opens the REST connection on port 8090 and the gRPC connection on port 50061 to the local Weaviate container. Until you call this, no interaction with the database is possible. The resulting `client` object is what all subsequent collection and query calls hang off of.
- `assert client.is_ready(), "..."` — calls Weaviate's health-check endpoint (`GET /v1/.well-known/ready`). If the container is not running or hasn't finished starting, `is_ready()` returns `False` and the `assert` raises an `AssertionError` with a descriptive message immediately. This "fail fast" pattern is intentional — much easier to diagnose than a cryptic `ConnectionRefusedError` 20 lines later when the first query fires.
- `print("Weaviate ready on port", WEAVIATE_HTTP_PORT)` — a visible confirmation line in the cell output. When demoing live, this tells the instructor at a glance that setup completed cleanly before moving to the next cell.

### Full code block

```python
import json
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

WEAVIATE_HTTP_PORT = 8090
WEAVIATE_GRPC_PORT = 50061  # non-default so it doesn't collide with the lab's 50051
CLASS_NAME = "Post"
CORPUS_PATH = Path("demo_corpus.jsonl")
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = weaviate.connect_to_local(port=WEAVIATE_HTTP_PORT, grpc_port=WEAVIATE_GRPC_PORT)
assert client.is_ready(), "Weaviate is not reachable on port 8090. Check the docker container."
print("Weaviate ready on port", WEAVIATE_HTTP_PORT)
```

---

## 1. Frame — show the corpus (0:00 – 0:03)

### Why this section exists
The demo's opening beat. Show learners the corpus is the same *shape* as the lab corpus (a JSONL of `Post`-schema rows) but a completely different *surface* (recipes, not Stack Exchange posts). They cannot copy this demo as their submission.

### Line-by-line

- `if not CORPUS_PATH.exists():` — checks whether `demo_corpus.jsonl` already exists on disk before doing any downloading or reshaping. This makes the cell **idempotent** — running it a second time is instant and produces the same result. It also means you can re-run the notebook from the top without re-downloading the dataset.
- `RECIPES_PARQUET = "https://..."` — a direct URL to the Hugging Face Hub's auto-generated Parquet snapshot of the dataset. Parquet is a columnar binary format that `datasets` can read efficiently. We bypass the dataset's native loading script because `datasets >= 3.0` refuses to execute remote Python scripts for security reasons; pointing straight at the Parquet file sidesteps that restriction without changing any of our downstream `ds.select()` / `r["name"]` code.
- `ds = load_dataset("parquet", data_files=RECIPES_PARQUET, split="train")` — tells `load_dataset` to treat the file as Parquet (first argument) rather than a named Hub dataset. `split="train"` selects the training split, which is the only split in this dataset. The returned `ds` behaves like a list of dicts — each row is a recipe with keys like `name`, `ingredients`, and `steps`.
- `recipes = ds.select(range(200))` — slices the first 200 rows out of the full dataset. `range(200)` produces indices 0–199. Keeping 200 matches the demo's target size and keeps ingest fast (~30 seconds on CPU). `ds.select()` is Hugging Face's preferred way to take a subset — it preserves the `Dataset` type and all its methods.
- `with CORPUS_PATH.open("w") as f:` — opens `demo_corpus.jsonl` for writing (`"w"` mode, which creates the file or truncates it if it exists). The `with` block is a **context manager**: Python guarantees the file is closed and flushed even if an exception is raised inside the block. Always prefer `with open(...)` over manual `f.close()`.
- `for i, r in enumerate(recipes):` — iterates over every recipe row. `enumerate` yields both the index `i` (0, 1, 2 …) and the row dict `r`. We need `i` to construct a unique string ID for each document.
- `doc = { ... }` — assembles a Python dict that matches the lab's `Post` schema field-for-field. Each key is a field name from the Weaviate schema:
  - `"id": f"recipes:{i}"` — a namespaced unique identifier. Prefixing with `"recipes:"` ensures these IDs can never accidentally collide with the lab's Stack Exchange IDs (which use a different prefix). Example: `"recipes:0"`, `"recipes:1"`.
  - `"subset": "recipes"` — a category label. The lab schema includes this field so that queries can filter to a specific subset of the corpus (e.g., only Stack Overflow posts, or only recipes). Here we always set it to `"recipes"`.
  - `"title": r["name"]` — the recipe's display name as it appears in the dataset, e.g. `"Harissa-Roasted Carrots"`. This is what gets printed in the ranked results.
  - `"question_text": r["ingredients"]` — the schema reserves this field for the "question-like" content of a document. Recipes don't have questions, so we repurpose it for the ingredient list. The schema shape stays valid; the meaning is domain-adapted.
  - `"answer_text": r["steps"]` — similarly, the cooking instructions fill the "answer" slot. The content is different from Stack Exchange but the key name is the same, so the same retrieval functions work on both corpora.
  - `"text": f"{r['name']}\n\n{r['ingredients']}\n\nSteps: {r['steps']}"` — the single concatenated field that BM25 and the dense embedder both operate on. **This is the most important field for retrieval.** If a word or phrase isn't in `text`, neither BM25 nor the embedder can find it. We join all three sections with blank lines so the structure is readable if you print the field.
  - `"license_version": "see dataset card"` — required field in the schema. We fill it with a placeholder and point to the dataset card, which is the authoritative source for licensing info.
  - `"source_url": "..."` — the dataset's canonical URL. Included for attribution and so that anyone reading the corpus file knows where the data came from.
- `f.write(json.dumps(doc) + "\n")` — `json.dumps(doc)` serializes the Python dict to a JSON string (a single line with no embedded newlines). Appending `"\n"` writes the line terminator. The resulting file has exactly one JSON object per line — that's the JSONL convention. Any tool that can read one JSON object can process the file line-by-line without needing to load the whole thing into memory at once.
- `print(f"Wrote {CORPUS_PATH} with 200 recipes.")` — a visible completion message so the instructor can confirm ingest succeeded before moving on.
- `else: print(f"{CORPUS_PATH} already exists — reusing.")` — the `else` branch of the outer `if not CORPUS_PATH.exists()` block. Tells the room explicitly why no download happened, so it doesn't look like the cell silently failed.
- `with CORPUS_PATH.open() as f: for _ in range(5): ...` — opens the file for reading (default mode) and reads the first 5 lines. `f.readline()` returns one line at a time including its trailing newline; `json.loads()` parses it back to a dict. We print just `id` and `title` — not the full ingredient list — so the output stays compact when projected on screen. `_` is a conventional variable name for "I don't need this value" (here, we don't need the loop index, only the loop body).

### Full code block

```python
if not CORPUS_PATH.exists():
    # The dataset has a Python loading script which datasets>=3.0 rejects;
    # load the Hub's auto-converted parquet snapshot instead.
    RECIPES_PARQUET = "https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite/resolve/refs%2Fconvert%2Fparquet/1.0.0/recipe_nlg_lite-train.parquet"
    ds = load_dataset("parquet", data_files=RECIPES_PARQUET, split="train")
    recipes = ds.select(range(200))

    with CORPUS_PATH.open("w") as f:
        for i, r in enumerate(recipes):
            doc = {
                "id": f"recipes:{i}",
                "subset": "recipes",
                "title": r["name"],
                "question_text": r["ingredients"],
                "answer_text": r["steps"],
                "text": f"{r['name']}\n\n{r['ingredients']}\n\nSteps: {r['steps']}",
                "license_version": "see dataset card",
                "source_url": "https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite",
            }
            f.write(json.dumps(doc) + "\n")
    print(f"Wrote {CORPUS_PATH} with 200 recipes.")
else:
    print(f"{CORPUS_PATH} already exists — reusing.")

with CORPUS_PATH.open() as f:
    for _ in range(5):
        row = json.loads(f.readline())
        print(f"- {row['id']}: {row['title']}")
```

---

## 2. Schema + ingest (0:03 – 0:08)

### 2a. `create_schema`

#### Why this section exists
Weaviate needs to know the field layout before we can insert anything. We define a `Post` class with **no vectorizer** because we supply vectors manually — that keeps the demo offline-capable and removes the "is the OpenAI key set?" failure mode.

#### Line-by-line

- `def create_schema():` — wraps the schema creation logic in a named function so the instructor can call it again cleanly if the demo needs to restart. Without the function wrapper, re-running the cell would require scrolling back to find it.
- `if client.collections.exists(CLASS_NAME):` — checks whether a collection named `"Post"` is already registered in Weaviate before trying to create one. If we called `create` on an already-existing class, Weaviate would return an error. This guard makes the function safe to call multiple times.
- `client.collections.delete(CLASS_NAME)` — drops the existing `"Post"` collection and all its data. This is intentionally a hard reset rather than an incremental update — trying to merge a new schema onto old data creates subtle bugs. A clean slate is simpler and more predictable.
- `client.collections.create(...)` — registers the new collection in Weaviate with the name, vectorizer settings, and field definitions passed as keyword arguments. This is the Weaviate v4 way to define a schema; in v3 you would have passed a raw dict to `client.schema.create_class(...)`.
- `name=CLASS_NAME` — sets the collection name to `"Post"` (the value of our constant). This must match what the lab uses so that `client.collections.get("Post")` works identically in both the demo and learner code.
- `vectorizer_config=Configure.Vectorizer.none()` — tells Weaviate **not** to run its own internal vectorizer on new objects. Instead we will supply a pre-computed vector ourselves in each `batch.add_object(vector=...)` call. This is the critical line that keeps the demo fully offline — no OpenAI API key, no Cohere key, no external HTTP calls during indexing.
- `properties=[Property(name=..., data_type=DataType.TEXT), ...]` — a list of eight field declarations. Each `Property` names a field and assigns a type. `DataType.TEXT` tells Weaviate to treat the value as a string and build an inverted index over it (which is what makes BM25 possible). Notice that the field is named `doc_id`, not `id` — Weaviate reserves the name `id` for its own internal UUID primary key, so we must use a different name for our corpus identifier.
- `print(f"Created class {CLASS_NAME}.")` — a visible confirmation line. If you don't see this printed, the `delete` or `create` call failed silently and retrieval will not work.

#### Full code block

```python
def create_schema():
    if client.collections.exists(CLASS_NAME):
        client.collections.delete(CLASS_NAME)
    client.collections.create(
        name=CLASS_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="subset", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="question_text", data_type=DataType.TEXT),
            Property(name="answer_text", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="license_version", data_type=DataType.TEXT),
            Property(name="source_url", data_type=DataType.TEXT),
        ],
    )
    print(f"Created class {CLASS_NAME}.")

create_schema()
```

### 2b. Load the embedder

#### Why this section exists
We need a sentence encoder to (a) embed each document at ingest and (b) embed each query at search time. Loading it once up-front means the first query is instant, not "wait 30 seconds while the model downloads in front of the class."

#### Line-by-line

- `embedder = SentenceTransformer(EMBEDDER_NAME)` — instantiates the embedding model. On first run, this downloads the model weights (~80 MB) from Hugging Face and saves them to `~/.cache/huggingface/hub/`. On every subsequent run, it reads from the cache — no internet required. The resulting `embedder` object exposes an `.encode(text)` method that accepts a string (or list of strings) and returns a NumPy array of shape `(384,)` — a 384-dimensional embedding vector. We store it in a module-level variable so both `index_corpus` (at ingest time) and the search functions (at query time) share the exact same model instance, guaranteeing that documents and queries are embedded in the same vector space.
- `print("Loaded embedder:", EMBEDDER_NAME)` — confirmation that the model loaded without errors. If the cache is corrupt or the download failed, `SentenceTransformer(...)` would raise an exception before this line; seeing it printed means the embedder is ready.

#### Full code block

```python
embedder = SentenceTransformer(EMBEDDER_NAME)
print("Loaded embedder:", EMBEDDER_NAME)
```

### 2c. `index_corpus`

#### Why this section exists
Stream the JSONL into Weaviate with a pre-computed vector for each row. Roughly 30 seconds for 200 docs on CPU.

#### Line-by-line

- `def index_corpus(path: Path):` — wraps the ingest logic in a function parameterized by the corpus file path. This means if we ever want to index a different JSONL file, we just pass a different `Path` — none of the logic changes.
- `rows = [json.loads(line) for line in path.open()]` — reads every line of the JSONL file and parses each one into a Python dict. The result is a list of 200 dicts, one per recipe. At this scale it's fine to hold the whole thing in memory; for a 200k-document corpus you'd process lines in chunks instead.
- `texts = [r["text"] for r in rows]` — builds a list of the `text` field from each row — that is the concatenated `name + ingredients + steps` string we wrote during corpus creation. **Only this field is embedded.** BM25 also searches `text` (because Weaviate's BM25 index covers all TEXT properties by default, but `text` is the richest one). Keeping the embed target and the BM25 target aligned means both retrieval modes search the same content.
- `vectors = embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)` — encodes all 200 texts in one call. Breaking it down:
  - `batch_size=32` — the encoder processes 32 texts at a time internally. Batching is faster than encoding one-by-one because the model runs as a matrix operation over the whole batch. 32 is a safe default on CPU; on GPU you'd raise this to 64 or 128.
  - `show_progress_bar=True` — prints a `tqdm` progress bar. Useful during the live demo so the room sees forward progress instead of a frozen cell for 30 seconds.
  - `normalize_embeddings=True` — divides each vector by its L2 norm (its length), producing **unit vectors** where every vector has length exactly 1. When all vectors are unit-length, cosine similarity reduces to a simple dot product — mathematically cleaner and slightly faster. Weaviate's default distance metric for `near_vector` is cosine, so this keeps the two consistent.
  - The return value is a NumPy array of shape `(200, 384)` — 200 rows (one per document), 384 columns (one per embedding dimension).
- `posts = client.collections.get(CLASS_NAME)` — retrieves a handle to the `"Post"` collection. Think of this as the Python-side proxy object for the Weaviate collection; it's cheap to call and does not copy any data. All subsequent inserts and queries go through this handle.
- `with posts.batch.dynamic() as batch:` — opens a **dynamic batch context**. Weaviate's dynamic batcher monitors how quickly the server is accepting objects and automatically adjusts how many objects it sends per network round-trip. You just call `batch.add_object(...)` in a loop; the batcher handles chunking, retries, and flushing. The `with` block flushes and closes the batch on exit.
- `for r, vec in zip(rows, vectors):` — iterates over rows and their corresponding vectors in lock-step. `zip` pairs element 0 of `rows` with element 0 of `vectors`, element 1 with element 1, and so on. This is safe because `encode` preserves input order, so `vectors[i]` is always the embedding of `rows[i]["text"]`.
- `batch.add_object(properties={...}, vector=vec.tolist())` — queues one document for insertion:
  - `properties=` is a dict that maps schema field names to their values for this document. Note `"doc_id": r["id"]` — we rename the source field `"id"` to `"doc_id"` here because, as mentioned in the schema section, Weaviate reserves the name `"id"` for its own UUID primary key.
  - `vector=vec.tolist()` — converts the NumPy array for this document into a plain Python list of floats. Weaviate's Python client requires a native Python list here, not a NumPy array. `.tolist()` does that conversion in one call.
- `print(f"Indexed {len(rows)} documents.")` — prints after the `with` block exits, meaning the batch has fully flushed. If this line prints `200`, every document made it into Weaviate successfully.

#### Full code block

```python
def index_corpus(path: Path):
    rows = [json.loads(line) for line in path.open()]
    texts = [r["text"] for r in rows]
    vectors = embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    posts = client.collections.get(CLASS_NAME)
    with posts.batch.dynamic() as batch:
        for r, vec in zip(rows, vectors):
            batch.add_object(
                properties={
                    "doc_id": r["id"],
                    "subset": r["subset"],
                    "title": r["title"],
                    "question_text": r["question_text"],
                    "answer_text": r["answer_text"],
                    "text": r["text"],
                    "license_version": r["license_version"],
                    "source_url": r["source_url"],
                },
                vector=vec.tolist(),
            )
    print(f"Indexed {len(rows)} documents.")

index_corpus(CORPUS_PATH)
```

### 2d. Verify the aggregate count

#### Why this section exists
A single line of independent verification. If this prints `200`, ingest worked end-to-end. If it doesn't, stop and fix before continuing.

#### Line-by-line

- `posts = client.collections.get(CLASS_NAME)` — re-fetches the collection handle. Collection handles are lightweight and it's fine to create multiple throughout the notebook; each call is just a local Python object, not a database round-trip.
- `count = posts.aggregate.over_all(total_count=True).total_count` — calls Weaviate's aggregate API, which runs a server-side count query. `over_all(total_count=True)` asks for the total object count across the entire collection (no filters). The result is a response object; `.total_count` extracts the integer from it. This is the Weaviate equivalent of `SELECT COUNT(*) FROM Post`.
- `print(f"Aggregate count: {count}")` — prints the count so the room can see it. If it says `200`, ingest was successful end-to-end.
- `assert count == 200, "Expected 200 documents — did ingest fail?"` — a hard stop if the count is wrong. An `AssertionError` here is a clear signal that something broke during the batch insert (network drop, schema mismatch, etc.) and the demo should not continue until it's fixed. It's far better to crash loudly here than to silently run retrieval on an incomplete corpus and produce confusing results.

#### Full code block

```python
posts = client.collections.get(CLASS_NAME)
count = posts.aggregate.over_all(total_count=True).total_count
print(f"Aggregate count: {count}")
assert count == 200, "Expected 200 documents — did ingest fail?"
```

---

## 3. Retrieval functions

### Why this section exists
Three thin wrappers — one per retrieval mode — plus a shared formatter. Keeping them short (~5 lines each) lets the room see the API surface, not boilerplate.

### Line-by-line — `_format`

- `def _format(results):` — the leading underscore in `_format` is a Python convention meaning "this is a private helper, not part of the public API." It signals to other developers (and to the instructor) that this function is only meant to be called by the three search functions in this same module.
- `rows = []` — initializes an empty list that will accumulate one dict per result. We build it up inside the loop and return it at the end.
- `for i, obj in enumerate(results.objects, start=1):` — iterates over the result objects Weaviate returned. `results.objects` is a list ordered by relevance (most relevant first). `enumerate(..., start=1)` gives us a 1-based index `i` for display purposes — rank 1 is the top result, rank 5 is the last (assuming `k=5`).
- `score = obj.metadata.score if obj.metadata.score is not None else obj.metadata.distance` — handles a key asymmetry in Weaviate's API: BM25 and hybrid queries populate `obj.metadata.score` (a number where **higher is better**), but pure-vector `near_vector` queries populate `obj.metadata.distance` (a number where **lower is better** — it's a cosine distance). By checking which field is non-`None`, one helper function works for all three retrieval modes. The `_format` function doesn't need to know which mode produced the results.
- `rows.append({"rank": i, "title": obj.properties["title"], "score": round(float(score), 4)})` — builds the display dict for this result. We keep only `rank`, `title`, and `score` — not the full recipe text — because that's all we want to show on screen. `round(..., 4)` keeps 4 decimal places so columns align neatly when displayed. `float(score)` converts NumPy scalar types (which Weaviate sometimes returns) to a plain Python float before rounding.
- `return rows` — returns the accumulated list of dicts to the caller (`bm25_search`, `dense_search`, or `hybrid_search`), which passes it directly to `show`.

### Line-by-line — `bm25_search`

- `posts = client.collections.get(CLASS_NAME)` — gets the collection handle. Cheap call — creates a local Python proxy, no network request.
- `res = posts.query.bm25(query=query, limit=k, return_metadata=MetadataQuery(score=True))` — runs a BM25 (Best Match 25) keyword search against the collection. BM25 is a ranking function that scores documents by how often the query terms appear in them, adjusted for document length and how rare those terms are across the corpus (IDF — Inverse Document Frequency). `limit=k` caps the number of results. `return_metadata=MetadataQuery(score=True)` explicitly requests that Weaviate include the BM25 relevance score for each result — without this, `obj.metadata.score` would be `None` and `_format` would crash.
- `return _format(res)` — passes the raw Weaviate response to the shared formatter, which pulls out rank, title, and score.

### Line-by-line — `dense_search`

- `qvec = embedder.encode(query, normalize_embeddings=True).tolist()` — embeds the search query using the same model and normalization settings that were used to embed the documents at ingest. **This symmetry is critical**: if the documents were embedded with normalization and the query isn't (or vice versa), the dot-product distances would be off-scale and results would be meaningless. `.tolist()` converts the NumPy array to a Python list, as required by the Weaviate client.
- `res = posts.query.near_vector(near_vector=qvec, limit=k, return_metadata=MetadataQuery(distance=True))` — runs an **Approximate Nearest Neighbor (ANN)** search. Weaviate finds the `k` stored document vectors that are closest to `qvec` in cosine distance. This is "dense" retrieval because every word in the query is compressed into the single dense 384-dimensional vector — there are no keyword lookups. `return_metadata=MetadataQuery(distance=True)` requests the cosine distance (0 = identical, 2 = opposite); note it's `distance`, not `score`.
- `return _format(res)` — formats results the same way as BM25, even though the metadata field is different (`distance` not `score`). `_format` handles that transparently.

### Line-by-line — `hybrid_search`

- `alpha: float = 0.5` — the mixing parameter that controls how much weight to give each retrieval signal. `alpha=0.0` means the result is **pure BM25** (the vector signal is ignored). `alpha=1.0` means the result is **pure dense** (keyword signal is ignored). `alpha=0.5` weights them equally. **This is the key parameter learners experiment with in the lab** — they observe how results shift as they move `alpha` toward 0 or 1 for different query types.
- `qvec = embedder.encode(query, normalize_embeddings=True).tolist()` — embeds the query exactly as in `dense_search`. Hybrid search needs the query in two forms: as a raw string (for the BM25 branch) and as an embedding vector (for the dense branch). We compute both before calling the API.
- `res = posts.query.hybrid(query=query, vector=qvec, alpha=alpha, limit=k, return_metadata=MetadataQuery(score=True))` — Weaviate runs both BM25 and ANN internally, then merges their ranked lists using Reciprocal Rank Fusion (RRF) weighted by `alpha`. The fused score is a blended relevance number — not a raw BM25 score and not a raw cosine distance, but a combined signal. `return_metadata=MetadataQuery(score=True)` requests this fused score.
- `return _format(res)` — formats the hybrid results the same way as the other two modes.

### Line-by-line — `show`

- `print(f"\n=== {header} ===")` — prints a blank line followed by a clearly delineated section header. The `\n` before `===` creates visual breathing room between result blocks when multiple queries run back-to-back in the same cell output. The `header` string includes the retrieval mode and query text (e.g., `"BM25 — 'harissa paste'"`), so you can scan the output quickly on a projected screen.
- `for r in rows: print(f"  {r['rank']:>2}. [{r['score']:>7}]  {r['title']}")` — prints one line per result using fixed-width format specifiers. `{r['rank']:>2}` right-aligns the rank in a 2-character column (so `1` and `10` line up). `{r['score']:>7}` right-aligns the score in a 7-character column with four decimal places. The result is a table-like layout that makes it easy to compare rankings across three retrieval modes when projected on screen.

### Full code block

```python
def _format(results):
    rows = []
    for i, obj in enumerate(results.objects, start=1):
        score = obj.metadata.score if obj.metadata.score is not None else obj.metadata.distance
        rows.append({"rank": i, "title": obj.properties["title"], "score": round(float(score), 4)})
    return rows

def bm25_search(query: str, k: int = 5):
    posts = client.collections.get(CLASS_NAME)
    res = posts.query.bm25(query=query, limit=k, return_metadata=MetadataQuery(score=True))
    return _format(res)

def dense_search(query: str, k: int = 5):
    posts = client.collections.get(CLASS_NAME)
    qvec = embedder.encode(query, normalize_embeddings=True).tolist()
    res = posts.query.near_vector(near_vector=qvec, limit=k, return_metadata=MetadataQuery(distance=True))
    return _format(res)

def hybrid_search(query: str, alpha: float = 0.5, k: int = 5):
    posts = client.collections.get(CLASS_NAME)
    qvec = embedder.encode(query, normalize_embeddings=True).tolist()
    res = posts.query.hybrid(
        query=query,
        vector=qvec,
        alpha=alpha,
        limit=k,
        return_metadata=MetadataQuery(score=True),
    )
    return _format(res)

def show(rows, header):
    print(f"\n=== {header} ===")
    for r in rows:
        print(f"  {r['rank']:>2}. [{r['score']:>7}]  {r['title']}")
```

---

## 4. BM25 demo (0:08 – 0:15)

### Why this section exists
Show the room that **rare exact tokens dominate BM25 results**. The narrative beat: *"same pattern you'll see with `EACCES` in the lab."*

### Line-by-line

- `bm25_queries = ["harissa paste", "panko breadcrumbs"]` — both strings contain **rare, specific ingredient names** that appear in only a handful of recipes in the 200-document corpus. That rarity is the whole point: BM25's IDF (Inverse Document Frequency) component gives very high weight to terms that appear in few documents. When a query term is rare, BM25 is extremely confident that a document containing it is relevant. The top-ranked BM25 results will be recipes that literally mention "harissa paste" or "panko breadcrumbs" — and they'll be ranked far above everything else.
- The commented spares (`# "espelette pepper", "preserved lemon"`) are queries the instructor pre-tested on Sunday to confirm they produce a strong BM25 win. If a primary query produces flat or confusing results during the live demo — which can happen if the corpus slice doesn't contain enough matches — the instructor can uncomment a spare and swap it in without breaking the cell.
- `for q in bm25_queries:` — loops over each query, running both retrievers for each one. Putting the loop here means adding a new query to the list is a one-character change, not a new code block.
- `show(bm25_search(q), f"BM25 — {q!r}")` — runs the BM25 search and prints the labeled result block. `{q!r}` renders `q` with its quotes included (e.g., `'harissa paste'`), making the output self-describing.
- `show(dense_search(q), f"Dense — {q!r}  (for contrast)")` — runs the same query through the dense embedder. Because "harissa paste" is a specific proper noun that doesn't embed to a unique direction in the 384-dimensional space, the dense results often return semantically adjacent but lexically unrelated recipes (e.g., other roasted vegetable dishes). That contrast — BM25 nails the exact term, dense meanders — is the visual lesson the room takes away from this section.

### Full code block

```python
bm25_queries = [
    "harissa paste",
    "panko breadcrumbs",
    # spares (pre-tested Sunday):
    # "espelette pepper", "preserved lemon"
]

for q in bm25_queries:
    show(bm25_search(q), f"BM25 — {q!r}")
    show(dense_search(q), f"Dense — {q!r}  (for contrast)")
```

---

## 5. Dense demo (0:15 – 0:22)

### Why this section exists
Show the room that **embeddings retrieve by intent, not by word overlap**. Narrative beat: *"No shared words but the right recipes still surface."*

### Line-by-line

- `dense_queries = ["quick weeknight dinner with what's in the fridge", "comfort food for a cold day"]` — these queries are written in the way a person would naturally speak, using words that **do not literally appear** in any recipe title or ingredient list. Phrases like "quick weeknight" or "comfort food" are not ingredients or technique keywords — they're *intent* signals. BM25 scores documents by exact term overlap; since these words don't appear in the corpus, BM25 will produce low, uniform scores with essentially random ordering. The dense embedder, by contrast, maps the query into a vector neighborhood that includes soups, stews, short-prep pastas, and simple one-pan dinners — the kinds of recipes that semantically match the intent even without literal keyword overlap.
- The commented spare (`# "something light and refreshing for summer"`) follows the same logic — a seasonal intent phrase with no corpus keyword match — pre-tested to confirm a strong dense win.
- The loop runs BM25 **first** and dense **second** in this section, the opposite order from the BM25 demo. Showing BM25's failure before the dense success makes the contrast more dramatic: the room watches BM25 return irrelevant or low-confidence results, then sees dense surface the right answer. This ordering reinforces which method wins for which query type.

### Full code block

```python
dense_queries = [
    "quick weeknight dinner with what's in the fridge",
    "comfort food for a cold day",
    # spare:
    # "something light and refreshing for summer"
]

for q in dense_queries:
    show(bm25_search(q), f"BM25 — {q!r}  (for contrast)")
    show(dense_search(q), f"Dense — {q!r}")
```

---

## 6. Hybrid + alpha sweep (0:22 – 0:27)

### Why this section exists
**The α knob is the lab's central deliverable.** Show learners with their own eyes what changing α does. One query, three settings, side-by-side.

### Line-by-line

- `hybrid_query = "easy 30-minute pasta with chicken"` — this query is deliberately designed to contain **both** a keyword signal and an intent signal. The words `pasta` and `chicken` appear verbatim in several recipe names and ingredient lists, so BM25 can find relevant documents based on exact term match. The phrase `easy 30-minute` does not appear literally in any recipe, but it encodes an intent — short prep time, simple technique — that the dense embedder can map to similar-feeling recipes. A query with both signals lets the room observe what each `alpha` setting actually trades off.
- `for alpha in (0.0, 0.5, 1.0):` — runs the same query three times with different blending weights. These three values are the canonical anchor points for explaining the `alpha` parameter:
  - `alpha=0.0` — hybrid collapses to **pure BM25**. Only the keyword ranking signal matters; the vector is passed to Weaviate but gets zero weight. Results are ordered by how often `pasta` and `chicken` appear in each document, adjusted for document length. "Easy 30-minute" contributes nothing.
  - `alpha=1.0` — hybrid collapses to **pure dense**. Only the vector similarity matters; BM25 scoring is ignored entirely. Results reflect semantic intent — quick, simple dishes — regardless of whether the word `pasta` or `chicken` appears.
  - `alpha=0.5` — equal weighting. Weaviate fuses the BM25 rank list and the ANN rank list using Reciprocal Rank Fusion (RRF), giving each list the same influence. In practice this often yields the most balanced and useful results for mixed queries like this one.
- `show(hybrid_search(hybrid_query, alpha=alpha), f"Hybrid α={alpha} — {hybrid_query!r}")` — prints each result block with the alpha value in the header (e.g., `=== Hybrid α=0.5 — 'easy 30-minute pasta with chicken' ===`). Having the alpha value visible in every header means the room can compare three labeled result lists side-by-side without losing track of which is which.

### Full code block

```python
hybrid_query = "easy 30-minute pasta with chicken"

for alpha in (0.0, 0.5, 1.0):
    show(hybrid_search(hybrid_query, alpha=alpha), f"Hybrid α={alpha} — {hybrid_query!r}")
```

---

## 7. Pivot to lab (0:27 – 0:30)

### Why this section exists
End the demo cleanly: close the client (releases gRPC connections) and hand off to the lab.

### Line-by-line

- `client.close()` — explicitly closes both the REST (HTTP) and gRPC connections to the Weaviate container. The Weaviate v4 client holds open a long-lived gRPC channel for batch performance; if you don't close it, the Python kernel may print a `ResourceWarning` about unclosed connections when it shuts down. In a production application you'd always close the client in a `finally` block or use it as a context manager (`with weaviate.connect_to_local(...) as client:`). Here we just call it at the end of the notebook.
- `print("Closed Weaviate client. Demo complete.")` — a visible end-of-demo marker in the cell output. When projecting on screen, this line signals to the room that the walkthrough is done and the lab is about to begin. It also confirms that `client.close()` returned without raising an exception.

### Full code block

```python
client.close()
print("Closed Weaviate client. Demo complete.")
```

---

## 8. Teardown — stop the container (after class)

### Why this section exists
Once the demo is over, stop and remove the Weaviate container so the `docker run` command in section 0 succeeds on the next session. A stopped-but-still-present container blocks reuse of the `weaviate-si-demo` name with `Conflict. The container name "/weaviate-si-demo" is already in use.`

### Line-by-line

- `!docker stop weaviate-si-demo` — the `!` prefix is Jupyter's shell-escape: it runs the rest of the line as a bash command in a subprocess, not as Python. `docker stop` sends `SIGTERM` to the container's main process, waits up to 10 seconds for it to exit cleanly, then sends `SIGKILL` if it hasn't stopped. This is a graceful shutdown — Weaviate gets a chance to flush any in-memory state before terminating.
- `!docker rm weaviate-si-demo` — removes the stopped container's metadata record from Docker's list of known containers. Without this step, the container name `"weaviate-si-demo"` remains reserved even though the container isn't running. The next `docker run --name weaviate-si-demo ...` would fail with `Conflict. The container name "/weaviate-si-demo" is already in use`. Since the demo container has no volume mount (`-v` flag), removing it discards no persistent data — all data was held in memory inside the container anyway.

### Full code block

```python
# Tear down the Weaviate container so cell-1's `docker run` succeeds next time
# (otherwise it errors with "name already in use").
!docker stop weaviate-si-demo
!docker rm weaviate-si-demo
```

---

## Pre-demo checklist (Monday morning, ~10 min before class)

- [ ] Weaviate container running on port 8090: `curl -s localhost:8090/v1/.well-known/ready` returns 200.
- [ ] Aggregate query against `Post` shows count = 200.
- [ ] All ~30 demo queries pre-tested; copy-paste-ready.
- [ ] `all-MiniLM-L6-v2` embedder cached so first encode is instant.
- [ ] Screen-share resolution / font size verified.
- [ ] Lab guide URL on a sticky note for the closing pivot.

## Recovery plan

- **Weaviate dies mid-demo:** switch to the pre-recorded backup screencast (record one Sunday). Narrate over it — don't lose 5 minutes restarting Docker.
- **A query doesn't show the expected contrast:** skip it, use a spare. Don't argue with the model in front of the room.
- **Dense embed is slow:** confirm the embedder is loading from cache, not downloading. If it's downloading, stall with a one-paragraph recap of dense retrieval from the reading.
