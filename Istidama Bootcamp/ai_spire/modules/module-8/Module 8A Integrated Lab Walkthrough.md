# Pharma RAG Demo — Line-by-Line Walkthrough

> Companion document to [pharma_rag_demo.ipynb](pharma_rag_demo.ipynb). For each notebook section, this file walks the code one line at a time and then prints the full code block at the end of the section.

**Disclaimer:** every drug entry in the demo corpus is marked `MOCK — for educational demonstration only, not medical advice`. The corpus stays in the instructor's personal demo folder; it is not vetted medical information and must not be committed to any Module 8 repository.

---

## Section 0 — Imports and module-level model loading

The most expensive thing in this notebook is loading the two models. We do it **exactly once**, at the module level, so every later call to `embedder.encode()` or `generator(...)` is fast. If the model is reloaded on every call, CPU inference balloons from ~1–3 s to ~30 s.

- `import json` — used for serialising/deserialising the `pharma_corpus.jsonl` records.
- `import os` — imported defensively in case we need env-var lookups; not strictly used in the demo today.
- `from pathlib import Path` — gives us `Path("pharma_corpus.jsonl")` for path operations that are platform-safe.
- `import weaviate` — the Weaviate Python client (v4 API).
- `from weaviate.classes.config import Configure, Property, DataType` — v4-style schema-builder helpers. `Configure.Vectorizer.none()` tells Weaviate we'll be supplying vectors ourselves; `Property` and `DataType` describe each field.
- `from weaviate.classes.query import MetadataQuery` — used to ask Weaviate to return distance metadata alongside the hits, so we can show `distance=` in the screen-share.
- `from sentence_transformers import SentenceTransformer` — local embedding model wrapper.
- `from transformers import pipeline` — Hugging Face's high-level helper that wraps the `flan-t5-base` text-to-text generator.
- `EMBED_MODEL_NAME` / `GEN_MODEL_NAME` — names pulled into constants so we can swap models in one place if we ever want to.
- `WEAVIATE_HOST`, `WEAVIATE_PORT` — the demo deliberately runs on port **8091** (not 8080) so it doesn't conflict with anything the learner spins up later in the integration.
- `CLASS_NAME = "Post"` — same class name as the integration's `index_helpers.py`, so the schema code path is identical.
- `CORPUS_PATH = Path("pharma_corpus.jsonl")` — where the reshaped corpus gets written.
- `print("Loading embedder…")` — chatty progress so the instructor knows where the long pause is.
- `embedder = SentenceTransformer(EMBED_MODEL_NAME)` — first call downloads/caches the model, then later calls are instant.
- `print("Loading generator…")` — same reason.
- `generator = pipeline("text2text-generation", model=GEN_MODEL_NAME)` — loads `flan-t5-base` for generation. This is the single most expensive line in the notebook.
- `print("Models ready.")` — sanity beat for the screen-share: the instructor moves on only after this prints.

```python
import json
import os
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

from sentence_transformers import SentenceTransformer
from transformers import pipeline

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8091
CLASS_NAME = "Post"
CORPUS_PATH = Path("pharma_corpus.jsonl")

print("Loading embedder…")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
print("Loading generator…")
generator = pipeline("text2text-generation", model=GEN_MODEL_NAME)
print("Models ready.")
```

---

## Section 1 — Authoring the 50-drug mock corpus

The corpus is hand-written. Each entry has the same four DailyMed-style fields, plus a mandatory `notice`. The MOCK label is appended to every record in one pass at the bottom so we can't forget it on an individual record.

- `MOCK_NOTICE = "MOCK — for educational demonstration only, not medical advice"` — single canonical string so every record has identical wording.
- `docs = [...]` — a Python list of 50 dicts. Each dict has `id` (the `pharma:NNN` pattern that matches the integration's schema), `title` (drug name + form), and the four content fields. IDs are sequential to make it trivial to spot a missing entry.
- The 50 entries cover six rough therapeutic clusters: analgesics/NSAIDs, cardiovascular, GI/PPIs, allergy/respiratory, antibiotics/antifungals/antivirals, endocrine/diabetes, and psychiatry/pain — chosen so the demo can pull from different regions of the embedding space.
- `for d in docs: d["notice"] = MOCK_NOTICE` — adds the notice to every record in one line. Doing it once at the bottom (instead of inline per entry) means we can't accidentally publish a record without the disclaimer.
- `print(f"Authored {len(docs)} mock drug entries.")` — quick sanity check on screen.
- `print(f"Sample: {docs[0]['title']}")` — shows the instructor the first record's title, useful when teaching the dict structure.

```python
MOCK_NOTICE = "MOCK — for educational demonstration only, not medical advice"

docs = [
    {"id": "pharma:001", "title": "Acetaminophen — adult tablets",
     "indication": "Mild to moderate pain; fever reduction.",
     "dosage": "Adults: 500–1000 mg every 4–6 hours, maximum 4 g/day.",
     "contraindications": "Severe hepatic impairment; alcohol use disorder.",
     "warnings": "Risk of hepatotoxicity at doses above 4 g/day. Many combination cold/flu products contain acetaminophen — check labels."},
    # … entries 002 through 050 follow the identical shape …
    {"id": "pharma:050", "title": "Ondansetron — oral tablets",
     "indication": "Prevention of nausea and vomiting (chemotherapy, post-op).",
     "dosage": "Adults: 8 mg twice daily.",
     "contraindications": "Concomitant apomorphine; congenital long QT syndrome.",
     "warnings": "QT prolongation, especially with electrolyte imbalance. Serotonin syndrome possible."},
]

for d in docs:
    d["notice"] = MOCK_NOTICE

print(f"Authored {len(docs)} mock drug entries.")
print(f"Sample: {docs[0]['title']}")
```

> Full 50-entry list is in the notebook cell. They are abbreviated above for readability — the demo notebook contains all 50.

---

## Section 2 — Reshaping into the `Post` schema

The M8A integration's `index_helpers.py` expects a flat `Post` shape. We map each pharma record into that shape, joining the four content fields into one `text` blob (which is what we'll embed) plus a structured `answer_text` (which is what we'll show to the model and to the user).

- `with open(CORPUS_PATH, "w") as f:` — opens `pharma_corpus.jsonl` for writing. Each record becomes one JSON object on its own line (JSONL format).
- `for d in docs:` — iterate over the 50 source dicts.
- `combined = (...)` — a single string with the four fields separated by clearly labelled headers (`INDICATION:`, `DOSAGE:`, `CONTRAINDICATIONS:`, `WARNINGS:`). The labels matter — `flan-t5-base` uses them as anchors when synthesizing across fields.
- `post = {...}` — the target schema. Note `subset: "pharma"` (this is what the integration's filtering uses to slice the eval), and `question_text: d["indication"]` (the indication makes a reasonable proxy "question" for the doc, even though we'll mostly query against `text`).
- `license_version: "n/a — mock corpus"` and `source_url: "n/a (instructor-curated mock)"` — both make it explicit that this corpus has no real upstream source.
- `notice: d["notice"]` — propagates the MOCK label into the indexed record so anyone aggregating the data sees it.
- `f.write(json.dumps(post) + "\n")` — JSONL format means one JSON object per line, separated by `\n`.
- `print(f"Wrote {CORPUS_PATH.resolve()} ({CORPUS_PATH.stat().st_size} bytes).")` — confirms the file exists and has non-zero size before the next step tries to read it.

```python
with open(CORPUS_PATH, "w") as f:
    for d in docs:
        combined = (
            f"INDICATION: {d['indication']}\n"
            f"DOSAGE: {d['dosage']}\n"
            f"CONTRAINDICATIONS: {d['contraindications']}\n"
            f"WARNINGS: {d['warnings']}"
        )
        post = {
            "id": d["id"],
            "subset": "pharma",
            "title": d["title"],
            "question_text": d["indication"],
            "answer_text": combined,
            "text": f"{d['title']}\n\n{combined}",
            "license_version": "n/a — mock corpus",
            "source_url": "n/a (instructor-curated mock)",
            "notice": d["notice"],
        }
        f.write(json.dumps(post) + "\n")

print(f"Wrote {CORPUS_PATH.resolve()} ({CORPUS_PATH.stat().st_size} bytes).")
```

---

## Section 3 — Connecting to Weaviate

The docker container is started outside the notebook (see the markdown cell above). This step just makes sure the client is talking to the right port and the server is healthy.

- `client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=50051)` — v4 client. `connect_to_local` is a convenience helper that builds an HTTP + gRPC connection. The HTTP port is the demo-specific **8091**; gRPC stays at the default 50051.
- `assert client.is_ready(), "Weaviate is not ready on port 8091"` — fail loud and early if the container isn't reachable. Catching this in section 3 is cheap; finding out in section 5 (mid-ingest) is not.
- `print("Weaviate connected and ready.")` — confirmation beat.

```python
client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=50051)
assert client.is_ready(), "Weaviate is not ready on port 8091"
print("Weaviate connected and ready.")
```

---

## Section 4 — Creating the `Post` collection

Idempotent: if the collection already exists from a previous demo run, drop it and recreate. Easier to reason about than mutation.

- `if client.collections.exists(CLASS_NAME): client.collections.delete(CLASS_NAME)` — drop the old collection so we start clean.
- `client.collections.create(name=CLASS_NAME, ...)` — define the schema.
- `vectorizer_config=Configure.Vectorizer.none()` — **important.** Tells Weaviate not to look for a vectorizer module. We're computing vectors ourselves with `all-MiniLM-L6-v2` and passing them in directly, so the server doesn't need any ML modules enabled (matches the `ENABLE_MODULES=` empty value in the docker run command).
- `properties=[Property(name=..., data_type=DataType.TEXT), ...]` — declares the nine `TEXT` fields. All `TEXT` keeps the schema simple; we don't need numeric or boolean fields for this demo.
- Note: the post's `id` is stored under the property name `post_id`, not `id`, because Weaviate reserves `id` for its own internal UUID.

```python
if client.collections.exists(CLASS_NAME):
    client.collections.delete(CLASS_NAME)
    print(f"Dropped existing {CLASS_NAME} collection.")

client.collections.create(
    name=CLASS_NAME,
    vectorizer_config=Configure.Vectorizer.none(),
    properties=[
        Property(name="post_id", data_type=DataType.TEXT),
        Property(name="subset", data_type=DataType.TEXT),
        Property(name="title", data_type=DataType.TEXT),
        Property(name="question_text", data_type=DataType.TEXT),
        Property(name="answer_text", data_type=DataType.TEXT),
        Property(name="text", data_type=DataType.TEXT),
        Property(name="license_version", data_type=DataType.TEXT),
        Property(name="source_url", data_type=DataType.TEXT),
        Property(name="notice", data_type=DataType.TEXT),
    ],
)
print(f"Created {CLASS_NAME} collection.")
```

---

## Section 5 — Ingesting the corpus

Embed once, write once, batch into Weaviate. Should finish in ~10 seconds for 50 docs.

- `posts_collection = client.collections.get(CLASS_NAME)` — handle to the collection we just created.
- `with open(CORPUS_PATH) as f: records = [json.loads(line) for line in f]` — read the JSONL file back into a list of dicts. One pass through the file.
- `texts = [r["text"] for r in records]` — extract just the `text` field; that's what we'll embed.
- `vectors = embedder.encode(texts, show_progress_bar=True).tolist()` — single batched call to the embedder. Returns a NumPy array; `.tolist()` converts to a plain Python list of lists so the Weaviate client can serialise it.
- `with posts_collection.batch.dynamic() as batch:` — open a dynamic batch context. Dynamic mode lets the client auto-tune batch size for the running server.
- `for record, vec in zip(records, vectors):` — walk records and vectors in lockstep.
- `batch.add_object(properties={...}, vector=vec)` — push one object into the batch. The `properties` dict matches the schema we declared in section 4; `vector=vec` supplies our precomputed embedding.
- `print(f"Ingested {len(records)} documents.")` — final confirmation.

```python
posts_collection = client.collections.get(CLASS_NAME)

with open(CORPUS_PATH) as f:
    records = [json.loads(line) for line in f]

texts = [r["text"] for r in records]
vectors = embedder.encode(texts, show_progress_bar=True).tolist()

with posts_collection.batch.dynamic() as batch:
    for record, vec in zip(records, vectors):
        batch.add_object(
            properties={
                "post_id": record["id"],
                "subset": record["subset"],
                "title": record["title"],
                "question_text": record["question_text"],
                "answer_text": record["answer_text"],
                "text": record["text"],
                "license_version": record["license_version"],
                "source_url": record["source_url"],
                "notice": record["notice"],
            },
            vector=vec,
        )

print(f"Ingested {len(records)} documents.")
```

---

## Section 6 — Aggregate count check

The pre-demo checklist literally lists "aggregate query against the demo Post class shows count = 50" as a go/no-go criterion.

- `agg = posts_collection.aggregate.over_all(total_count=True)` — Weaviate's count aggregation. `total_count=True` is the flag that asks for the document count.
- `print(f"Documents in {CLASS_NAME}: {agg.total_count}")` — log the result.
- `assert agg.total_count == 50, ...` — hard-fail if we're not at 50. Cheaper to discover this here than in front of a class.
- `print("Pre-demo check: PASS")` — visible success beat for the screen-share.

```python
agg = posts_collection.aggregate.over_all(total_count=True)
print(f"Documents in {CLASS_NAME}: {agg.total_count}")
assert agg.total_count == 50, f"Expected 50, got {agg.total_count}"
print("Pre-demo check: PASS")
```

---

## Section 7 — The RAG pipeline

Three functions: `retrieve`, `rag_pipeline`, and a `groundedness_score` helper. Plus a `show` formatter for clean screen-share output.

- `STOPWORDS = {...}` — small English stopword set. We strip these before computing groundedness so the metric measures **content-token overlap**, not "the/of/and" overlap. Without this, every short answer would get an artificially high score.
- `PROMPT_TEMPLATE = (...)` — the canonical prompt for the demo. Three deliberate choices in the template:
  - First line: the mock-corpus disclaimer ("This is a mock pharmaceutical product information corpus for educational purposes only"). This both preserves methodology shape and stays visible on screen during the demo.
  - Second line: explicit refusal instruction — "If the context does not contain the answer, say 'I don't know.'" This is what makes the borderline-abstention beat work.
  - Last line: `Answer:` — gives `flan-t5-base` a clear continuation point.
- `def tokenize(text)` — lowercase, replace non-alphanumeric chars with spaces, split, drop stopwords. The expression `"".join(c.lower() if c.isalnum() else " " for c in text).split()` is a simple punctuation-stripper that avoids importing `re`.
- `def groundedness_score(answer, contexts)` — returns `|answer_tokens ∩ context_tokens| / |answer_tokens|`. Empty answer → 0.0 (avoids divide-by-zero). High score means the answer's words come from the retrieved context; low score means the model made stuff up.
- `def retrieve(question, k=5)`:
  - `q_vec = embedder.encode(question).tolist()` — embed the query with the same model used at indexing time. **This matters** — using a different model would produce mathematically incomparable vectors.
  - `response = posts_collection.query.near_vector(near_vector=q_vec, limit=k, return_metadata=MetadataQuery(distance=True))` — k-NN search by vector. `return_metadata=MetadataQuery(distance=True)` asks Weaviate to include the cosine distance so we can show it.
  - List comprehension at the end repackages each hit into a clean dict.
- `def rag_pipeline(question, k=5)`:
  - `hits = retrieve(question, k=k)` — top-5 by default. Five is the lab default and a reasonable balance: enough to allow synthesis, not so many that the prompt blows past the model's context.
  - `context = "\n\n---\n\n".join(h["text"] for h in hits)` — concatenate hits with a visible separator. The separator helps `flan-t5-base` not blur across documents.
  - `prompt = PROMPT_TEMPLATE.format(context=context, question=question)` — build the final prompt.
  - `raw = generator(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]` — deterministic decoding (`do_sample=False`) so the demo is reproducible. `max_new_tokens=128` is plenty for a paragraph-length answer.
  - Returns a dict with `question`, `answer`, `groundedness_score`, and the top-k retrieved titles + distances — everything the on-screen output needs.
- `def show(result)` — pretty-prints the four pieces for the screen-share. The instructor calls `show(rag_pipeline(...))` for each demo query.

```python
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "that", "this", "these",
    "those", "it", "its", "if", "then", "than", "so", "do", "does", "did", "not", "no",
    "i", "you", "he", "she", "we", "they", "have", "has", "had", "can", "could", "should",
    "would", "may", "might", "will", "shall",
}

PROMPT_TEMPLATE = (
    "Note: This is a mock pharmaceutical product information corpus for educational purposes only.\n"
    "Answer the question using only the context below. If the context does not contain the answer, "
    "say 'I don't know.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

def tokenize(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t not in STOPWORDS}

def groundedness_score(answer: str, contexts: list[str]) -> float:
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = tokenize(" ".join(contexts))
    return len(answer_tokens & context_tokens) / len(answer_tokens)

def retrieve(question: str, k: int = 5) -> list[dict]:
    q_vec = embedder.encode(question).tolist()
    response = posts_collection.query.near_vector(
        near_vector=q_vec,
        limit=k,
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        {"title": obj.properties["title"], "text": obj.properties["text"], "distance": obj.metadata.distance}
        for obj in response.objects
    ]

def rag_pipeline(question: str, k: int = 5) -> dict:
    hits = retrieve(question, k=k)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    raw = generator(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
    contexts = [h["text"] for h in hits]
    return {
        "question": question,
        "answer": raw.strip(),
        "groundedness_score": round(groundedness_score(raw, contexts), 3),
        "retrieved": [{"title": h["title"], "distance": round(h["distance"], 3)} for h in hits],
    }

def show(result: dict) -> None:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"groundedness_score: {result['groundedness_score']}")
    print("Retrieved (top-k):")
    for r in result["retrieved"]:
        print(f"  - {r['title']}  (distance={r['distance']})")
    print()

print("RAG pipeline defined.")
```

---

## Section 8 — Single-fact extractive queries

Two queries with a single-document answer. Expect a tight, correct answer and a **high groundedness score** (typically ≥ 0.7).

- `show(rag_pipeline("What is the maximum daily dose of acetaminophen for adults?"))` — the answer lives in `pharma:001`'s `dosage` field ("maximum 4 g/day"). The retriever should pull `pharma:001` as the top hit; the generator should extract "4 g/day" almost verbatim.
- `show(rag_pipeline("What is the typical metformin starting dose for type 2 diabetes?"))` — answer lives in `pharma:006` ("start 500 mg once or twice daily with meals"). Same pattern: top hit is `pharma:006`, answer is extractive.

```python
show(rag_pipeline("What is the maximum daily dose of acetaminophen for adults?"))
```

```python
show(rag_pipeline("What is the typical metformin starting dose for type 2 diabetes?"))
```

---

## Section 9 — Synthesis query

One query whose answer requires combining the `contraindications` and `warnings` fields of `pharma:005` (lisinopril). Expect the model to produce a multi-part answer drawn from both fields, with groundedness still high.

- `show(rag_pipeline("What should a patient avoid while taking lisinopril?"))` — the answer is a mix of:
  - From `contraindications`: pregnancy.
  - From `warnings`: potassium supplements, potassium-sparing diuretics.
  - The top hit will be `pharma:005`; supporting hits (e.g., `pharma:010` losartan, also an ARB-class agent) may appear too, which is fine.

```python
show(rag_pipeline("What should a patient avoid while taking lisinopril?"))
```

---

## Section 10 — Borderline / abstention queries

The two most important queries in the demo. Neither answer is in the corpus.

- `show(rag_pipeline("What is the recommended dose of acetaminophen for a 6-month-old?"))` — the acetaminophen doc only covers adults. The model should either say "I don't know" or hedge ("I don't know — the corpus only describes adult dosing"). If it instead invents a dose, the **groundedness score will be low** — that's the live teaching moment: *"the model hallucinated; look at the score."*
- `show(rag_pipeline("Can ibuprofen be combined with warfarin safely?"))` — neither doc mentions the other; the interaction isn't anywhere in the corpus. Same expected behavior: abstain or hedge.

Recovery plan if the model produces a confident answer anyway: that **is** the lesson. Point at the score, narrate it, move on. Don't apologize for the model.

```python
show(rag_pipeline("What is the recommended dose of acetaminophen for a 6-month-old?"))
```

```python
show(rag_pipeline("Can ibuprofen be combined with warfarin safely?"))
```

---

## Section 11 — Pivot and cleanup

- `client.close()` — closes both the HTTP and gRPC connections cleanly. The v4 client holds open connections, and leaving them open in a Jupyter kernel will eventually surface as warnings.
- `print("Weaviate client closed. Demo complete.")` — visible end-state for the screen-share.

While `client.close()` is running, the instructor reads out the pivot beat: same RAG machinery, different corpus, learner's `answer_keyword_recall_main` will land at 0.00–0.10 on the canonical Stack Exchange config — and that's not a bug, that's the canonical-baseline framing.

```python
client.close()
print("Weaviate client closed. Demo complete.")
```

---

## Appendix — Pre-demo checklist (Wednesday morning, ~10 min before class)

- [ ] Weaviate container running on port 8091. `curl -s localhost:8091/v1/.well-known/ready` returns 200.
- [ ] Notebook section 6 prints `Documents in Post: 50` and `Pre-demo check: PASS`.
- [ ] All five demo queries pre-run; outputs noted.
- [ ] `flan-t5-base` and `all-MiniLM-L6-v2` cached locally; first call is instant.
- [ ] MOCK disclaimer visible at top of screen-share (the notebook header has it).
- [ ] Integration guide URL on a sticky note for the closing pivot.

## Appendix — Recovery scenarios

| Symptom | Action |
| --- | --- |
| `flan-t5-base` inference is 30+ s/call | Generator is loading per call. Restart the kernel and rerun section 0 — the module-level load fixes this. |
| Borderline query returns a confident answer | This is the lesson. Point at the low `groundedness_score`; narrate "the model hallucinated; that's why we measure groundedness." |
| Weaviate dies mid-demo | Fall back to the pre-recorded screencast (same fallback as the integration lab). |
| Spot a wrong fact in your own mock corpus | Disclose on screen: "This is a mock entry I wrote for demo purposes — don't take it as medical guidance." Better than pretending the corpus is canonical. |
| Learner asks for the corpus to use elsewhere | Decline. It's instructor-built mock content. Do not commit `pharma_corpus.jsonl` to any M8 repo. |
