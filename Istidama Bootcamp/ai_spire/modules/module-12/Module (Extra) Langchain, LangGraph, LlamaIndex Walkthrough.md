# Module (Extra) Langchain, LangGraph, LlamaIndex Lab — Code Walkthrough

This document walks every line of [self_correcting_doc_assistant.ipynb](self_correcting_doc_assistant.ipynb). Each section starts with a line-by-line explanation, then shows the full code block at the end so you can copy-paste it cleanly.

The notebook builds a **corrective-RAG** (Retrieval-Augmented Generation) assistant over a small fictional employee handbook, entirely with local, free tooling (Ollama + a local HuggingFace embedder — no API keys). It deliberately uses three libraries for three different jobs: **LlamaIndex** loads/chunks/embeds/retrieves documents, **LangChain** wraps prompts and LLM calls into typed, structured chains, and **LangGraph** wires those chains into a stateful graph that grades its own retrieval, retries with a rewritten query when the context is weak, and falls back to an honest "I don't know" instead of hallucinating. Read alongside the notebook.

---

## 0. Setup — install, configure, verify Ollama

### Why this section exists
Before any retrieval or generation can happen, the environment needs the right packages installed, one central place for every tunable constant, and a fast, reliable check that the local Ollama server is actually running with the right model pulled. Cells 2–4 correspond to this section (cells 0–1 are the notebook's markdown intro and setup instructions).

### 0a. Install dependencies (cell 2)

#### Line-by-line
- `import subprocess, sys` — `subprocess` lets Python launch and manage external processes (here, `pip`); `sys` gives access to `sys.executable`, the absolute path to the exact Python interpreter currently running the notebook kernel.
- `PACKAGES = [...]` — a single list of every third-party dependency the notebook needs, each annotated with an inline comment explaining its role:
  - `"llama-index-core"` — the core LlamaIndex library: document loading, chunking (node parsing), indexing, and retrieval abstractions.
  - `"llama-index-vector-stores-chroma"` — the adapter package that lets a LlamaIndex `VectorStoreIndex` write to and read from a Chroma collection instead of LlamaIndex's default in-memory store.
  - `"llama-index-embeddings-huggingface"` — the adapter that lets LlamaIndex call a local HuggingFace `sentence-transformers`-style model for embeddings instead of an API-based one (e.g., OpenAI).
  - `"langchain"` and `"langchain-core"` — the chain-composition layer: prompt templates, the `|` (pipe) operator for chaining runnables, and output parsers. `langchain-core` holds the shared abstractions that every other LangChain integration package (including `langchain-ollama`) depends on.
  - `"langchain-ollama"` — the LangChain integration that wraps a local Ollama model behind LangChain's standard `ChatModel` interface (`ChatOllama`).
  - `"langgraph"` — the state-machine/graph library used in Stage 3 to build the retrieve → grade → rewrite/generate/fallback control flow.
  - `"chromadb"` — the actual vector database engine. LlamaIndex's Chroma adapter is just a thin wrapper around this package.
  - `"gradio"` — used only in the very last cell to build the interactive web UI.
- `subprocess.check_call([sys.executable, "-m", "pip", "install", *PACKAGES, "-q"])` — runs `pip install` as a subprocess rather than shelling out with `!pip install`. Using `sys.executable -m pip` (rather than a bare `pip` command) guarantees the packages install into the *exact* interpreter running this kernel, avoiding the common bug where `pip` on `PATH` points to a different Python than the notebook. `*PACKAGES` unpacks the list into individual command-line arguments (equivalent to typing each package name separately). `"-q"` requests quiet output so the cell doesn't dump hundreds of lines of installer log. `check_call` (as opposed to `subprocess.run`) raises `CalledProcessError` immediately if `pip` exits non-zero, so a failed install stops the notebook here instead of silently continuing with missing packages.
- `print("✅ Installation complete — restart the kernel, then continue.")` — a reminder that newly installed packages are not visible to an already-running kernel; the user must restart before the next cell can `import` them.

#### Full code block
```python
# ── Install all dependencies. Run once, then RESTART the kernel. ─────────────
import subprocess, sys                       # stdlib: run shell commands; sys.executable = this Python

PACKAGES = [
    "llama-index-core",                      # LlamaIndex core — load, chunk, embed, index, retrieve
    "llama-index-vector-stores-chroma",      # LlamaIndex adapter: use Chroma as the vector store
    "llama-index-embeddings-huggingface",    # LlamaIndex adapter: run HuggingFace models locally
    "langchain",                             # LangChain — compose prompts, LLM calls, output parsers
    "langchain-core",                        # shared abstractions required by all LangChain packages
    "langchain-ollama",                      # LangChain adapter: connects chains to a local Ollama server
    "langgraph",                             # LangGraph — state-machine graphs for agentic control flow
    "chromadb",                              # Chroma — fast local vector database with on-disk persistence
    "gradio",                                # Gradio — builds the interactive UI in the final cell
]

subprocess.check_call(                       # run pip; raises CalledProcessError if any package fails
    [sys.executable,                         # use the exact Python interpreter running this notebook
     "-m", "pip", "install",                # invoke pip's install sub-command
     *PACKAGES,                              # unpack the list into individual positional arguments
     "-q"]                                   # quiet: suppress verbose installer output
)
print("✅ Installation complete — restart the kernel, then continue.")
```

### 0b. Imports & configuration (cell 3)

#### Line-by-line
- `import os` — imported for general OS interop; available to later cells even though this cell itself doesn't call it directly.
- `import textwrap` — used later (Stage 1 verification) to wrap long retrieved chunk text to a fixed column width for readable console output.
- `from pathlib import Path` — object-oriented filesystem paths; used for `DATA_DIR` and `STORAGE_DIR` so `.mkdir()`, `.glob()`, and `/`-joining work without manual string concatenation.
- `from operator import add` — the built-in two-argument `add(a, b)` function, reused here as a **LangGraph reducer**: when a graph node returns a value for a field configured with this reducer, LangGraph calls `add(old_value, new_value)` to merge it instead of overwriting it.
- `from typing import TypedDict, Annotated, List, Optional, Literal` — type-hinting tools used to define `GraphState` (Stage 3) precisely: `TypedDict` gives dict-shaped state a declared schema, `Annotated` attaches the reducer metadata to a field, `Literal` constrains a field to an exact enumerated set of string values (used in the Pydantic models in Stage 2).
- `DATA_DIR = Path("data")` — the folder (relative to the notebook's working directory) that holds the five handbook markdown files this notebook writes and later ingests.
- `STORAGE_DIR = Path("/tmp/chroma_handbook")` — deliberately placed under `/tmp` rather than next to the notebook. The comment explains why: this module's folder name contains parentheses and commas (`Module (Extra) Langchain, Langgraph, LLamaindex`), and Chroma's Rust-backed SQLite storage layer breaks on special characters in its path. `/tmp` sidesteps that entirely.
- `DATA_DIR.mkdir(exist_ok=True)` — creates `data/` if it doesn't already exist; `exist_ok=True` means calling this again on a later re-run is a no-op instead of raising `FileExistsError`.
- `STORAGE_DIR.mkdir(parents=True, exist_ok=True)` — same idea, but `parents=True` is also needed here because `/tmp/chroma_handbook` may require creating intermediate directories (in this case just the leaf directory under `/tmp`, but the flag is defensive regardless).
- `LLM_MODEL = "phi3:latest"` — the Ollama model tag used for the two generation-facing chains: answering the user's question and rewriting a failed search query.
- `GRADER_MODEL = "phi3:latest"` — the model used for the relevance-grading LLM call in Stage 3. It's set to the same tag as `LLM_MODEL` here (the comment notes "same = fine, no cost diff" since everything is local and free), but keeping it as a separate constant means a learner could later swap in a smaller/faster model for grading without touching generation.
- `EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"` — the HuggingFace model ID for the sentence-embedding model. BGE-small is a compact (~130 MB), CPU-friendly embedding model well suited to a small handbook corpus.
- `CHROMA_COLLECTION = "handbook"` — the name of the Chroma collection that stores the embedded document chunks; reused everywhere a collection handle is opened (`ingest()`, `load_index()`).
- `TOP_K = 4` — the default number of chunks `retrieve()` returns per query. Larger values give the LLM more context but risk diluting it with less-relevant text.
- `MAX_RETRIES = 2` — the ceiling on how many times the LangGraph control loop will rewrite the search query and retry retrieval before giving up and routing to the fallback node. This is the parameter that bounds the self-correction loop so it cannot run forever.
- `CHUNK_SIZE = 512` — the maximum number of tokens per chunk when `SentenceSplitter` breaks documents apart in Stage 1.
- `CHUNK_OVERLAP = 64` — the number of tokens shared between two adjacent chunks, so a sentence or fact that happens to fall on a chunk boundary is still fully present in at least one chunk.
- `print("✅ Config loaded.")` and the following `print(f"...")` — visible confirmation that every constant loaded without error, plus a quick summary of which LLM and embedding model are active — useful when a learner has changed `LLM_MODEL` and wants to confirm the change took effect.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  IMPORTS & CONFIGURATION
#  All constants live here — one place to change everything.
# ─────────────────────────────────────────────────────────────
import os                                    # stdlib: interact with the operating system
import textwrap                              # stdlib: reflow long strings to a max line width
from pathlib import Path                     # Path objects for OS-independent file-system operations
from operator import add                     # built-in add(a,b) — used as a LangGraph list reducer
from typing import TypedDict, Annotated, List, Optional, Literal  # type hints used throughout

DATA_DIR    = Path("data")                   # folder that holds our handbook markdown files
STORAGE_DIR = Path("/tmp/chroma_handbook")  # /tmp avoids special chars (parentheses, commas) in notebook path that break Chroma's Rust backend
DATA_DIR.mkdir(exist_ok=True)               # create data/ now; harmless if it already exists
STORAGE_DIR.mkdir(parents=True, exist_ok=True)  # parents=True needed since /tmp/chroma_handbook may not exist yet

# ── LLM (Ollama — local, free, no API key) ─────────────────────────────────
LLM_MODEL    = "phi3:latest"                # Ollama model used for generation and query rewriting
GRADER_MODEL = "phi3:latest"               # Ollama model used for relevance grading (same = fine, no cost diff)

# ── Embeddings (local HuggingFace — also free) ────────────────────────────
EMBED_MODEL_NAME  = "BAAI/bge-small-en-v1.5"  # downloads ~130 MB once, then runs fully offline
CHROMA_COLLECTION = "handbook"              # name of the Chroma collection that stores our vectors

# ── Retrieval & graph tuning ───────────────────────────────────────────────
TOP_K         = 4                           # how many chunks to return per retrieval query
MAX_RETRIES   = 2                           # max query rewrites before falling back to "I don't know"
CHUNK_SIZE    = 512                         # maximum tokens per text chunk
CHUNK_OVERLAP = 64                          # tokens shared between neighbouring chunks for context continuity

print("✅ Config loaded.")
print(f"   LLM: {LLM_MODEL} (via Ollama)  |  Embeddings: {EMBED_MODEL_NAME} (local)")
```

### 0c. Ollama connection check (cell 4)

#### Line-by-line
- `import urllib.request, json` — both are Python standard-library modules, deliberately chosen over a third-party HTTP client so this health check has zero extra dependencies. `urllib.request` opens raw HTTP connections; `json` parses the response body.
- `def check_ollama(model: str = LLM_MODEL) -> bool:` — defaults to whatever `LLM_MODEL` currently is, so calling `check_ollama()` with no arguments checks the model configured in the previous cell. Returning `bool` lets calling code branch on success/failure later if needed.
- `with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:` — hits Ollama's REST API directly (`/api/tags` lists every model currently pulled) rather than shelling out to the `ollama` CLI. The comment in the cell header explains why: an HTTP call returns in well under a second, whereas spawning the CLI as a subprocess can hang or time out unpredictably on slower systems. `timeout=10` bounds how long Python will wait for a response before raising, so a genuinely unreachable server fails fast instead of hanging the cell.
- `data = json.loads(response.read())` — `response.read()` returns the raw HTTP body as bytes; `json.loads` parses it into a Python dict, expected to have a `"models"` key holding a list of installed-model metadata.
- `except OSError:` — catches both `ConnectionRefusedError` (server not listening) and socket timeouts, since both are subclasses of `OSError` in Python 3. On failure it prints instructions to start the server and returns `False` immediately — the function never reaches the model-matching logic if Ollama itself isn't up.
- `available = [m["name"] for m in data.get("models", [])]` — extracts just the `"name"` field (a string like `"phi3:latest"`) from each installed model's metadata dict. `.get("models", [])` defensively defaults to an empty list if the key is somehow missing, rather than raising `KeyError`.
- `base_name = model.split(":")[0]` — strips the tag suffix from the configured model string, e.g. `"phi3:latest"` → `"phi3"`. This lets the function match on the model *family* regardless of which tag (`:latest`, `:3b`, etc.) is actually installed.
- `match = next((n for n in available if n.split(":")[0] == base_name), None)` — a generator expression combined with `next(..., None)` finds the first installed model whose base name matches, or `None` if there's no match at all. This is the idiomatic Python pattern for "find first match or a sensible default" without writing an explicit loop-and-break.
- `if match:` — if any tag of the right model family is installed, print success. `if match != model:` further checks whether the *exact* configured tag (e.g. `"phi3:latest"`) differs from what's actually installed, and if so warns the user that Ollama will silently use the installed tag instead — informative rather than a hard failure, since Ollama's own resolution logic handles this gracefully.
- `else:` branch — no model in that family is installed at all; prints the list of what *is* available and the exact `ollama pull` command to fix it, then returns `False`.
- `check_ollama()` — the module-level call at the bottom runs the check immediately as the cell executes, so the pass/fail result is visible in the cell's output without the user having to call the function themselves.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  OLLAMA CONNECTION CHECK
#  Confirms Ollama is running and the model is pulled.
#  Uses the Ollama HTTP API (fast, ~100 ms) instead of the
#  subprocess CLI so this cell never times out on slow systems.
#  Fix any issues here before proceeding to Stage 2.
# ─────────────────────────────────────────────────────────────
import urllib.request, json                 # stdlib: HTTP request and JSON parsing — no extra deps

def check_ollama(model: str = LLM_MODEL) -> bool:  # default to the configured LLM_MODEL
    try:
        with urllib.request.urlopen(        # open the Ollama REST endpoint that lists pulled models
            "http://localhost:11434/api/tags",
            timeout=10                      # 10 s is plenty; the HTTP response arrives in < 1 s
        ) as response:
            data = json.loads(response.read())   # parse the JSON body into a Python dict
    except OSError:                         # covers ConnectionRefusedError + socket timeout
        print("❌ Ollama is not responding on localhost:11434.")
        print("   Start the server with:  ollama serve")
        return False

    available = [m["name"] for m in data.get("models", [])]  # list of "name:tag" strings
    base_name = model.split(":")[0]                           # "llama3.2:3b" → "llama3.2"
    match = next(                                             # find first name whose base matches
        (n for n in available if n.split(":")[0] == base_name), None
    )

    if match:
        print(f"✅ Ollama running  |  model '{match}' is available")
        if match != model:                  # warn if the tag differs (e.g. :latest vs :3b)
            print(f"   ℹ️  Config uses '{model}' but installed tag is '{match}'.")
            print(f"      Ollama will use '{match}' automatically.")
        return True
    else:
        print(f"⚠️  Ollama is running but no '{base_name}' model is installed.")
        print(f"   Available: {available}")
        print(f"   Pull it with:  ollama pull {model}")
        return False

check_ollama()                              # run the check immediately when this cell executes
```

---

## 1. Sample data — the fictional employee handbook (cell 5)

### Why this section exists
The whole lab needs a small, self-contained corpus with a mix of answerable and unanswerable questions baked in on purpose, so learners can see both the "happy path" (context found, answer generated) and the "self-correction path" (context not found, graceful fallback) without needing real company data.

### Line-by-line
- `PTO_POLICY_MD = """ ... """` — a triple-quoted markdown string that becomes `pto_policy.md`. It states **20 PTO days per year** and that new employees get the same entitlement immediately — this is the fact the "easy question" (`"How many PTO days do new employees get?"`) is designed to retrieve later in Stage 3.
- `REMOTE_WORK_MD = """ ... """` — becomes `remote_work_policy.md`; documents the hybrid-first policy (3 remote days/week) and a **$75/month home-office stipend** — used by one of the Gradio example questions.
- `EXPENSE_POLICY_MD = """ ... """` — becomes `expense_policy.md`; documents reimbursement categories, the $100-per-person meal cap, and the 30-day submission window.
- `CODE_OF_CONDUCT_MD = """ ... """` — becomes `code_of_conduct.md`; documents prohibited conduct and the HR/ethics-hotline reporting channel.
- `ONBOARDING_MD = """ ... """` — becomes `onboarding.md`; documents the first-two-weeks timeline and key contact emails.
- Each string constant ends with a trailing `# comment` after the closing `"""` — Python allows a comment on the same physical line as the end of a multi-line string literal; these comments are pure documentation for whoever reads the notebook source, summarizing what fact each document is "for" in this teaching exercise.
- `docs = { "pto_policy.md": PTO_POLICY_MD, ... }` — a dict mapping the target filename to its content string, built explicitly (not derived from variable names) so the mapping is unambiguous and easy to scan.
- `for filename, content in docs.items():` — iterates the five (filename, text) pairs.
- `(DATA_DIR / filename).write_text(content)` — `Path.__truediv__` (the `/` operator) joins `DATA_DIR` and `filename` into a full path object (e.g. `data/pto_policy.md`); `.write_text()` creates or overwrites that file with `content` in one call, with no manual `open()`/`close()` needed.
- `print(f"✅ Created {len(docs)} handbook documents in {DATA_DIR}/")` — confirms all five files were written.
- `for f in sorted(DATA_DIR.iterdir()):` — lists every entry in `data/` alphabetically (`sorted()` on `Path` objects sorts by their string form) so the printed order is stable and predictable across runs, even though dict iteration order and filesystem listing order aren't guaranteed to match.
- `print(f"   {f.name}")` — prints just the bare filename (`f.name`), not the full path, keeping the confirmation list compact.

### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  SAMPLE DATA — Fictional Employee Handbook (5 markdown files)
#
#  Answerable:    "How many PTO days do new employees get?"
#  Unanswerable:  "What is the policy on cryptocurrency payments?"
# ─────────────────────────────────────────────────────────────

PTO_POLICY_MD = """
# PTO Policy

## Paid Time Off for Full-Time Employees

All full-time employees at Meridian Software receive **20 PTO days per calendar year**
starting from their first day of employment. New employees in their first 90 days receive
the same entitlement and may take PTO immediately with manager approval.

PTO days do not roll over into the following year. Any unused days are forfeited on
December 31. Employees who resign or are terminated are paid out for unused PTO days
accrued in the current calendar year.

### Requesting Time Off
Submit a PTO request in the HR portal at least two weeks in advance for planned absences.
For urgent or unplanned absences, notify your manager as soon as possible.
"""                                         # paid time off — contains the "20 PTO days" answer fact

REMOTE_WORK_MD = """
# Remote Work Policy

## Hybrid-First Work Environment

Meridian Software operates on a **hybrid-first** model. Employees may work remotely up to
**3 days per week**. The remaining 2 days should be worked from the office or an approved
co-working space.

Fully remote arrangements require written approval from the department head and HR,
and are reviewed annually.

### Equipment and Stipend
The company provides a laptop and standard peripherals. A monthly **home-office stipend
of $75** covers internet and ergonomic expenses. Submit receipts through Expensify within
30 days of purchase.
"""                                         # hybrid work rules and the $75/month home-office stipend

EXPENSE_POLICY_MD = """
# Expense Policy

## Reimbursable Business Expenses

Meridian Software reimburses reasonable, necessary business expenses with receipts.
Common reimbursable categories:
- Client meals and entertainment (up to **$100 per person**)
- Conference registration and travel
- Professional development books and courses (up to **$500 per year**)
- Software subscriptions pre-approved by your manager

### Submitting Expenses
Submit within **30 days** of purchase via the Expensify portal.
Expenses over **$500** require manager pre-approval before purchase.
Receipts are required for all expenses over $25.
"""                                         # expense limits and the 30-day submission rule

CODE_OF_CONDUCT_MD = """
# Code of Conduct

## Our Commitment

Meridian Software is committed to a respectful, inclusive, harassment-free workplace.
All employees, contractors, and visitors are expected to treat one another with dignity.

### Prohibited Conduct
- Harassment or discrimination based on race, gender, age, religion, disability,
  or any other protected characteristic
- Bullying, intimidation, or retaliation against anyone who raises a concern
- Sharing confidential company or client information outside authorized channels

### Reporting
Report violations to HR at hr@meridian.example or the anonymous ethics hotline:
1-800-555-ETHICS. All reports are treated confidentially.
"""                                         # conduct rules and how to report violations

ONBOARDING_MD = """
# Onboarding Guide

## Your First Two Weeks at Meridian

**Day 1:** IT setup, security training, and a tour with your onboarding buddy — a peer
assigned to help you get oriented during your first month.

**Week 1:** Meet your team, attend the Friday All Hands, and complete mandatory compliance
training (approximately 2 hours).

**Week 2:** Shadow a senior team member, complete your 30-day goals document with your
manager, and schedule your first 1-on-1.

### Key Contacts
- IT helpdesk: it@meridian.example
- HR onboarding: onboarding@meridian.example
- Your People Operations partner: listed in your offer letter
"""                                         # first-two-weeks timeline and key contact emails

docs = {                                    # maps output filename → markdown content
    "pto_policy.md":          PTO_POLICY_MD,      # PTO document
    "remote_work_policy.md":  REMOTE_WORK_MD,     # remote work document
    "expense_policy.md":      EXPENSE_POLICY_MD,  # expense document
    "code_of_conduct.md":     CODE_OF_CONDUCT_MD, # conduct document
    "onboarding.md":          ONBOARDING_MD,      # onboarding document
}

for filename, content in docs.items():      # iterate over each (filename, text) pair
    (DATA_DIR / filename).write_text(content)  # write the markdown string to disk inside data/

print(f"✅ Created {len(docs)} handbook documents in {DATA_DIR}/")
for f in sorted(DATA_DIR.iterdir()):        # list files in alphabetical order for a quick sanity-check
    print(f"   {f.name}")                   # print just the filename, not the full path
```

Note: at runtime the `data/` folder in this module actually also contains fifteen bootcamp curriculum markdown files (`01_dev_environment_and_collaboration.md` through `15_monitoring_evaluation_and_automated_quality_benchmarks.md`), which is why the Gradio UI's example questions in Section 5 include both handbook questions ("How many PTO days...") and curriculum questions ("What are the two phases of a RAG pipeline?"). This cell only *writes* the five handbook files — the curriculum files pre-exist in the folder and get picked up automatically by `SimpleDirectoryReader` in Stage 1 since it loads every file in `DATA_DIR`.

---

## 2. Stage 1 — LlamaIndex: the data layer (cells 7–12)

### Why this section exists
LlamaIndex owns the entire data pipeline: reading files off disk, splitting them into overlapping chunks, embedding those chunks into vectors with a local model, and storing/retrieving them through Chroma. The notebook's own markdown (cell 6) frames the habit to build here explicitly: **always eyeball retrieval before adding an LLM** — most bad RAG apps are actually bad retrieval apps, and cells 9–12 exist specifically to make retrieval quality visible before Stage 2 introduces any generation.

### 2a. `ingest()` and `load_index()` (cell 7)

#### Line-by-line
- `from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings` — four core LlamaIndex classes: `VectorStoreIndex` is the searchable index abstraction built on top of a vector store; `SimpleDirectoryReader` loads every file in a folder into LlamaIndex `Document` objects; `StorageContext` bundles together the vector store (and other storage backends) that an index is built against; `Settings` is LlamaIndex's global configuration singleton — setting `Settings.embed_model` once affects every subsequent embedding call in the process.
- `from llama_index.core.node_parser import SentenceSplitter` — the chunker. It splits documents into "Nodes" (LlamaIndex's term for chunks) preferentially at sentence boundaries rather than at hard character/token cutoffs, so chunks don't end mid-sentence when avoidable.
- `from llama_index.vector_stores.chroma import ChromaVectorStore` — the adapter class that lets a plain `chromadb` collection be used wherever LlamaIndex expects a `VectorStore`.
- `from llama_index.embeddings.huggingface import HuggingFaceEmbedding` — wraps a local `sentence-transformers`-compatible HuggingFace model behind LlamaIndex's embedding interface.
- `import chromadb` — the underlying vector database client library, used directly (not just through the LlamaIndex adapter) to create the client and manage the collection lifecycle.
- `_chroma_client = None` — a module-level variable, initialized once and shared by both `ingest()` and `load_index()` via the `global` keyword. The leading underscore is the Python convention marking it as an internal implementation detail, not part of the public API of this cell.
- `def ingest() -> VectorStoreIndex:` — the -> annotation documents that this function returns a built, queryable index.
- `global _chroma_client` — required because the function *reassigns* `_chroma_client` inside its body (`if _chroma_client is None: _chroma_client = chromadb.EphemeralClient()`); without `global`, that assignment would create a new local variable instead of mutating the module-level one.
- `Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)` — sets LlamaIndex's global embedding model for this process. Every subsequent chunk/query embedding call anywhere in the notebook uses this model implicitly, without needing to be passed explicitly to each call.
- `Settings.llm = None` — explicitly disables LlamaIndex's own LLM integration. This is important: LlamaIndex ships with defaults that will try to reach OpenAI's API if left unconfigured, which would break the "no API keys, fully local" premise of the lab. All generation in this notebook goes through LangChain's `ChatOllama` instead.
- `documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()` — reads every file in `data/` (both the five handbook markdown files and the pre-existing curriculum markdown files) and returns a list of LlamaIndex `Document` objects, one per file, each holding the raw file text plus metadata like the source filename.
- `splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)` — instantiates the chunker with the tunables from the config cell (512 tokens per chunk, 64 tokens of overlap).
- `if _chroma_client is None: _chroma_client = chromadb.EphemeralClient()` — the cell's header comment explains the reasoning in detail: `chromadb`'s 1.x Rust backend shares SQLite state across `EphemeralClient` instances in a way that, on this environment, corrupts the schema if the client object is replaced (e.g., by calling `ingest()` a second time and creating a fresh client each time) — producing a `"no such table: collections"` error. The fix is to create the in-memory client exactly once per kernel session and reuse it. `EphemeralClient()` (as opposed to `PersistentClient()`) keeps everything in RAM with no on-disk SQLite file at all — this also happens to route around a separate schema bug that `PersistentClient` hits in this environment (noted in the cell comment).
- `try: _chroma_client.delete_collection(CHROMA_COLLECTION) except Exception: pass` — even though the client itself is reused, the *collection* inside it is deleted and recreated on every `ingest()` call so re-running ingestion always starts from a clean slate (no duplicate or stale vectors from a previous run). The `try/except` swallows the error Chroma raises if the collection doesn't exist yet (e.g., on the very first call) — deletion failing for that reason is expected and harmless.
- `chroma_collection = _chroma_client.get_or_create_collection(CHROMA_COLLECTION)` — creates a fresh, empty collection named `"handbook"` (since it was just deleted, "get or create" always takes the "create" branch here in practice, but using the combined call is defensive against any collection state we didn't anticipate).
- `vector_store = ChromaVectorStore(chroma_collection=chroma_collection)` — wraps the raw Chroma collection handle in the LlamaIndex adapter so it satisfies LlamaIndex's `VectorStore` interface.
- `storage_context = StorageContext.from_defaults(vector_store=vector_store)` — bundles the vector store into a `StorageContext`, LlamaIndex's container for all the storage backends an index needs (here, just the vector store; the doc store and index store fall back to in-memory defaults).
- `index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, transformations=[splitter], show_progress=True)` — the single call that does the actual work: chunks every document with `splitter`, embeds each chunk using `Settings.embed_model`, and writes the resulting vectors into the Chroma collection via `storage_context`. `transformations=[splitter]` is how you plug a custom chunker into the pipeline — without it, LlamaIndex would use its own default splitting logic. `show_progress=True` renders a progress bar during embedding, useful feedback since embedding all documents can take some seconds.
- `return index` — hands back the fully built, queryable `VectorStoreIndex`.
- `def load_index() -> VectorStoreIndex:` — a cheaper counterpart to `ingest()`: it reconnects to the *existing* in-memory collection without re-reading files or re-embedding anything.
- `if _chroma_client is None: raise RuntimeError(...)` — guards against calling `load_index()` before `ingest()` has ever run in this kernel session. Because `EphemeralClient` keeps data purely in RAM, there is nothing to "load" from disk after a kernel restart — the error message says so explicitly.
- The rest of `load_index()` mirrors the schema/embedding setup in `ingest()` (re-set `Settings.embed_model` and `Settings.llm`, get the same collection, wrap it in the same adapter classes) but calls `VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)` instead of `.from_documents(...)` — this reconnects to a vector store that already has embedded data in it, rather than embedding anything new.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 1 — INGEST   (mirrors src/ingest.py)
#
#  LlamaIndex pipeline:
#    SimpleDirectoryReader  → load markdown files as Documents
#    SentenceSplitter       → split each Document into overlapping Nodes (chunks)
#    HuggingFaceEmbedding   → embed each Node text into a dense float vector
#    ChromaVectorStore      → store vectors in Chroma (in-memory)
#    VectorStoreIndex       → LlamaIndex index abstraction over the vector store
#
#  Why EphemeralClient instead of PersistentClient?
#  chromadb 1.x's Rust backend has an SQLite schema bug with PersistentClient
#  ("no such table: collections") on this Ollama version's environment.
#  EphemeralClient keeps everything in-memory for the life of the kernel —
#
#
#  Why create the client only once?
#  chromadb 1.x's Rust backend shares SQLite state across all EphemeralClient
#  instances. Replacing the client (e.g. on re-run) causes the old instance's
#  GC to corrupt the shared schema, producing "no such table: collections".
#  Fix: keep one client per kernel session; delete + recreate the collection
#  to get a clean slate on each ingest() call.
# ─────────────────────────────────────────────────────────────

from llama_index.core import (
    VectorStoreIndex,                       # LlamaIndex index class — wraps a vector store for search
    SimpleDirectoryReader,                  # loads all files in a folder as LlamaIndex Documents
    StorageContext,                         # bundles all storage backends (vector, doc, index stores)
    Settings,                               # global LlamaIndex config object (embed model, LLM, etc.)
)
from llama_index.core.node_parser import SentenceSplitter  # splits Documents into Nodes at sentence boundaries
from llama_index.vector_stores.chroma import ChromaVectorStore  # LlamaIndex adapter that wraps a Chroma collection
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # loads and runs a local HuggingFace embedding model
import chromadb                             # Chroma client library for creating and querying vector collections

_chroma_client = None                       # module-level singleton; shared by ingest() and load_index()


def ingest() -> VectorStoreIndex:
    """Load documents from DATA_DIR, chunk, embed, and store in-memory."""
    global _chroma_client
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)  # set the global embedding model
    Settings.llm = None                     # disable LlamaIndex's own LLM — we use LangChain for generation

    print(f"📂  Loading documents from {DATA_DIR}/ ...")
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()  # read every file in data/ into Document objects
    print(f"    Loaded {len(documents)} document(s).")

    splitter = SentenceSplitter(            # create the text chunker
        chunk_size=CHUNK_SIZE,              # maximum tokens per chunk
        chunk_overlap=CHUNK_OVERLAP         # shared tokens between adjacent chunks
    )

    # Create the client once; reuse it on subsequent ingest() calls to avoid
    # Rust backend GC corrupting the shared SQLite schema (chromadb 1.x bug).
    if _chroma_client is None:
        _chroma_client = chromadb.EphemeralClient()  # in-memory — no disk, no SQLite schema issues

    # Delete the old collection (if it exists) so re-runs start from a clean slate
    try:
        _chroma_client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass                                # collection didn't exist yet — that's fine

    chroma_collection = _chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)  # wrap in LlamaIndex adapter
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("⚙️   Building index (downloads BGE model on first run, ~130 MB) ...")
    index = VectorStoreIndex.from_documents(  # main call: chunk → embed → store
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    print("✅  Index built and held in memory (re-run ingest() after kernel restart)")
    return index


def load_index() -> VectorStoreIndex:
    """Return the in-memory index built by ingest() — no re-embedding needed."""
    if _chroma_client is None:
        raise RuntimeError("No index in memory. Run ingest() first.")  # EphemeralClient doesn't persist across kernels
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = None
    chroma_collection = _chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
```

### 2b. Run ingestion (cell 8)

#### Line-by-line
- `index = ingest()` — the single call that triggers the entire load → chunk → embed → store pipeline defined above. The cell's comment notes the BGE embedding model downloads once (~130 MB) on the very first run and is cached thereafter, and that re-running this cell re-embeds everything from scratch — if a learner only wants to *reconnect* to an already-built index without re-embedding, they should call `load_index()` instead.

#### Full code block
```python
# Run the full ingestion pipeline.
# The BGE model (~130 MB) downloads on the first run — all subsequent runs skip the download.
# Re-running this cell re-embeds everything. To skip re-embedding, use load_index() instead.

index = ingest()                            # load → chunk → embed → persist; returns VectorStoreIndex
```

### 2c. Visualize corpus statistics (cell 9)

#### Line-by-line
- `import matplotlib.pyplot as plt` and `import numpy as np` — standard plotting and numeric libraries used for the two-panel bar chart.
- `doc_names, doc_chars = [], []` — two parallel lists that will be filled in lock-step: one document name per entry, one character count per entry.
- `for f in sorted(DATA_DIR.glob("*.md")):` — iterates every markdown file in `data/` in alphabetical order (`glob("*.md")` matches only markdown files, so any non-`.md` artifacts in the folder are ignored).
- `doc_names.append(f.stem[:30])` — `f.stem` is the filename without its extension (e.g. `"pto_policy"` from `"pto_policy.md"`); truncating to 30 characters keeps long curriculum filenames from overflowing the y-axis labels.
- `doc_chars.append(len(f.read_text()))` — reads the full file content and records its character count — a quick, computation-free proxy for document size.
- `chars_per_token = 4` — a widely used rough heuristic (roughly 4 characters per token for English text) used here purely for a ballpark chunk-count estimate, not an exact tokenizer count.
- `estimated_chunks = [max(1, round(c / (CHUNK_SIZE * chars_per_token))) for c in doc_chars]` — estimates how many `CHUNK_SIZE`-token chunks each document will produce: converts `CHUNK_SIZE` tokens to an estimated character count, divides the document's character count by that, and rounds. `max(1, ...)` guarantees every document shows at least 1 estimated chunk even if it's shorter than one chunk's worth of text.
- `fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * len(doc_names) + 2))` — creates a figure with two side-by-side subplots; the height scales with the number of documents (`0.5 * len(doc_names)`) so the horizontal bar chart stays readable whether there are 5 documents or 20.
- `fig.suptitle(...)` — the overall figure title, shown above both subplots.
- `ax1 = axes[0]` / `blue_shades = plt.cm.Blues(np.linspace(0.45, 0.9, len(doc_names)))` — generates one shade of blue per document by sampling the `Blues` colormap at evenly spaced points between 0.45 and 0.9 (avoiding the colormap's near-white and near-black extremes, which would be hard to read against a white background or black text).
- `bars1 = ax1.barh(doc_names, doc_chars, color=blue_shades)` — a horizontal bar chart (`barh`) — horizontal bars are used here specifically because document names as y-axis labels stay horizontal and readable, unlike x-axis labels on a vertical bar chart which would need rotation.
- `ax1.set_xlabel("Characters")`, `ax1.set_title("Document Size")` — axis and subplot labeling.
- `ax1.invert_yaxis()` — by default `barh` plots the first list item at the bottom; inverting the y-axis puts it at the top instead, so the bars read top-to-bottom in the same order as `doc_names` (matching the alphabetical file order from the `glob` iteration).
- `for bar, val in zip(bars1, doc_chars): ax1.text(val + max(doc_chars) * 0.01, bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center", fontsize=8)` — annotates each bar with its exact character count, formatted with a thousands separator (`{val:,}`). The x-position (`val + max(doc_chars) * 0.01`) places the label just past the end of each bar (offset by 1% of the largest bar's length so it doesn't overlap the bar itself); the y-position (`bar.get_y() + bar.get_height() / 2`) centers the text vertically on the bar.
- `ax2 = axes[1]` and the mirrored block for `estimated_chunks` — identical structure to the left panel but using an orange colormap and the estimated-chunk-count data, so the two panels are visually distinct at a glance (blue = size, orange = estimated chunks) while sharing the same document ordering.
- `plt.tight_layout()` — auto-adjusts subplot spacing so labels and titles don't overlap or get clipped.
- `plt.show()` — renders the figure in the notebook output.
- `print(f"Total documents: {len(doc_names)} | Total estimated chunks: {sum(estimated_chunks)}")` — a plain-text summary line beneath the chart, useful for a quick sanity check without having to read values off the bars.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  VISUALIZATION — Corpus Statistics
#  After ingestion, show document sizes and estimated chunk
#  counts so you can spot imbalances before querying.
# ─────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np

doc_names, doc_chars = [], []
for f in sorted(DATA_DIR.glob("*.md")):
    doc_names.append(f.stem[:30])          # truncate long names for readability
    doc_chars.append(len(f.read_text()))

chars_per_token  = 4                        # rough estimate: 1 token ≈ 4 characters
estimated_chunks = [max(1, round(c / (CHUNK_SIZE * chars_per_token))) for c in doc_chars]

fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * len(doc_names) + 2))
fig.suptitle("Corpus Statistics After Ingestion", fontsize=13, fontweight="bold")

# Left: document character counts
ax1 = axes[0]
blue_shades = plt.cm.Blues(np.linspace(0.45, 0.9, len(doc_names)))
bars1 = ax1.barh(doc_names, doc_chars, color=blue_shades)
ax1.set_xlabel("Characters")
ax1.set_title("Document Size")
ax1.invert_yaxis()
for bar, val in zip(bars1, doc_chars):
    ax1.text(val + max(doc_chars) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:,}", va="center", fontsize=8)

# Right: estimated chunk counts
ax2 = axes[1]
orange_shades = plt.cm.Oranges(np.linspace(0.45, 0.9, len(doc_names)))
bars2 = ax2.barh(doc_names, estimated_chunks, color=orange_shades)
ax2.set_xlabel(f"Estimated Chunks  (chunk_size={CHUNK_SIZE} tokens, ~4 chars/token)")
ax2.set_title("Estimated Chunk Count per Document")
ax2.invert_yaxis()
for bar, val in zip(bars2, estimated_chunks):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
             str(val), va="center", fontsize=8)

plt.tight_layout()
plt.show()
print(f"Total documents: {len(doc_names)}  |  Total estimated chunks: {sum(estimated_chunks)}")
```

### 2d. `retrieve()` (cell 10)

#### Why this section exists
This function is deliberately the clean boundary — the "seam" — between the LlamaIndex data layer and everything downstream. LangChain and LangGraph code call `retrieve()` and never import anything from LlamaIndex directly, which means the data layer could later be swapped for a different retrieval backend without touching Stage 2 or Stage 3 code at all.

#### Line-by-line
- `def retrieve(query: str, top_k: int = TOP_K) -> List[str]:` — defaults `top_k` to the global `TOP_K` constant (4) but allows any caller to override it per-call. Returns a plain `List[str]` — deliberately *not* LlamaIndex objects — reinforcing the seam: downstream code only ever sees strings.
- `idx = load_index()` — reconnects to the already-built in-memory index (cheap — no re-embedding), rather than calling `ingest()` again, which would be far slower.
- `retriever = idx.as_retriever(similarity_top_k=top_k)` — builds a retriever object configured to return the `top_k` nearest chunks by cosine similarity to the query embedding.
- `nodes = retriever.retrieve(query)` — embeds `query` using the same embedding model set on `Settings.embed_model`, runs a nearest-neighbor search against the Chroma collection, and returns a list of `NodeWithScore` objects (each pairing a chunk's text/metadata with its similarity score).
- `return [n.node.get_content() for n in nodes]` — extracts just the plain text content from each `NodeWithScore` (`n.node` is the underlying chunk, `.get_content()` returns its text), discarding the score and metadata — this is what makes the return type `List[str]` rather than a list of LlamaIndex-specific objects.
- `print("✅  retrieve() function defined.")` — confirms the function was defined without error.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 1 — RETRIEVER   (mirrors src/retriever.py)
#
#  This function is the SEAM between the data layer and the rest.
#  LangChain and LangGraph cells call retrieve() directly —
#  they never import anything from LlamaIndex.
#  This clean interface makes each layer independently swappable.
# ─────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    """Return the top-k most relevant chunk texts for `query`."""
    idx = load_index()                      # reload the persisted index (fast — just opens Chroma)
    retriever = idx.as_retriever(           # create a retriever object from the index
        similarity_top_k=top_k             # how many chunks to return (ranked by cosine similarity)
    )
    nodes = retriever.retrieve(query)       # embed the query and find the nearest chunk vectors
    return [n.node.get_content() for n in nodes]  # extract plain text from each NodeWithScore object


print("✅  retrieve() function defined.")
```

### 2e. Stage 1 verification — eyeball the chunks (cell 11)

#### Line-by-line
- `test_query = "How many PTO days do new employees get?"` — the same "easy" question that Stage 3 will later run through the full graph; using it here first establishes a baseline for what good retrieval looks like before any grading/generation logic is layered on top.
- `chunks = retrieve(test_query)` — calls the function just defined, returning up to `TOP_K` chunk strings.
- `print(f"Query: {test_query!r}")` — the `!r` conversion flag prints the query with its quotes included (`'How many PTO days...'`), making the printed line self-describing.
- `print(f"Retrieved {len(chunks)} chunk(s):\n")` — a count check; if this doesn't equal `TOP_K`, fewer relevant chunks exist than requested.
- `for i, chunk in enumerate(chunks, 1):` — `start=1` gives human-friendly 1-based numbering ("Chunk 1", "Chunk 2", ...) instead of the 0-based default.
- `print(f"─── Chunk {i} " + "─" * 48)` — builds a visual divider line between chunks using repeated box-drawing dash characters, so each chunk is clearly delimited in console output.
- `print(textwrap.fill(chunk[:500], width=72))` — `chunk[:500]` truncates very long chunks to their first 500 characters (chunks can run close to `CHUNK_SIZE` tokens, which would otherwise print as one very long unwrapped line); `textwrap.fill(..., width=72)` reflows that text into lines no wider than 72 characters, which is what actually makes it human-readable in a notebook cell output or terminal.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 1 VERIFICATION — Eyeball the chunks
#
#  Read these chunks carefully before adding any LLM.
#  Are they relevant to the question? Is the text complete?
# ─────────────────────────────────────────────────────────────

test_query = "How many PTO days do new employees get?"  # the question we want to answer in Stage 3
chunks = retrieve(test_query)               # call the retriever we just defined above

print(f"Query: {test_query!r}")
print(f"Retrieved {len(chunks)} chunk(s):\n")
for i, chunk in enumerate(chunks, 1):      # enumerate starting at 1 for human-readable numbering
    print(f"─── Chunk {i} " + "─" * 48)
    print(textwrap.fill(chunk[:500], width=72))  # wrap long lines at 72 chars for readable output
    print()
```

### 2f. Visualize retrieval similarity scores (cell 12)

#### Line-by-line
- `import matplotlib.pyplot as plt` and `import textwrap as tw` — re-imports matplotlib under the same alias as before (harmless — Python caches modules, so this is effectively free) and imports `textwrap` under a *different* alias (`tw`) than the module-level `textwrap` imported in cell 3, purely a local naming choice in this cell.
- `def retrieve_with_scores(query: str, top_k: int = TOP_K):` — a variant of `retrieve()` that keeps the similarity score instead of discarding it, specifically for this visualization; it duplicates most of `retrieve()`'s body rather than modifying `retrieve()` itself, keeping the "seam" function's return type (`List[str]`) unchanged for downstream callers.
- `idx = load_index()` / `nodes = idx.as_retriever(similarity_top_k=top_k).retrieve(query)` — same retrieval mechanics as `retrieve()`, chained onto one line here since the intermediate `retriever` variable isn't reused elsewhere in this cell.
- `return [(n.node.get_content(), n.score) for n in nodes]` — returns a list of `(text, score)` tuples instead of just text — `n.score` is the cosine similarity LlamaIndex attached to each `NodeWithScore` during retrieval.
- `vis_query = "How many PTO days do new employees get?"` — the same test query as the previous cell, so this chart directly visualizes the confidence behind the chunks already eyeballed in Section 2e.
- `scored = retrieve_with_scores(vis_query)` — runs the query and captures both text and score.
- `labels = [f"Chunk {i+1}" for i in range(len(scored))]` — builds x-axis category labels ("Chunk 1", "Chunk 2", ...).
- `scores = [s for _, s in scored]` — unpacks just the score half of each tuple into its own list for plotting (`_` is the conventional "don't need this" placeholder for the discarded text).
- `previews = [tw.shorten(text[:100], width=55, placeholder="…") for text, _ in scored]` — builds a short preview string for each chunk: first take the first 100 characters, then `textwrap.shorten` collapses it further to fit within 55 characters, breaking on whole words and appending an ellipsis (`"…"`) if truncated, rather than cutting mid-word.
- `bar_colors = ["#2ecc71" if s >= 0.7 else "#f39c12" if s >= 0.5 else "#e74c3c" for s in scores]` — a chained conditional expression that color-codes each bar by its own similarity score: green (`#2ecc71`) for high relevance (≥ 0.7), orange/yellow (`#f39c12`) for medium (≥ 0.5), red (`#e74c3c`) for low (< 0.5) — matching the thresholds documented in the cell's header comment.
- `fig, ax = plt.subplots(figsize=(10, 4.5))` — a single-panel figure this time (unlike the two-panel corpus-stats chart).
- `bars = ax.bar(labels, scores, color=bar_colors, edgecolor="white", linewidth=1)` — a standard vertical bar chart; `edgecolor="white"` with `linewidth=1` draws a thin white outline around each bar, which helps visually separate adjacent bars of similar color.
- `ax.set_ylim(0, 1.15)` — caps the y-axis just above 1.0 (cosine similarity scores here are bounded near 0–1) with a little headroom (1.15) so the score labels printed above each bar (see below) don't get clipped at the top of the plot.
- `ax.set_ylabel(...)`, `ax.set_title(f'Retrieval Scores — "{vis_query}"', ...)` — standard axis/plot labeling, with the query text embedded directly in the title.
- `ax.axhline(0.7, color="#2ecc71", ..., label="High relevance (≥ 0.7)")` and the `0.5` line — draws two dashed horizontal reference lines at the same thresholds used to color the bars, so a reader can visually confirm which bars cross which threshold without cross-referencing the color legend.
- `for bar, score, preview in zip(bars, scores, previews):` — iterates the three parallel sequences together to annotate each bar individually.
  - `cx = bar.get_x() + bar.get_width() / 2` — computes the horizontal center of each bar for text placement.
  - `ax.text(cx, score + 0.025, f"{score:.3f}", ha="center", va="bottom", ...)` — prints the exact score (3 decimal places) just above each bar's top.
  - `ax.text(cx, -0.06, preview, ha="center", va="top", fontsize=7, color="#444", rotation=12)` — prints the shortened chunk-text preview just below the x-axis, slightly rotated (12°) so overlapping preview strings for adjacent bars are easier to read without colliding.
- `ax.legend(loc="upper right", fontsize=9)` — shows the two threshold-line labels defined via `axhline(..., label=...)`.
- `plt.tight_layout()` then `plt.subplots_adjust(bottom=0.22)` — `tight_layout()` does its usual automatic spacing pass, then `subplots_adjust` explicitly reserves extra space (22% of the figure height) at the bottom specifically to make room for the rotated preview text placed below the x-axis, which `tight_layout()` alone wouldn't account for.
- `plt.show()` — renders the chart.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  VISUALIZATION — Retrieval Similarity Scores
#  Cosine similarity for each returned chunk reveals how
#  confidently the vector index matched the query.
#  Green ≥ 0.7 | Yellow ≥ 0.5 | Red < 0.5
# ─────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import textwrap as tw


def retrieve_with_scores(query: str, top_k: int = TOP_K):
    """Return list of (chunk_text, cosine_score) for the query."""
    idx = load_index()
    nodes = idx.as_retriever(similarity_top_k=top_k).retrieve(query)
    return [(n.node.get_content(), n.score) for n in nodes]


vis_query    = "How many PTO days do new employees get?"
scored       = retrieve_with_scores(vis_query)
labels       = [f"Chunk {i+1}" for i in range(len(scored))]
scores       = [s for _, s in scored]
previews     = [tw.shorten(text[:100], width=55, placeholder="…") for text, _ in scored]
bar_colors   = ["#2ecc71" if s >= 0.7 else "#f39c12" if s >= 0.5 else "#e74c3c"
                for s in scores]

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(labels, scores, color=bar_colors, edgecolor="white", linewidth=1)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Cosine Similarity Score")
ax.set_title(f'Retrieval Scores — "{vis_query}"', fontsize=12, fontweight="bold")

ax.axhline(0.7, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.8,
           label="High relevance (≥ 0.7)")
ax.axhline(0.5, color="#f39c12", linestyle="--", linewidth=1, alpha=0.8,
           label="Medium relevance (≥ 0.5)")

for bar, score, preview in zip(bars, scores, previews):
    cx = bar.get_x() + bar.get_width() / 2
    ax.text(cx, score + 0.025, f"{score:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(cx, -0.06, preview, ha="center", va="top",
            fontsize=7, color="#444", rotation=12)

ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.show()
```

---

## 3. Stage 2 — LangChain: the generation layer (cells 15–18)

### Why this section exists
LangChain's job here is to turn "call an LLM" into "get back a typed, validated Python object." The notebook's own markdown (cell 14) states the key advantage directly: once an answer is a Pydantic object, later code (Stage 3's graph nodes) can read `grade.verdict` or `answer.confidence` as plain attribute access — never string-parsing raw LLM output.

### 3a. Output models — `CitedAnswer` and `RelevanceGrade` (cell 15)

#### Line-by-line
- `from pydantic import BaseModel, Field` — `BaseModel` is Pydantic's base class for defining a validated data schema; `Field` attaches metadata (here, a `description` used both as documentation and, as seen in cells 16–17, folded into the LLM's system prompt so the model knows what each field means).
- `class CitedAnswer(BaseModel):` — defines the exact shape every generated answer must take.
- `answer: str = Field(description="A clear, direct answer based only on the provided context.")` — the free-text answer field; the description explicitly constrains it to context-grounded content, both documenting intent and (via the prompt) instructing the LLM.
- `sources: List[str] = Field(description="Document names or section titles that support the answer.")` — a list of citation strings the model should return alongside its answer, giving the user a way to verify the claim.
- `confidence: Literal["high", "medium", "low"] = Field(description=(...))` — `Literal["high", "medium", "low"]` is a type-level constraint: Pydantic will reject (raise a validation error on) any value for this field that isn't exactly one of those three strings, rather than allowing free text. The description explains the semantics: `high` = context fully answers, `medium` = partial, `low` = tangential.
- `class RelevanceGrade(BaseModel):` — the second schema, used by the grader chain (a completely separate LLM call from the generation chain).
- `verdict: Literal["relevant", "not_relevant"] = Field(...)` — again a `Literal` type constrains the grader to exactly one of two allowed strings — this binary verdict is what Stage 3's conditional routing branches on.
- `reason: str = Field(description="One sentence explaining the verdict.")` — a short free-text justification, primarily useful for the human-readable trace printed by each graph node in Stage 3.
- `print("✅  Pydantic models defined: CitedAnswer, RelevanceGrade")` — confirmation the class definitions parsed without error.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 — OUTPUT MODELS
#
#  Define the SHAPE of what we want back from the LLM before
#  writing a single prompt.  Pydantic enforces the schema so
#  we always get a Python object — never raw text to parse.
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field       # Pydantic: define data schemas with validation


class CitedAnswer(BaseModel):               # the structured shape of every generated answer
    """Structured answer returned by the generation chain."""
    answer: str = Field(                    # the text answer — must come from the provided context only
        description="A clear, direct answer based only on the provided context."
    )
    sources: List[str] = Field(             # list of document names that back up the answer
        description="Document names or section titles that support the answer."
    )
    confidence: Literal["high", "medium", "low"] = Field(  # forces exactly one of three values
        description=(
            "high: context directly and completely answers the question. "
            "medium: context partially answers it. "
            "low: context is only tangentially related."
        )
    )


class RelevanceGrade(BaseModel):            # the grader's verdict for Stage 3's conditional routing
    """Grader verdict on whether retrieved context is useful for a question."""
    verdict: Literal["relevant", "not_relevant"] = Field(  # exactly two allowed verdicts
        description="'relevant' if the context helps answer the question, 'not_relevant' otherwise."
    )
    reason: str = Field(                    # one-sentence explanation shown in the node-path trace
        description="One sentence explaining the verdict."
    )


print("✅  Pydantic models defined: CitedAnswer, RelevanceGrade")
```

### 3b. Generation chain (cell 16)

#### Line-by-line
- `from langchain_ollama import ChatOllama` — LangChain's chat-model wrapper around a locally running Ollama server, exposing the same standard interface (`.invoke()`, pipe-composability) as any other LangChain chat model.
- `from langchain_core.prompts import ChatPromptTemplate` — builds a template composed of role-tagged messages (`"system"`, `"human"`) with `{placeholder}` slots that get filled in at invocation time.
- `from langchain_core.output_parsers import JsonOutputParser` — a parser that takes a raw LLM response string, expects it to be a JSON document, and returns the parsed Python dict.
- `GENERATION_PROMPT = ChatPromptTemplate.from_messages([...])` — builds the two-message template. The cell's header comment explains an important environment constraint: this Ollama version (`<0.5.0`) doesn't support passing a full JSON *schema* through the `format` parameter, so instead the schema is spelled out in plain language inside the system prompt, and JSON output is only loosely enforced by `format="json"` (see below) plus explicit instructions.
- The `"system"` message — establishes the assistant's persona ("helpful assistant for Meridian Software employees"), a hard constraint ("Answer questions using ONLY the provided context... do not invent or extrapolate policies") which is the core anti-hallucination instruction, and then spells out the exact JSON shape expected (`answer`, `sources`, `confidence` with allowed values) as plain text — this is the manual schema-description workaround mentioned above.
- The `"human"` message — injects `{context}` (the retrieved chunks, joined) and `{question}` (the user's question) into a templated instruction, ending with an explicit reminder to answer from the context only.
- `_generation_llm = ChatOllama(model=LLM_MODEL, format="json")` — instantiates the chat model bound to `LLM_MODEL` (`phi3:latest`). `format="json"` tells Ollama to constrain its output to be syntactically valid JSON (though not necessarily matching any particular schema — that part relies on the prompt instructions above). The leading underscore marks this as an internal helper variable, not meant to be used directly outside this cell.
- `generation_chain = (GENERATION_PROMPT | _generation_llm | JsonOutputParser() | (lambda d: CitedAnswer(**d)))` — a LangChain **Runnable pipeline** built with the `|` operator, where each stage's output becomes the next stage's input:
  1. `GENERATION_PROMPT` — fills in `{context}`/`{question}` and produces the final message list.
  2. `_generation_llm` — sends those messages to Ollama and gets back a raw JSON-formatted string response.
  3. `JsonOutputParser()` — parses that string into a plain Python `dict`.
  4. `(lambda d: CitedAnswer(**d))` — the final stage is an inline lambda rather than a named function; it unpacks the dict's keys as keyword arguments into `CitedAnswer(...)`, which triggers Pydantic validation — if the LLM's JSON doesn't have the right keys or the `confidence` value isn't one of the three allowed literals, this line raises a validation error rather than silently passing through malformed data.
- `print("✅  generation_chain ready: ...")` — confirms the chain object was constructed (this does not call the LLM — chains are lazy; nothing runs until `.invoke()` is called, as happens in cell 18).

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 — GENERATION CHAIN   (mirrors src/generation.py)
#
#  Ollama <0.5.0 does not support JSON-schema in the `format`
#  field, so we use format="json" (forces valid JSON output)
#  and add schema instructions to the system prompt instead.
#  JsonOutputParser parses the JSON string → dict, then we
#  validate it into a CitedAnswer Pydantic object.
#
#  Chain pattern:
#    ChatPromptTemplate | ChatOllama(format="json") | JsonOutputParser() | CitedAnswer
# ─────────────────────────────────────────────────────────────

from langchain_ollama import ChatOllama                      # LangChain adapter: wraps a local Ollama model
from langchain_core.prompts import ChatPromptTemplate        # builds structured (system, human) prompt templates
from langchain_core.output_parsers import JsonOutputParser   # parses the LLM's JSON string into a Python dict


GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",                            # system message: role + JSON schema instructions
        "You are a helpful assistant for Meridian Software employees. "
        "Answer questions using ONLY the provided context. "
        "If the context does not contain enough information, say so — "
        "do not invent or extrapolate policies.\n\n"
        "Respond with a JSON object containing exactly these fields:\n"
        '  "answer":     string — a clear, direct answer based only on the context\n'
        '  "sources":    list of strings — document names or section titles that support the answer\n'
        '  "confidence": one of "high", "medium", or "low"\n'
        "                high=context directly answers it, medium=partial, low=tangential",
    ),
    (
        "human",                             # human message: inject retrieved context and the question
        "Context from the employee handbook:\n\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer the question using only the context above.",
    ),
])

_generation_llm = ChatOllama(model=LLM_MODEL, format="json")  # format="json" forces valid JSON output
generation_chain = (
    GENERATION_PROMPT
    | _generation_llm
    | JsonOutputParser()                     # parses JSON string → Python dict
    | (lambda d: CitedAnswer(**d))           # validates dict → CitedAnswer Pydantic object
)

print("✅  generation_chain ready:  prompt → ChatOllama(json) → CitedAnswer")
```

### 3c. Grader chain (cell 17)

#### Why this section exists
The grader is a second, independent LLM call whose only job is to answer one narrow question — "is this retrieved context relevant to the user's question?" — and return a structured verdict. In Stage 3, that verdict becomes the condition the control graph branches on.

#### Line-by-line
- `GRADER_PROMPT = ChatPromptTemplate.from_messages([...])` — a separate prompt template from `GENERATION_PROMPT`, built the same way (system + human messages) but for a different task.
- The `"system"` message — establishes a strict, skeptical "grader" persona rather than a helpful-assistant persona, explicitly instructing the model to mark borderline/tangential context as `not_relevant` rather than being lenient — the comment notes this "avoids leniency bias," a known failure mode where LLM judges default to rating most things as acceptable.
- The `"human"` message — injects `{context}` (the same joined retrieved chunks used for generation) and `{question}` (the original question), then asks directly: "Is this passage relevant to the question?"
- `_grader_llm = ChatOllama(model=GRADER_MODEL, format="json")` — a second `ChatOllama` instance bound to `GRADER_MODEL` (currently also `phi3:latest`, but architecturally independent — a learner could point grading at a different, e.g. faster or stricter, model without touching generation).
- `grader_chain = (GRADER_PROMPT | _grader_llm | JsonOutputParser() | (lambda d: RelevanceGrade(**d)))` — the identical four-stage pipe pattern as `generation_chain`, but ending in `RelevanceGrade` instead of `CitedAnswer`. Reusing the exact same pattern (prompt → LLM(json) → JsonOutputParser → Pydantic-validate) for two structurally different tasks is the core reusable idiom this section is teaching.
- `print("✅  grader_chain ready: ...")` — confirms construction, no LLM call yet.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 — GRADER CHAIN   (mirrors src/grader.py)
#
#  The grader is a second LLM call that acts as a judge.
#  It answers one question: "Is this retrieved context relevant?"
#
#  Ollama <0.5.0 does not support JSON-schema in the `format`
#  field, so we use format="json" (forces valid JSON output)
#  and add schema instructions to the system prompt instead.
#  JsonOutputParser parses the JSON string → dict, then we
#  validate it into a RelevanceGrade Pydantic object.
#
#  In Stage 3, the verdict drives the graph's conditional routing.
# ─────────────────────────────────────────────────────────────

GRADER_PROMPT = ChatPromptTemplate.from_messages([  # separate prompt template for the grader
    (
        "system",                           # strict grader persona — avoids leniency bias
        "You are a retrieval quality grader. "
        "Decide whether a retrieved passage can meaningfully help answer a user's question. "
        "Be strict: if the passage is off-topic or only tangentially related, "
        "mark it as not_relevant.\n\n"
        "Respond with a JSON object containing exactly these fields:\n"
        '  "verdict": either "relevant" or "not_relevant"\n'
        '  "reason":  one sentence explaining your verdict',
    ),
    (
        "human",                            # inject the retrieved text and the question
        "Retrieved passage:\n\n{context}\n\n"  # {context} = the chunks joined into one string
        "User question: {question}\n\n"   # {question} = the original user question
        "Is this passage relevant to the question?",
    ),
])

_grader_llm = ChatOllama(model=GRADER_MODEL, format="json")  # format="json" forces valid JSON output
grader_chain = (
    GRADER_PROMPT
    | _grader_llm
    | JsonOutputParser()                    # parses JSON string → Python dict
    | (lambda d: RelevanceGrade(**d))       # validates dict → RelevanceGrade Pydantic object
)

print("✅  grader_chain ready:  prompt → ChatOllama(json) → RelevanceGrade")
```

### 3d. Stage 2 verification — linear RAG, no graph yet (cell 18)

#### Why this section exists
Before assembling the full self-correcting graph, this cell confirms the two pieces built so far — Stage 1 retrieval and Stage 2 generation — connect correctly, using the simplest possible flow: retrieve, then generate, with no grading and no retry loop. The cell explicitly calls out that this version *will* hallucinate on out-of-scope questions, motivating why Stage 3 exists.

#### Line-by-line
- `def linear_rag(question: str) -> CitedAnswer:` — a minimal, non-self-correcting RAG function, used purely as a sanity check.
- `chunks = retrieve(question)` — Stage 1's retriever, called directly with the raw question (no query rewriting here — that capability doesn't exist yet at this point in the notebook).
- `context_text = "\n\n---\n\n".join(chunks)` — joins the retrieved chunk strings into one block of text, separated by a clearly visible `"---"` divider on its own line so the LLM (and any human reading the prompt) can tell where one chunk ends and the next begins.
- `return generation_chain.invoke({"context": context_text, "question": question})` — calls the Stage 2 generation chain built in cell 16, filling its two template slots, and returns the resulting `CitedAnswer` object directly.
- `q = "How many PTO days do new employees get?"` — the "easy" question again, used consistently across the notebook as the answerable baseline case.
- `ans = linear_rag(q)` — runs the full retrieve-then-generate flow once.
- `print(f"Question: {q}")`, `print(f"Answer: {ans.answer}")`, `print(f"Confidence: {ans.confidence}")`, `print(f"Sources: {ans.sources}")` — because `ans` is a validated `CitedAnswer` Pydantic object, each field is accessed as a plain attribute (`ans.answer`, `ans.confidence`, `ans.sources`) — no string parsing or dict-key lookups needed, which is the structured-output payoff the section's intro promised.
- The trailing `print("─" * 60)` and two explanatory `print(...)` lines — plain narrative text warning that this linear version has no self-correction and will hallucinate on questions the handbook doesn't cover, explicitly setting up Stage 3 as the fix.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 VERIFICATION — Linear RAG (no graph yet)
#
#  Flow: retrieve → generate   (no grading, no retry)
#  This confirms Stage 1 retrieval + Stage 2 generation connect.
# ─────────────────────────────────────────────────────────────

def linear_rag(question: str) -> CitedAnswer:  # a straight-line RAG call with no self-correction
    """Retrieve chunks, then generate a structured answer. No looping."""
    chunks = retrieve(question)             # Stage 1: get the top-k relevant chunks
    context_text = "\n\n---\n\n".join(chunks)  # join chunks with a separator so the LLM sees clear boundaries
    return generation_chain.invoke(         # Stage 2: run the LangChain chain
        {"context": context_text,           # fill the {context} slot in the prompt
         "question": question}              # fill the {question} slot in the prompt
    )                                       # returns a CitedAnswer Pydantic object


q   = "How many PTO days do new employees get?"  # a question that IS in the handbook
ans = linear_rag(q)                         # run the linear RAG call

print(f"Question:   {q}")
print(f"Answer:     {ans.answer}")          # the text answer extracted from CitedAnswer
print(f"Confidence: {ans.confidence}")      # "high", "medium", or "low"
print(f"Sources:    {ans.sources}")         # list of source names cited by the model
print()
print("─" * 60)
print("NOTE: this linear version has no self-correction.")
print("Try asking it something not in the docs — it will hallucinate.")
print("Stage 3 fixes that.")
```

---

## 4. Stage 3 — LangGraph: the control layer (cells 20–26)

### Why this section exists
The notebook's own diagram (cell 19) lays out the corrective-RAG graph: `retrieve → grade`, and then a conditional branch — relevant context routes to `generate`, not-relevant context with retries remaining routes to `rewrite` (which loops back to `retrieve`), and not-relevant context with retries exhausted routes to `fallback`. Unlike a LangChain chain (a straight line), a LangGraph graph can loop — this is what makes the assistant self-correcting rather than one-shot.

### 4a. Graph state (cell 20)

#### Line-by-line
- `class GraphState(TypedDict):` — the shared "notebook" every node in the graph reads from and writes to. Because it's a `TypedDict`, LangGraph can validate at runtime that a node's returned dict only contains declared keys of the declared types.
- `question: str` — the original user question, set once when the graph is invoked and never overwritten by any node — this is what stays constant across retry loops, distinguishing it from `search_query` below.
- `search_query: str` — the *current* query text used for retrieval; starts equal to `question` but may be replaced by `node_rewrite` on a retry loop.
- `context: List[str]` — the chunk texts returned by the most recent `retrieve()` call; overwritten (not accumulated) each time `node_retrieve` runs.
- `grade: str` — the grader's verdict string (`"relevant"` or `"not_relevant"`), set by `node_grade` and read by the routing function to decide the next edge.
- `grade_reason: str` — the grader's one-sentence explanation, kept for the printed trace/transparency but not used in any routing decision.
- `answer: Optional[dict]` — `None` until either `node_generate` or `node_fallback` sets it; typed as `Optional[dict]` because it can legitimately be absent for most of the graph's execution.
- `retry_count: int` — incremented once per pass through `node_rewrite`; compared against `MAX_RETRIES` by the routing function to decide when to stop looping and fall back.
- `node_path: Annotated[List[str], add]` — the execution trace. The `Annotated[List[str], add]` type attaches `operator.add` (imported in cell 3) as this field's **reducer**: instead of each node's returned value for `node_path` *overwriting* the running list (which is LangGraph's default merge behavior for a plain field), the `add` reducer concatenates the new list onto the existing one. Since Python's `+` operator on two lists is concatenation, `add(["retrieve"], ["grade"])` extends rather than replaces, letting every node simply return `{"node_path": ["its own name"]}` and have the full path accumulate automatically across the whole run.
- `print("✅  GraphState defined")` — confirms the class definition succeeded.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 — GRAPH STATE
#
#  The state is a typed dict — a shared "notebook" every node
#  reads from and writes back to.
#
#  Each node returns a PARTIAL dict (only the keys it changed).
#  LangGraph merges those partial updates into the running state.
#
#  node_path uses operator.add as a reducer: instead of
#  overwriting the list, LangGraph APPENDS each node's name.
# ─────────────────────────────────────────────────────────────

class GraphState(TypedDict):                # typed dictionary — each field has a declared type
    question:     str                       # the original user question — set once, never changed
    search_query: str                       # the current search query — may be rewritten by node_rewrite
    context:      List[str]                 # text chunks returned by the most recent retrieve() call
    grade:        str                       # grader verdict: "relevant" or "not_relevant"
    grade_reason: str                       # the grader's one-sentence explanation of its verdict
    answer:       Optional[dict]            # final answer dict (set by node_generate or node_fallback)
    retry_count:  int                       # incremented by node_rewrite; stops the loop at MAX_RETRIES
    node_path:    Annotated[List[str], add] # execution trace — add reducer APPENDS instead of overwriting


print("✅  GraphState defined")
```

### 4b. Graph nodes (cell 21)

#### Why this section exists
Every node follows the same shape — a function that takes the full `GraphState` and returns only the partial dict of keys it changed — which is exactly the contract LangGraph expects. Keeping each node to a single job (retrieve, grade, rewrite, generate, or give up) is what makes the graph in section 4c easy to reason about.

#### Line-by-line
- `from langchain_ollama import ChatOllama` — the cell comment notes this is a repeated import (already loaded in cell 16); Python's module cache makes re-importing free, and it's included here purely so this cell is self-contained/readable on its own.
- **`node_retrieve`**
  - `def node_retrieve(state: GraphState) -> dict:` — the standard node signature: full state in, partial update out.
  - `query = state.get("search_query") or state["question"]` — uses `.get()` (safe against a missing key) for `search_query` but direct indexing for `question` (expected to always be present since it's set once at graph invocation). The `or` fallback means: if `search_query` is falsy (empty string, `None`, or unset), use the original `question` instead — in practice `search_query` is always initialized to the question at invocation time, so this mainly matters as a defensive default.
  - `chunks = retrieve(query)` — calls the Stage 1 `retrieve()` function with whichever query is currently active (original or rewritten).
  - `return {"context": chunks, "node_path": ["retrieve"]}` — a partial update: only `context` and `node_path` are returned; every other field in `GraphState` is left untouched by LangGraph's merge. `node_path` is returned as a single-element list (`["retrieve"]`) because of the `add` reducer — it gets concatenated onto the running trace, not overwritten.
- **`node_grade`**
  - `context_text = "\n\n---\n\n".join(state["context"])` — same join pattern used in `linear_rag`, applied here to the state's current `context` list.
  - `result: RelevanceGrade = grader_chain.invoke({"context": context_text, "question": state["question"]})` — calls the Stage 2 grader chain, deliberately grading against the *original* `question`, not the (possibly rewritten) `search_query` — the type annotation on `result` is purely for readability/IDE support, since Python doesn't enforce it at runtime.
  - `return {"grade": result.verdict, "grade_reason": result.reason, "node_path": ["grade"]}` — stores the verdict (which the routing function reads next) and the human-readable reason.
- **`node_rewrite`**
  - `_rewrite_prompt = ChatPromptTemplate.from_messages([("human", "...")])` — a single-message (human-only, no system message) prompt template, defined once at module level rather than rebuilt inside the node function on every call.
  - The template text — explains to the LLM that the previous query failed and asks it to rewrite the query to be more specific, ending with an explicit instruction ("return only the rewritten query, no explanation") to keep the output clean plain text rather than a chatty response.
  - `_rewrite_chain = _rewrite_prompt | ChatOllama(model=LLM_MODEL)` — a simple two-stage chain with **no** `JsonOutputParser` and **no** Pydantic model — unlike `generation_chain`/`grader_chain`, this one just returns the raw `ChatOllama` message object, because a rewritten query is just plain text with no structured fields to validate.
  - `def node_rewrite(state: GraphState) -> dict:` — the node function.
  - `old_query = state.get("search_query") or state["question"]` — same fallback pattern as `node_retrieve`, fetching whatever query is currently in play.
  - `response = _rewrite_chain.invoke({"query": old_query})` — asks the LLM to produce a better query.
  - `new_query = response.content.strip()` — `response` is a LangChain message object; `.content` extracts its raw text; `.strip()` removes any leading/trailing whitespace or newlines the LLM might have added around its answer.
  - `return {"search_query": new_query, "retry_count": state["retry_count"] + 1, "node_path": ["rewrite"]}` — updates `search_query` for the *next* pass through `node_retrieve`, and increments `retry_count` — this increment is the mechanism that eventually causes `route_after_grade` (section 4c) to stop looping and route to `fallback`.
- **`node_generate`**
  - `context_text = "\n\n---\n\n".join(state["context"])` — joins the current context, same pattern again.
  - `result: CitedAnswer = generation_chain.invoke({"context": context_text, "question": state["question"]})` — calls the Stage 2 generation chain, against the original question (never the rewritten search query, which exists only to improve *retrieval*, not to change what's being answered).
  - `return {"answer": result.model_dump(), "node_path": ["generate"]}` — `result.model_dump()` converts the Pydantic `CitedAnswer` object into a plain Python `dict` before storing it in state — necessary because `GraphState.answer` is typed as `Optional[dict]`, not as a Pydantic model, so state stays a plain, serializable structure throughout.
- **`node_fallback`**
  - The docstring and inline comment both call this "the most important node in the graph" — it is the mechanism that prevents hallucination when no relevant context can be found after exhausting all retries.
  - `return {"answer": {"answer": (...), "sources": [], "confidence": "low"}, "node_path": ["fallback"]}` — hand-builds an answer dict with the same shape as `CitedAnswer.model_dump()` would produce, so downstream code (`run_assistant`, the Gradio UI) can treat a fallback answer identically to a generated one without any special-casing. `sources: []` because nothing relevant was found to cite; `confidence: "low"` signals to any caller that this is a fallback, not a genuine high-confidence answer. The answer text itself is honest ("I could not find relevant information...") rather than an invented guess, and redirects the user to a human (HR/manager) — this is the concrete behavior that fulfills the module's "admits defeat instead of hallucinating" design goal.
- `print("✅  All 5 node functions defined: ...")` — confirms all five function definitions ran without error.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 — GRAPH NODES
#
#  Each node is a function:  GraphState → dict
#  It reads what it needs from state, does ONE job,
#  and returns only the keys it changed.
# ─────────────────────────────────────────────────────────────

from langchain_ollama import ChatOllama     # repeated import for clarity — already loaded in c14


# ── 1. retrieve ──────────────────────────────────────────────────────────────
def node_retrieve(state: GraphState) -> dict:  # node signature: receives full state, returns partial update
    """Fetch the top-k chunks for the current search query."""
    query = state.get("search_query") or state["question"]  # use rewritten query if available, else original
    print(f"  [retrieve] query={query!r}")
    chunks = retrieve(query)               # call the Stage 1 retriever
    return {
        "context":   chunks,               # store the retrieved chunks in state
        "node_path": ["retrieve"],         # append "retrieve" to the execution trace
    }


# ── 2. grade ─────────────────────────────────────────────────────────────────
def node_grade(state: GraphState) -> dict:
    """Ask the grader LLM whether the retrieved context is relevant."""
    context_text = "\n\n---\n\n".join(state["context"])  # join all chunks with a visible separator
    result: RelevanceGrade = grader_chain.invoke({  # type annotation helps editors and readers
        "context":  context_text,          # the retrieved text to evaluate
        "question": state["question"],     # the original question (not the rewritten query)
    })
    print(f"  [grade]    verdict={result.verdict!r}  |  {result.reason}")
    return {
        "grade":        result.verdict,    # "relevant" or "not_relevant" — drives the conditional edge
        "grade_reason": result.reason,     # one-sentence explanation for transparency
        "node_path":    ["grade"],         # append "grade" to the execution trace
    }


# ── 3. rewrite ───────────────────────────────────────────────────────────────
_rewrite_prompt = ChatPromptTemplate.from_messages([  # dedicated prompt for query rewriting
    (
        "human",
        "The search query below did not return useful results from an employee handbook.\n"
        "Rewrite it to be more specific and more likely to find the relevant policy.\n\n"
        "Original query: {query}\n\n"    # {query} is filled with the current (failing) query
        "Rewritten query (return only the rewritten query, no explanation):"
    )
])
_rewrite_chain = _rewrite_prompt | ChatOllama(model=LLM_MODEL)  # simple chain: prompt → raw text response

def node_rewrite(state: GraphState) -> dict:
    """Rephrase the search query to retrieve better context on the next loop."""
    old_query = state.get("search_query") or state["question"]  # current query to rephrase
    response  = _rewrite_chain.invoke({"query": old_query})  # ask the LLM for a better query
    new_query = response.content.strip()   # .content extracts the text; .strip() removes whitespace
    print(f"  [rewrite]  '{old_query}' → '{new_query}'")
    return {
        "search_query": new_query,         # the improved query that retrieve will use next loop
        "retry_count":  state["retry_count"] + 1,  # increment the counter to enforce MAX_RETRIES
        "node_path":    ["rewrite"],       # append "rewrite" to the execution trace
    }


# ── 4. generate ──────────────────────────────────────────────────────────────
def node_generate(state: GraphState) -> dict:
    """Call the generation chain to produce the final structured answer."""
    context_text = "\n\n---\n\n".join(state["context"])  # join chunks for the generation prompt
    result: CitedAnswer = generation_chain.invoke({
        "context":  context_text,          # the relevant retrieved text
        "question": state["question"],     # the original user question (not the rewritten query)
    })
    print(f"  [generate] confidence={result.confidence!r}")
    return {
        "answer":    result.model_dump(),  # convert CitedAnswer Pydantic object → plain dict for state storage
        "node_path": ["generate"],         # append "generate" to the execution trace
    }


# ── 5. fallback ──────────────────────────────────────────────────────────────
def node_fallback(state: GraphState) -> dict:
    """Return an honest 'I don't know' when MAX_RETRIES is exhausted."""
    # This node is the most important in the graph: it prevents hallucination
    # when no relevant context can be found after all retry attempts.
    print(f"  [fallback] {MAX_RETRIES} retries exhausted — returning honest answer")
    return {
        "answer": {
            "answer": (
                "I could not find relevant information in the Meridian employee handbook "  # honest admission
                "to answer this question. Please check with HR or your manager directly."  # helpful redirect
            ),
            "sources":    [],              # no sources because we found nothing relevant
            "confidence": "low",           # signals to the caller that this is a fallback answer
        },
        "node_path": ["fallback"],         # append "fallback" to the execution trace
    }


print("✅  All 5 node functions defined: retrieve, grade, rewrite, generate, fallback")
```

### 4c. Build & compile the graph (cell 22)

#### Line-by-line
- `from langgraph.graph import StateGraph, START, END` — `StateGraph` is the builder class you configure with nodes and edges; `START` and `END` are sentinel constants representing the graph's entry and exit points (not real nodes — they can't have logic attached, only edges).
- `def route_after_grade(state: GraphState) -> str:` — a **routing function**: unlike a node, it doesn't return a state update — it returns a plain string naming which node to go to next. LangGraph calls this after the `"grade"` node runs, whenever a conditional edge is configured to use it (see `add_conditional_edges` below).
- `if state["grade"] == "relevant": return "generate"` — the happy path: good context found, go straight to generation.
- `elif state["retry_count"] >= MAX_RETRIES: return "fallback"` — context is still not relevant, but the retry budget (`MAX_RETRIES`, currently 2) has been exhausted — give up gracefully rather than looping indefinitely. This comparison is what makes `MAX_RETRIES` an effective hard ceiling on the loop.
- `else: return "rewrite"` — context not relevant, but retries remain — go rephrase the query and try again.
- `builder = StateGraph(GraphState)` — creates the graph builder, parameterized on the `GraphState` schema defined in section 4a, so LangGraph can validate that every node's input/output conforms to that schema.
- `builder.add_node("retrieve", node_retrieve)` through `builder.add_node("fallback", node_fallback)` — registers each of the five functions from section 4b under a string name; that name is what's used to reference the node in edges and is what shows up in the printed execution trace (`node_path`).
- `builder.add_edge(START, "retrieve")` — a fixed (unconditional) edge: every graph run always begins at `"retrieve"`.
- `builder.add_edge("retrieve", "grade")` — fixed edge: after retrieval, always grade — there's no branching here, grading always happens.
- `builder.add_edge("rewrite", "retrieve")` — fixed edge: this is the loop-back edge that turns the graph into an actual retry loop — after rewriting the query, control always returns to `"retrieve"` to try again with the new query.
- `builder.add_edge("generate", END)` and `builder.add_edge("fallback", END)` — both terminal nodes end the graph; there are two distinct paths to `END` depending on whether generation succeeded or fallback fired.
- `builder.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})` — the one branching point in the graph. After `"grade"` runs, LangGraph calls `route_after_grade(state)`, gets back one of the three strings (`"generate"`, `"rewrite"`, `"fallback"`), and looks that string up in the provided mapping to find the actual next node to run. The mapping's keys must exactly match every possible return value of the routing function; its values are the target node names.
- `graph = builder.compile()` — finalizes the builder into a runnable object (LangGraph calls this a "Pregel" graph, after the Pregel graph-processing model). Before `.compile()`, `builder` is just configuration; after, `graph` can be `.invoke()`-d.
- `print(...)` lines — confirm the compile succeeded and restate the graph's shape and current tuning constants (`MAX_RETRIES`, `TOP_K`) for a quick visual sanity check.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 — BUILD & COMPILE THE GRAPH   (mirrors src/graph.py)
#
#  StateGraph is the builder object.  You declare nodes and
#  edges, then call .compile() to produce a runnable Pregel graph.
# ─────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END  # StateGraph=builder, START/END=sentinel node names


def route_after_grade(state: GraphState) -> str:  # routing function: inspects state, returns next node name
    """Branch after grading: generate | rewrite | fallback."""
    if state["grade"] == "relevant":        # context is good enough — go straight to generation
        return "generate"
    elif state["retry_count"] >= MAX_RETRIES:  # context is still bad after MAX_RETRIES rewrites
        return "fallback"                   # give up gracefully rather than looping forever
    else:
        return "rewrite"                    # context bad but retries remain — rephrase and try again


builder = StateGraph(GraphState)            # create the graph builder, typed to our state schema

builder.add_node("retrieve", node_retrieve)  # register node: name used in edges + trace output
builder.add_node("grade",    node_grade)    # register node
builder.add_node("rewrite",  node_rewrite)  # register node
builder.add_node("generate", node_generate) # register node
builder.add_node("fallback", node_fallback) # register node

builder.add_edge(START,      "retrieve")    # START is a sentinel: the graph enters at "retrieve"
builder.add_edge("retrieve", "grade")       # fixed edge: after every retrieve, always grade
builder.add_edge("rewrite",  "retrieve")    # fixed edge: the retry loop — rewrite feeds back into retrieve
builder.add_edge("generate", END)           # fixed edge: generation ends the graph
builder.add_edge("fallback", END)           # fixed edge: fallback also ends the graph

builder.add_conditional_edges(              # conditional: the routing function decides which edge to take
    "grade",                                # this node's output triggers the routing decision
    route_after_grade,                      # the function that inspects state and returns a node name
    {                                       # map: returned string → actual node object
        "generate": "generate",
        "rewrite":  "rewrite",
        "fallback": "fallback",
    },
)

graph = builder.compile()                   # compile the builder into a runnable LangGraph Pregel object

print("✅  LangGraph graph compiled successfully")
print(f"    Nodes: retrieve → grade → (generate | rewrite → ... | fallback)")
print(f"    Config: MAX_RETRIES={MAX_RETRIES}, TOP_K={TOP_K}")
```

### 4d. Visualize the graph structure (cell 23)

#### Line-by-line
- `from IPython.display import Image, display` — notebook-display helpers: `Image` wraps raw image bytes for rendering, `display` shows any renderable object inline in the cell output.
- `try: png_bytes = graph.get_graph().draw_mermaid_png()` — `graph.get_graph()` returns LangGraph's internal graph representation (nodes/edges), and `.draw_mermaid_png()` renders it via Mermaid (a diagram-as-code tool) into PNG bytes — this call typically requires either a local Mermaid renderer or a network round-trip to a hosted rendering service, which is why it's wrapped in a `try` block: it can legitimately fail in offline or sandboxed environments.
- `display(Image(png_bytes))` — shows the rendered diagram inline if the Mermaid call succeeded.
- `except Exception as mermaid_err:` — catches any failure from the Mermaid path (network error, missing renderer, etc.) broadly, since the failure modes aren't fully predictable across environments, and falls back to building an equivalent diagram manually.
- `import matplotlib.pyplot as plt`, `import matplotlib.patches as mpatches`, `import networkx as nx` — the fallback path's dependencies: `networkx` for representing and laying out a directed graph, `matplotlib.patches` for drawing shapes (used elsewhere in the notebook too, e.g. section 4f).
- `G = nx.DiGraph()` — creates an empty **directed** graph object (edges have direction, matching the actual control flow).
- `node_list = ["START", "retrieve", "grade", "rewrite", "generate", "fallback", "END"]` — the fixed node set, listed in a specific order that's reused below (`ordered_colors`) to keep node-to-color mapping consistent.
- `G.add_nodes_from(node_list)` — registers all seven nodes.
- `G.add_edges_from([...])` — registers the same edges as the actual compiled graph, but *manually* — this list is a hand-maintained mirror of the real edges in `builder`, not generated from `graph` itself, meaning it would need to be kept in sync by hand if the graph's structure ever changed. It lists all three conditional branches out of `"grade"` (to `generate`, `rewrite`, `fallback`) plus the loop-back edge (`rewrite → retrieve`) and both terminal edges (`generate → END`, `fallback → END`).
- `pos = {...}` — a hand-specified `(x, y)` coordinate for every node, rather than an automatic layout algorithm (like `nx.spring_layout`) — this gives full manual control over the diagram's shape, positioning `generate`, `rewrite`, and `fallback` as three siblings below `grade` to visually emphasize the three-way branch.
- `node_color_map = {...}` — a fixed hex color per node, matching the same color scheme used later in section 4f's trace visualization (`retrieve`=blue, `grade`=purple, `generate`=green, `rewrite`=orange, `fallback`=red) so a learner sees a consistent visual language across the notebook.
- `ordered_colors = [node_color_map[n] for n in node_list]` — converts the color map into a list ordered to match `node_list`, since `nx.draw_networkx_nodes` expects colors as a list aligned to its `nodelist` order rather than a dict.
- `edge_labels = {...}` — text labels for just the three conditional edges out of `"grade"` (`"relevant"`, `"not relevant\n+ retries left"`, `"not relevant\n+ retries exhausted"`), explaining *why* each branch is taken; the unconditional edges are left unlabeled since their meaning is self-evident.
- `fig, ax = plt.subplots(figsize=(8, 8))` — a square figure, appropriate for a compact flow diagram.
- `nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ordered_colors, node_size=2200, alpha=0.95)` — draws each node as a colored circle at its specified position; `alpha=0.95` gives a very slight transparency.
- `nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white", font_weight="bold")` — draws each node's name centered inside its circle in bold white text.
- `nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=22, edge_color="#555", width=1.8, connectionstyle="arc3,rad=0.08", min_source_margin=30, min_target_margin=30)` — draws directional arrows between nodes. `connectionstyle="arc3,rad=0.08"` curves the edges slightly rather than drawing them as straight lines, which helps visually separate edges that would otherwise overlap (e.g., the three edges radiating out of `"grade"`); `min_source_margin`/`min_target_margin` keep the arrows from overlapping the node circles at either end.
- `nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7.5, label_pos=0.45)` — places the three text labels defined above near their respective edges, positioned 45% of the way along each edge (`label_pos=0.45`) rather than exactly at the midpoint, likely to avoid overlapping the curved edge's midpoint crossing.
- `ax.set_title(...)`, `ax.axis("off")` — a bold title, and hiding the x/y axis ticks/frame entirely since this is a diagram, not a data plot.
- `legend_handles = [mpatches.Patch(color=..., label=...), ...]` — builds a manual legend explaining which library "owns" each colored node (e.g., `"retrieve ← LlamaIndex"`, `"grade ← LangChain"`), directly reinforcing the notebook's central three-libraries-three-layers teaching point from within the diagram itself.
- `ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5, framealpha=0.9)` — renders that legend.
- `plt.tight_layout()`, `plt.show()` — standard spacing cleanup and render.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  VISUALIZATION — LangGraph Structure
#  Renders the compiled graph as a flow diagram.
#  Tries the built-in Mermaid PNG renderer first;
#  falls back to a matplotlib/networkx diagram if unavailable.
# ─────────────────────────────────────────────────────────────

from IPython.display import Image, display

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    display(Image(png_bytes))
    print("✅  Graph rendered via LangGraph Mermaid renderer.")

except Exception as mermaid_err:
    print(f"Mermaid renderer unavailable ({mermaid_err}) — using matplotlib fallback.")

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    G = nx.DiGraph()
    node_list = ["START", "retrieve", "grade", "rewrite", "generate", "fallback", "END"]
    G.add_nodes_from(node_list)
    G.add_edges_from([
        ("START",    "retrieve"),
        ("retrieve", "grade"),
        ("grade",    "generate"),    # conditional: relevant
        ("grade",    "rewrite"),     # conditional: not_relevant + retries remain
        ("grade",    "fallback"),    # conditional: retries exhausted
        ("rewrite",  "retrieve"),    # loop back
        ("generate", "END"),
        ("fallback", "END"),
    ])

    pos = {
        "START":    (2.0, 6.0),
        "retrieve": (2.0, 4.8),
        "grade":    (2.0, 3.6),
        "generate": (0.3, 2.0),
        "rewrite":  (2.0, 2.0),
        "fallback": (3.7, 2.0),
        "END":      (2.0, 0.8),
    }
    node_color_map = {
        "START":    "#95a5a6",  "END":      "#95a5a6",
        "retrieve": "#3498db",  "grade":    "#9b59b6",
        "generate": "#2ecc71",  "rewrite":  "#f39c12",
        "fallback": "#e74c3c",
    }
    ordered_colors = [node_color_map[n] for n in node_list]

    edge_labels = {
        ("grade", "generate"): "relevant",
        ("grade", "rewrite"):  "not relevant\n+ retries left",
        ("grade", "fallback"): "not relevant\n+ retries exhausted",
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ordered_colors,
                           node_size=2200, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                            font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=22,
                           edge_color="#555", width=1.8,
                           connectionstyle="arc3,rad=0.08", min_source_margin=30,
                           min_target_margin=30)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=7.5, label_pos=0.45)

    ax.set_title("LangGraph Corrective-RAG Structure", fontsize=13, fontweight="bold")
    ax.axis("off")

    legend_handles = [
        mpatches.Patch(color="#3498db", label="retrieve  ← LlamaIndex"),
        mpatches.Patch(color="#9b59b6", label="grade     ← LangChain"),
        mpatches.Patch(color="#2ecc71", label="generate  ← LangChain"),
        mpatches.Patch(color="#f39c12", label="rewrite   ← LangChain"),
        mpatches.Patch(color="#e74c3c", label="fallback"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    plt.show()
```

### 4e. Run the assistant — easy and hard questions (cells 24–25)

#### Line-by-line
- `def run_assistant(question: str) -> dict:` — the full app packaged as one function: build initial state, invoke the graph, print the trace (the "teaching payoff"), pretty-print the answer, return the final state.
- `print(f"\n{'═' * 60}")`, `print(f"QUESTION: {question}")`, `print('═' * 60)` — a heavy double-line border (`═` repeated 60 times) framing the question, making each run's output visually distinct when several runs' outputs appear one after another in the notebook.
- `initial_state: GraphState = {...}` — constructs the complete starting state dict. The comment notes every field must be present — LangGraph validates the state schema, so omitting a required key here would raise an error at invocation time, not silently default it.
  - `"question": question` — stored once, never touched again by any node.
  - `"search_query": question` — starts identical to `question`; `node_rewrite` may later replace it.
  - `"context": []` — empty until `node_retrieve` runs for the first time.
  - `"grade": ""` and `"grade_reason": ""` — empty strings until `node_grade` runs.
  - `"answer": None` — until either `node_generate` or `node_fallback` sets it.
  - `"retry_count": 0` — starts at zero; only `node_rewrite` increments it.
  - `"node_path": []` — starts empty; each node appends its own name via the `add` reducer configured in `GraphState`.
- `final_state = graph.invoke(initial_state)` — runs the compiled graph to completion. This call blocks and internally loops through however many nodes it takes (potentially several `retrieve → grade → rewrite` cycles) until it reaches `END`, returning the fully accumulated final state dict.
- `path = " → ".join(final_state["node_path"])` — joins the accumulated node-name list into a single arrow-separated string.
- `print(f"\nNODE PATH: {path}")` — the comment calls this "THE teaching payoff": it's the one line that makes visible every routing decision the graph made during this run — whether it went straight to `generate` or looped through `rewrite` one or more times before reaching `fallback`.
- `ans = final_state["answer"]` — pulls out the final answer dict (set by whichever of `node_generate`/`node_fallback` actually ran).
- `print(f"\nANSWER: {ans['answer']}")`, `print(f"CONFIDENCE: {ans['confidence']}")` — plain dict-key access here (not attribute access) because by this point `answer` has already been converted to a plain dict (via `.model_dump()` in `node_generate`, or hand-built as a dict in `node_fallback`) — it's no longer a Pydantic object.
- `if ans["sources"]: print(f"SOURCES: {', '.join(ans['sources'])}")` — only prints the sources line if the list is non-empty, since a fallback answer always has an empty `sources` list and printing an empty line there would be noise.
- `return final_state` — hands back the entire final state dict (not just the answer) so a caller can inspect any field, e.g. `grade_reason` or `retry_count`, for deeper debugging.
- `result_easy = run_assistant("How many PTO days do new employees get?")` (cell 24) — the comment states the expected path explicitly: `retrieve → grade → generate` — the context should be found relevant on the first try, no rewriting needed.
- `result_hard = run_assistant("What is the company's policy on cryptocurrency payments?")` (cell 25) — the comment explains why this question is chosen: cryptocurrency payments are not mentioned anywhere in the handbook, so the grader should mark the retrieved context `not_relevant`, which triggers `node_rewrite`; after `MAX_RETRIES` (2) rewrite-and-retry cycles still turn up nothing relevant, `node_fallback` fires. The expected path is spelled out: `retrieve → grade → rewrite → retrieve → grade → fallback`.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  STAGE 2 — RUN THE ASSISTANT   (mirrors src/app.py)
#
#  run_assistant() is the full app in one function:
#    1. Build the initial state dict
#    2. Invoke the compiled graph
#    3. Print the node path  ← the teaching payoff
#    4. Pretty-print the structured answer
# ─────────────────────────────────────────────────────────────

def run_assistant(question: str) -> dict:   # accepts any question; returns the final graph state
    """Run the corrective-RAG graph and pretty-print results."""
    print(f"\n{'═' * 60}")
    print(f"QUESTION: {question}")
    print('═' * 60)

    initial_state: GraphState = {           # every field must be present — LangGraph validates the schema
        "question":     question,           # the user's question — stored here and never overwritten
        "search_query": question,           # starts as the raw question; node_rewrite may update this
        "context":      [],                 # empty until the first node_retrieve runs
        "grade":        "",                 # empty until the first node_grade runs
        "grade_reason": "",                 # empty until the first node_grade runs
        "answer":       None,               # None until node_generate or node_fallback runs
        "retry_count":  0,                  # starts at 0; incremented by node_rewrite each loop
        "node_path":    [],                 # starts empty; each node appends its own name via the add reducer
    }

    final_state = graph.invoke(initial_state)  # run the graph to completion; blocks until END is reached

    path = " → ".join(final_state["node_path"])  # join the execution trace list into a readable string
    print(f"\nNODE PATH:  {path}")          # THE teaching payoff: shows every decision the graph made

    ans = final_state["answer"]             # retrieve the final answer dict from the completed state
    print(f"\nANSWER:     {ans['answer']}")  # the text answer
    print(f"CONFIDENCE: {ans['confidence']}")  # "high", "medium", or "low"
    if ans["sources"]:                      # only print sources if the model cited any
        print(f"SOURCES:    {', '.join(ans['sources'])}")

    return final_state                      # return full state so the caller can inspect any field


# ── Easy question — expected: retrieve → grade → generate ────────────────────
result_easy = run_assistant("How many PTO days do new employees get?")
```
```python
# ── Unanswerable question — expected: retrieve → grade → rewrite → retrieve → grade → fallback
# Cryptocurrency payments are not mentioned anywhere in the handbook.
# The grader marks the context as not_relevant → triggers node_rewrite.
# After MAX_RETRIES exhausted with no relevant context → node_fallback fires.

result_hard = run_assistant("What is the company's policy on cryptocurrency payments?")
```

### 4f. Visualize execution trace comparison (cell 26)

#### Line-by-line
- `import matplotlib.pyplot as plt` and `import matplotlib.patches as mpatches` — re-imports (cheap, cached) for the box-and-arrow diagram this cell builds.
- `NODE_COLORS = {...}` — the same five node→hex-color mapping used in section 4d's fallback diagram, kept consistent across the notebook.
- `def draw_trace(ax, path, title, question):` — a reusable helper that draws one vertical stack of labeled, colored boxes (one per node in `path`) connected by downward arrows, onto whichever `ax` (subplot axis) is passed in — called twice below, once per question.
- `ax.set_xlim(0, 1)` and `ax.set_ylim(-0.6, len(path) - 0.4)` — fixes the coordinate system: a narrow horizontal range (boxes are centered, not wide) and a vertical range sized to exactly fit however many nodes are in this particular `path`, with small margins top and bottom.
- `ax.axis("off")` — hides tick marks/axis lines since this is a diagram, not a data plot.
- `ax.set_title(f'{title}\n"{question}"', fontsize=9, fontweight="bold", pad=8, wrap=True)` — a two-line title: the run's label (e.g., "✅ Answerable") above the actual question text in quotes; `wrap=True` lets long question text wrap onto additional lines rather than overflowing the subplot width.
- `for i, node in enumerate(path):` — iterates the node names in order.
  - `y = len(path) - 1 - i` — computes a y-coordinate that places the *first* node in the path at the *top* of the stack (since matplotlib's y-axis increases upward by default, and index `0` should visually appear highest) — this inverted arithmetic is what makes the diagram read top-to-bottom in execution order.
  - `color = NODE_COLORS.get(node, "#95a5a6")` — looks up this node's color, defaulting to gray (`#95a5a6`) for any name not in the map (a defensive fallback, though every actual node name is covered).
  - `box = mpatches.FancyBboxPatch((0.15, y - 0.32), 0.70, 0.64, boxstyle="round,pad=0.04", facecolor=color, edgecolor="white", linewidth=1.5, zorder=2)` — draws a rounded-corner rectangle centered horizontally (`x` from 0.15 to 0.85, i.e. centered in the 0–1 range) at this node's vertical slot; `zorder=2` ensures boxes draw above the (lower-zorder) arrows drawn between them.
  - `ax.add_patch(box)` — actually adds the shape to the axis (creating a `Patch` object alone doesn't render it).
  - `ax.text(0.50, y, node, ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=3)` — the node's name, centered inside its box, drawn above the box (`zorder=3`) so text is never obscured.
  - `if i < len(path) - 1: ax.annotate("", xy=(0.50, y - 0.32), xytext=(0.50, y - 0.68), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6), zorder=1)` — draws a downward arrow from just below the current box to just above the next box, but only if this isn't the last node in the path (no arrow needed after the final box). `ax.annotate("", ...)` with an empty string is a common matplotlib idiom purely for drawing an arrow without any accompanying text.
- `easy_path = result_easy["node_path"]` and `hard_path = result_hard["node_path"]` — pulls the two execution traces captured back in section 4e.
- `max_len = max(len(easy_path), len(hard_path))` — the longer of the two paths (the hard/unanswerable one, since it loops through `rewrite`), used to size the figure so both subplots share a comparable vertical scale.
- `fig, axes = plt.subplots(1, 2, figsize=(10, max_len * 1.15 + 1.5))` — two side-by-side subplots, with height driven by `max_len` so the taller trace isn't cramped.
- `fig.suptitle("LangGraph Execution Trace Comparison", ...)` — overall figure title.
- `draw_trace(axes[0], easy_path, "✅  Answerable", "How many PTO days do new employees get?")` and the equivalent call for `axes[1]` with `hard_path` and the crypto question — invokes the helper once per subplot, producing a direct visual side-by-side of the short (no-loop) path versus the longer (one-loop) path.
- `legend_handles = [mpatches.Patch(color=c, label=n) for n, c in NODE_COLORS.items()]` — builds one legend swatch per node color, generated directly from the `NODE_COLORS` dict rather than hand-listed, so it can never drift out of sync with the actual colors used in the boxes.
- `fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))` — a single shared legend for the whole figure (`fig.legend`, not `ax.legend`, since it should apply to both subplots at once), arranged in 5 columns (`ncol=5`, one per node color) and pinned near the bottom center via `bbox_to_anchor`.
- `plt.tight_layout(rect=[0, 0.07, 1, 1])` — the usual automatic spacing pass, but constrained to leave the bottom 7% of the figure (`rect=[0, 0.07, 1, 1]`) free for the legend placed just below the subplots, so `tight_layout`'s automatic adjustment doesn't fight with the manually positioned legend.
- `plt.show()` — renders the figure.
- The two trailing `print(f"...")` lines — restate both paths as plain text (with node counts) beneath the chart, giving a copy-pasteable/searchable text record alongside the visual.

#### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  VISUALIZATION — Execution Trace Comparison
#  Side-by-side node paths from the two runs above.
#  The color-coded boxes make every routing decision visible.
# ─────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NODE_COLORS = {
    "retrieve": "#3498db",
    "grade":    "#9b59b6",
    "rewrite":  "#f39c12",
    "generate": "#2ecc71",
    "fallback": "#e74c3c",
}


def draw_trace(ax, path, title, question):
    """Draw a vertical stack of labeled boxes with connecting arrows."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(path) - 0.4)
    ax.axis("off")
    ax.set_title(f"{title}\n\"{question}\"", fontsize=9, fontweight="bold", pad=8,
                 wrap=True)

    for i, node in enumerate(path):
        y = len(path) - 1 - i
        color = NODE_COLORS.get(node, "#95a5a6")
        box = mpatches.FancyBboxPatch(
            (0.15, y - 0.32), 0.70, 0.64,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white", linewidth=1.5, zorder=2,
        )
        ax.add_patch(box)
        ax.text(0.50, y, node, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=3)
        if i < len(path) - 1:
            ax.annotate(
                "", xy=(0.50, y - 0.32), xytext=(0.50, y - 0.68),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6),
                zorder=1,
            )


easy_path = result_easy["node_path"]
hard_path = result_hard["node_path"]
max_len   = max(len(easy_path), len(hard_path))

fig, axes = plt.subplots(1, 2, figsize=(10, max_len * 1.15 + 1.5))
fig.suptitle("LangGraph Execution Trace Comparison", fontsize=13, fontweight="bold")

draw_trace(axes[0], easy_path,
           "✅  Answerable", "How many PTO days do new employees get?")
draw_trace(axes[1], hard_path,
           "❌  Unanswerable", "What is the company's policy on cryptocurrency payments?")

legend_handles = [mpatches.Patch(color=c, label=n) for n, c in NODE_COLORS.items()]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()

print(f"\nEasy path  ({len(easy_path)} nodes): {' → '.join(easy_path)}")
print(f"Hard path  ({len(hard_path)} nodes): {' → '.join(hard_path)}")
```

---

## 5. Final cell — Gradio UI (cell 29)

### Why this section exists
The notebook's markdown (cell 28) frames this cell's three jobs directly: rebuild the index (so any documents dropped into `data/` since the last run are picked up automatically), wrap the compiled LangGraph graph so its outputs become plain return values instead of `print()` side effects (which Gradio can't display), and launch an interactive question box wired to that wrapper. It depends on every earlier cell having already run, since it reuses `ingest()`, `graph`, `GraphState`, `DATA_DIR`, and `STORAGE_DIR` directly.

### Line-by-line
- `import shutil` and `import gradio as gr` — `shutil` provides `rmtree` for recursively deleting the old Chroma storage directory; `gradio` (aliased `gr`, the ecosystem-standard convention) builds the web UI.
- `print(f"🔄  Rebuilding index from {DATA_DIR}/ ...")` — a visible status line before the (potentially slow) re-embedding work begins.
- `if STORAGE_DIR.exists(): shutil.rmtree(STORAGE_DIR)` — removes the old Chroma database directory entirely if present, guaranteeing the subsequent `ingest()` call starts from a genuinely clean slate rather than potentially mixing old and new vectors. (In practice, since `STORAGE_DIR` is unused by the in-memory `EphemeralClient` path used by `ingest()`/`load_index()` earlier in the notebook, this line is defensive cleanup for any on-disk artifacts rather than something `ingest()` itself reads from.)
- `index = ingest()` — re-runs the full load → chunk → embed → store pipeline from Section 2a, picking up any `.md` files currently in `data/` — including ones a learner may have added mid-session.
- `doc_count = len(list(DATA_DIR.glob("*.md")))` — counts markdown files currently in `data/` for the confirmation message; `glob()` returns a generator, so it's wrapped in `list()` before `len()` can be called on it.
- `print(f"✅  {doc_count} document(s) indexed and ready.\n")` — confirms the rebuild completed, with a trailing blank line for visual separation before the UI launches.
- `def query_rag(question: str):` — the callback function Gradio will invoke every time the user submits a question; its signature (one string argument in, a tuple of display strings out) matches what the `gr.Blocks` wiring below expects.
- `question = question.strip()` — removes leading/trailing whitespace the user might have typed or pasted.
- `if not question: return "Please enter a question.", "", "", ""` — guards against an empty submission; returns a friendly prompt-message plus empty strings for the other three output fields (confidence, sources, path) rather than running the graph on empty input.
- `final_state = graph.invoke({...})` — builds and runs the same initial-state shape used by `run_assistant()` in section 4e, but inline here rather than calling that helper directly — likely because `run_assistant()` is designed to `print()` its results (not appropriate for a UI callback), whereas this inline version only extracts return values.
- `ans = final_state["answer"]` — the completed answer dict, exactly as in `run_assistant()`.
- `node_path = " → ".join(final_state["node_path"])` — the arrow-joined trace string, same pattern as before, displayed in its own UI panel this time instead of printed.
- `answer = ans["answer"]` — the answer text.
- `confidence = ans["confidence"].upper()` — upper-cases the confidence string (`"high"` → `"HIGH"`) purely for visual emphasis in the UI's small info chip.
- `sources = ", ".join(ans["sources"]) if ans.get("sources") else "—"` — joins multiple source names into one comma-separated string if any exist, or displays a plain em-dash placeholder (`"—"`) if the list is empty (e.g., a fallback answer) — avoids showing an empty/blank field in the UI.
- `return answer, confidence, sources, node_path` — returns a 4-tuple; Gradio maps each element to one of the four output components declared in the `outputs=[...]` list further down.
- `with gr.Blocks(title="Doc Assistant", theme=gr.themes.Soft()) as demo:` — `gr.Blocks` is Gradio's low-level, fully custom layout API (as opposed to the simpler `gr.Interface` shortcut), used here because the UI needs a specific multi-row layout with example questions. `theme=gr.themes.Soft()` applies a pre-built visual theme; `title=` sets the browser tab title. The `with ... as demo:` block captures every component defined inside it into the `demo` app object.
- `gr.Markdown("## 📚 Doc Assistant\n...")` — a static, non-interactive block of rendered markdown at the top of the page, explaining what the app does and that it self-corrects and can say "I don't know."
- `with gr.Row(): question_box = gr.Textbox(...); submit_btn = gr.Button(...)` — a horizontal layout row containing the question input and the submit button side by side.
  - `gr.Textbox(label="Your question", placeholder="e.g. What are the two phases of a RAG pipeline?", lines=2, scale=5)` — a multi-line (2-row) text input with placeholder example text; `scale=5` gives it 5 parts of the row's available width relative to the button's `scale=1`, so the textbox is visually much wider than the button.
  - `gr.Button("Ask ↵", variant="primary", scale=1, min_width=80)` — `variant="primary"` gives it the theme's accent color (visually signaling it's the main action); `min_width=80` prevents it from shrinking below a usable size even in a narrow layout.
- `answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)` — a larger, read-only (`interactive=False`) output box sized for a full paragraph answer.
- `with gr.Row(): confidence_box = ...; sources_box = ...; path_box = ...` — a second row holding three more read-only output boxes, each with a different `scale` (1, 2, 3 respectively) so `path_box` (which tends to hold the longest text, e.g. a full node-path trace) gets proportionally the most horizontal space.
- `gr.Examples(label="Example questions — click to load", examples=[[...], ...], inputs=question_box)` — a clickable list of nine pre-written example questions; clicking one auto-fills `question_box` (specified via `inputs=question_box`) without submitting it. The example list deliberately spans both corpora in `data/`: handbook questions (PTO, stipend, expenses, code of conduct) and bootcamp curriculum questions (RAG phases, transformer attention, vector index vs. relational database), plus the intentionally unanswerable cryptocurrency question — giving a new user a guided tour of both the "it works" and "it honestly says no" behaviors without having to think of their own test questions.
- `for trigger in (submit_btn.click, question_box.submit): trigger(fn=query_rag, inputs=question_box, outputs=[answer_box, confidence_box, sources_box, path_box])` — wires the *same* callback to two different Gradio events using a loop rather than duplicating the `.click(...)`/`.submit(...)` call twice: clicking the button (`submit_btn.click`) and pressing Enter inside the textbox (`question_box.submit`) both trigger `query_rag`. `inputs=question_box` passes the textbox's current value as `query_rag`'s single argument; `outputs=[...]` maps the four returned tuple values, in order, to `answer_box`, `confidence_box`, `sources_box`, and `path_box` respectively.
- `demo.launch(share=False)` — starts Gradio's local web server and opens/embeds the UI. `share=False` is the explicit (and also the default) choice not to generate a public internet-facing tunnel URL — the comment notes `share=True` would do that instead, useful context for anyone who might want to demo this to someone outside their machine.

### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  FINAL CELL — Gradio UI
#
#  Run this after all cells above have been executed.
#  It rebuilds the index from data/ (so any new .md files you
#  dropped in are automatically picked up), then opens an
#  interactive question-answering interface in the notebook.
# ─────────────────────────────────────────────────────────────

import shutil
import gradio as gr

# ── 1. Rebuild the vector index from the current data/ folder ─
#       This wipes stale vectors and re-embeds every .md file,
#       so newly added documents are always included.
print(f"🔄  Rebuilding index from {DATA_DIR}/ ...")
if STORAGE_DIR.exists():
    shutil.rmtree(STORAGE_DIR)          # remove old Chroma database
index = ingest()                         # load → chunk → embed → persist
doc_count = len(list(DATA_DIR.glob("*.md")))
print(f"✅  {doc_count} document(s) indexed and ready.\n")


# ── 2. Query wrapper ──────────────────────────────────────────
#       graph.invoke() writes to `print()` inside each node.
#       This wrapper captures the structured fields Gradio needs.
def query_rag(question: str):
    """Run the corrective-RAG graph; return displayable strings."""
    question = question.strip()
    if not question:
        return "Please enter a question.", "", "", ""

    final_state = graph.invoke({            # run the full LangGraph pipeline
        "question":     question,
        "search_query": question,           # may be rewritten by node_rewrite
        "context":      [],
        "grade":        "",
        "grade_reason": "",
        "answer":       None,
        "retry_count":  0,
        "node_path":    [],                 # accumulated by the add reducer
    })

    ans        = final_state["answer"]
    node_path  = " → ".join(final_state["node_path"])
    answer     = ans["answer"]
    confidence = ans["confidence"].upper()  # HIGH / MEDIUM / LOW
    sources    = ", ".join(ans["sources"]) if ans.get("sources") else "—"

    return answer, confidence, sources, node_path


# ── 3. Build the Gradio interface ─────────────────────────────
with gr.Blocks(title="Doc Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "## 📚 Doc Assistant\n"
        "Ask anything about the **employee handbook** or the **bootcamp curriculum**.\n\n"
        "The corrective-RAG graph retrieves, grades context quality, "
        "rewrites the query if needed, and honestly says *I don't know* "
        "when nothing relevant is found."
    )

    # ── Input row ─────────────────────────────────────────────
    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What are the two phases of a RAG pipeline?",
            lines=2,
            scale=5,
        )
        submit_btn = gr.Button("Ask ↵", variant="primary", scale=1, min_width=80)

    # ── Answer panel ──────────────────────────────────────────
    answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)

    # ── Metadata row ──────────────────────────────────────────
    with gr.Row():
        confidence_box = gr.Textbox(
            label="Confidence", interactive=False, scale=1
        )
        sources_box = gr.Textbox(
            label="Sources",    interactive=False, scale=2
        )
        path_box = gr.Textbox(
            label="Graph path (LangGraph trace)", interactive=False, scale=3
        )

    # ── Example questions covering both doc sets ──────────────
    gr.Examples(
        label="Example questions — click to load",
        examples=[
            ["How many PTO days do new employees get?"],
            ["What is the monthly home-office stipend for remote workers?"],
            ["How do I submit an expense reimbursement?"],
            ["How do I report a code of conduct violation?"],
            ["What is RAG and why does it reduce hallucination?"],
            ["What are the two phases of a RAG pipeline?"],
            ["What does a transformer's attention mechanism do?"],
            ["What is the difference between a vector index and a relational database?"],
            ["What is the company's policy on cryptocurrency payments?"],
        ],
        inputs=question_box,
    )

    # ── Wire both the button click and Enter key ──────────────
    for trigger in (submit_btn.click, question_box.submit):
        trigger(
            fn=query_rag,
            inputs=question_box,
            outputs=[answer_box, confidence_box, sources_box, path_box],
        )

demo.launch(share=False)                    # share=True generates a public URL via Gradio tunnel
```

---

## 6. Shutdown / cleanup (cell 30)

### Why this section exists
Long-lived local resources (the Ollama server process, the in-memory Chroma index, the Jupyter kernel itself) should be releasable on demand once a learner is done with the notebook. This cell provides that, but with every line commented out by default so the cell is completely inert if run accidentally — a learner opts in explicitly by uncommenting only the section they need.

### Line-by-line
- The header comment block — explains the cell's "safe by default" design directly: everything below is commented out so running it does nothing until the user opts in.
- `# import subprocess` / `# result = subprocess.run(["pkill", "-f", "ollama"], capture_output=True)` / `# print(...)` — if uncommented, this would run `pkill -f ollama` as a subprocess to terminate any running Ollama server process matching that name pattern; `capture_output=True` captures stdout/stderr instead of letting them print directly, so the cell can inspect `result.returncode` to decide which message to print (`"Ollama stopped."` if the kill succeeded, i.e. return code 0, or `"Ollama was not running."` otherwise).
- `# chroma_client = None` / `# index = None` / `# print(...)` — if uncommented, this drops the local references to the Chroma client singleton and the built index; note this reassigns a *new local name* `chroma_client` rather than the module-level `_chroma_client` used elsewhere in the notebook (a minor naming mismatch in the commented-out code — since it's inert, it doesn't affect the notebook's behavior, but a learner who uncomments it and expects it to release `_chroma_client` specifically should be aware only local `chroma_client`/`index` names are cleared, not the underscore-prefixed module singleton). Dropping the references allows Python's garbage collector to reclaim the memory once nothing else references the same objects.
- `# import IPython` / `# IPython.Application.instance().kernel.do_shutdown(restart=True)` — if uncommented, this would command the running Jupyter kernel to shut down and immediately restart itself, clearing every variable in the notebook's namespace — the most thorough reset available, short of manually restarting from the Jupyter UI.
- `# print("Shutdown cell loaded — uncomment the sections above to use them.")` — even this final print is commented out, consistent with the cell's "does nothing unless you opt in" design; running the cell as-shipped produces no output at all.

### Full code block
```python
# ─────────────────────────────────────────────────────────────
#  SHUTDOWN / CLEANUP
#
#  Uncomment and run the sections you need when you are done.
#  Everything below is commented out so this cell is safe to
#  run accidentally — it will do nothing until you opt in.
# ─────────────────────────────────────────────────────────────

# ── Stop the Ollama server ───────────────────────────────────
# import subprocess
# result = subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
# print("Ollama stopped." if result.returncode == 0 else "Ollama was not running.")

# # # ── Release the in-memory Chroma index ──────────────────────
# chroma_client = None          # drop the singleton so Python can GC the memory
# index = None                   # drop the VectorStoreIndex reference too
# print("Chroma in-memory index released.")

# # # ── Restart the kernel (clears all variables) ────────────────
# import IPython
# IPython.Application.instance().kernel.do_shutdown(restart=True)

# print("Shutdown cell loaded — uncomment the sections above to use them.")
```
