# Module 9B — Entity Linking Demo: Line-by-Line Walkthrough

This document explains the code in `module_9b_entity_linking_demo.ipynb` one section at a time. Each section matches a numbered section in the notebook: first a plain explanation, then the full code block you can copy and paste.

---

## Section 1 — Setup & Orientation

### Explanation

**Imports**

```python
import spacy
from neo4j import GraphDatabase
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List
```

We load everything up front. `spacy` runs the NLP work. `GraphDatabase` is the official Neo4j driver — it opens and manages our database connections. `dataclass` and `field` let us build a clean result object later without writing an `__init__` by hand. The `typing` imports just label what types our functions take and return.

---

**Loading the spaCy model**

```python
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded:", nlp.meta["name"])
```

`spacy.load` loads a pre-trained English model into memory. `en_core_web_sm` is spaCy's small English pipeline. It can tokenize text, tag parts of speech, parse grammar, and — the part we care about — recognize named entities. The `print` just confirms the right model loaded before we run anything real.

---

**Connecting to Neo4j**

```python
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
driver.verify_connectivity()
print("Neo4j connection established.")
```

`GraphDatabase.driver` sets up a pool of reusable connections rather than a single one, so many queries can run efficiently. `7687` is Neo4j's default port. `verify_connectivity()` does a quick check that the server is up and the password is correct. We run it now, on purpose: if the database is down, we want to find out immediately — not after processing 150 documents.

---

**Inspecting the schema**

```python
with driver.session() as session:
    labels = session.run("CALL db.labels() YIELD label RETURN label").value()
    print("KG labels:", labels)

    rel_types = session.run(
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
    ).value()
    print("KG relationships:", rel_types)
```

`driver.session()` grabs a connection from the pool and hands it back automatically when the `with` block ends. `CALL db.labels()` is a built-in Neo4j command that lists every node label in the graph; we do the same for relationship types. `.value()` turns the result into a plain Python list. The check matters because if `SUBCLASS_OF` is missing here, the hierarchy step in Section 5 will quietly return nothing — easier to catch now than to debug later.

---

**Previewing the four Cypher beats**

```python
BEAT_PREVIEWS = [
    "MATCH (n:Entity) WHERE toLower(n.name) = toLower($surface) RETURN n.id, n.name, labels(n)",
    "# NER label → KG type dict  →  filter candidates by allowed_types",
    "MATCH (c:Entity {id: $cand_id})-[:SUBCLASS_OF*0..]->(ancestor:Entity) RETURN collect(labels(ancestor))",
    "# Return LinkResult(..., predicted_node_id=None, reason='nil-no-candidates')",
]
for i, beat in enumerate(BEAT_PREVIEWS, 1):
    print(f"Beat {i}: {beat}")
```

These strings are just a preview — nothing here runs. They give a quick roadmap of the four steps before we build them. `enumerate(BEAT_PREVIEWS, 1)` starts the count at 1, so the output reads "Beat 1" through "Beat 4" instead of starting at 0.

---

### Full Section 1 Code Block

```python
import spacy                                                                          # NLP library — tokenization, POS tagging, named entity recognition
from neo4j import GraphDatabase                                                       # Official Neo4j Python driver for opening and managing database connections
from dataclasses import dataclass, field                                              # Eliminates boilerplate __init__ and __repr__ for result container classes
from typing import Optional, Set, Dict, List                                          # Type hint helpers used in all function signatures below

nlp = spacy.load("en_core_web_sm")                                                   # Loads the small English NLP pipeline (tokenizer + NER) into memory
print("spaCy model loaded:", nlp.meta["name"])                                       # Confirms the correct model name is active before any real processing runs

NEO4J_URI  = "bolt://localhost:7687"                                                  # Bolt protocol endpoint; 7687 is Neo4j's default port
NEO4J_USER = "neo4j"                                                                  # Default Neo4j username
NEO4J_PASS = "password"                                                               # Database password

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))              # Creates a reusable connection pool to the Neo4j instance
driver.verify_connectivity()                                                          # Immediately checks the server is up and credentials are correct — fail fast
print("Neo4j connection established.")                                                # Confirms the database is reachable before any real queries run

with driver.session() as session:                                                     # Borrows a connection from the pool; returns it automatically when the block exits
    labels = session.run("CALL db.labels() YIELD label RETURN label").value()        # Lists every node label in the graph; .value() converts result to a plain list
    print("KG labels:", labels)                                                       # Displays available labels so we can verify the schema is populated

    rel_types = session.run(                                                          # Opens a second schema-inspection query
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"  # Built-in Neo4j procedure that returns all edge types present in the graph
    ).value()                                                                         # Converts the cursor result to a plain Python list
    print("KG relationships:", rel_types)                                             # Lets us catch a missing SUBCLASS_OF edge before it silently breaks Section 5

BEAT_PREVIEWS = [                                                                     # Read-only string snapshots of the four linking steps — nothing executes yet
    "MATCH (n:Entity) WHERE toLower(n.name) = toLower($surface) RETURN n.id, n.name, labels(n)",  # Beat 1: case-insensitive candidate lookup by name
    "# NER label → KG type dict  →  filter candidates by allowed_types",             # Beat 2: synopsis of the type-filter step
    "MATCH (c:Entity {id: $cand_id})-[:SUBCLASS_OF*0..]->(ancestor:Entity) RETURN collect(labels(ancestor))",  # Beat 3: variable-length hierarchy traversal
    "# Return LinkResult(..., predicted_node_id=None, reason='nil-no-candidates')",  # Beat 4: NIL fallback synopsis
]
for i, beat in enumerate(BEAT_PREVIEWS, 1):                                          # enumerate(..., 1) starts the counter at 1 so output reads "Beat 1"–"Beat 4"
    print(f"Beat {i}: {beat}")                                                        # Prints each step's number alongside its preview string
```

---

## Section 2 — spaCy NER on a Headline

### Explanation

**The two headlines**

```python
headlines = [
    "Paris hosts the climate summit",
    "Apple unveils new device",
]
```

We use two headlines on purpose. In the first, "Paris" could be a city or a person. In the second, "Apple" could be a company or a fruit. Keeping it to two lets us follow every decision the linker makes without getting lost in a big batch.

---

**Running the NER pipeline**

```python
for headline in headlines:
    doc = nlp(headline)
    print(f"\nHeadline: {headline!r}")
    print(f"  Tokens : {[t.text for t in doc]}")
```

`nlp(text)` runs the full spaCy pipeline and returns a `Doc` object, which holds everything spaCy found — tokens, entities, and so on. `[t.text for t in doc]` pulls out each token's text, so we can see how the headline was split into words.

---

**Inspecting the entity spans**

```python
if doc.ents:
    for ent in doc.ents:
        print(
            f"  Entity : {ent.text!r:20s} "
            f"label={ent.label_:10s} "
            f"char=[{ent.start_char}:{ent.end_char}]"
        )
else:
    print("  (no entities detected)")
```

`doc.ents` is the list of entities spaCy found. For each one we look at three things:

- `.text` — the words as they appear (e.g., `"Paris"`)
- `.label_` — the category, like `"GPE"` (a place) or `"ORG"` (an organization)
- `.start_char` / `.end_char` — where it sits in the original string

The `else` branch covers headlines with no entities. The `:20s` and `:10s` just line the columns up neatly.

**Why the label matters:** in "Apple unveils new device," the label `ORG` already rules out the fruit, since the fruit wouldn't get an entity tag at all. So NER does some of the disambiguation for free; the type filter in Section 4 handles the rest.

---

**The NIL concept**

Sometimes the right answer is "no match." If the headline were "Paris Hilton hosts the climate summit," spaCy would tag "Paris Hilton" as a `PERSON`. Our graph only has places, no people. The correct move is to return nothing (`predicted_node_id=None`, called NIL) rather than force a wrong match.

---

### Full Section 2 Code Block

```python
headlines = [                                                                         # Two ambiguous headlines chosen to illustrate the disambiguation problem
    "Paris hosts the climate summit",                                                 # "Paris" could be a city or a person — GPE label will guide the linker
    "Apple unveils new device",                                                       # "Apple" could be a company or a fruit — ORG label rules out the fruit
]

for headline in headlines:                                                            # Process each headline through the full NLP + entity inspection loop
    doc = nlp(headline)                                                               # Runs the spaCy pipeline; returns a Doc holding tokens, entities, and more
    print(f"\nHeadline: {headline!r}")                                               # Prints the raw headline text with surrounding quotes for clarity
    print(f"  Tokens : {[t.text for t in doc]}")                                     # Shows how spaCy split the headline into individual word tokens

    if doc.ents:                                                                      # Checks whether spaCy found any named entities in this headline
        for ent in doc.ents:                                                          # Iterates over each recognized entity span in the document
            print(
                f"  Entity : {ent.text!r:20s} "                                      # The entity text, quoted and padded to 20 chars for aligned columns
                f"label={ent.label_:10s} "                                           # The NER category (e.g., GPE, ORG), padded to 10 chars
                f"char=[{ent.start_char}:{ent.end_char}]"                            # Character offsets showing where the entity sits in the original string
            )
    else:
        print("  (no entities detected)")                                             # Fallback message when the NER pipeline finds nothing in the headline

print("\n--- Disambiguation note ---")                                               # Section separator before the explanatory notes
print("'Paris' → GPE: could be the city (in our KG) or a celebrity (not in KG).")  # Illustrates why the NER label alone is not always enough to resolve
print("'Apple' → ORG: the NER label rules out the fruit; company still needs KG lookup.")  # Shows how ORG narrows candidates before any database query
print("The linker must pick the single best KG node, or return NIL.")               # States the final goal: one answer, or an explicit abstention
```

---

## Section 3 — Candidates via Parameterized Cypher

### Explanation

**The candidate query**

```python
CANDIDATE_QUERY = """
MATCH (n:Entity)
WHERE toLower(n.name) = toLower($surface)
RETURN n.id AS id, n.name AS name, labels(n) AS kg_labels
"""
```

This query does three things:

1. `MATCH (n:Entity)` — only look at nodes labeled `:Entity`, so Neo4j doesn't scan the whole graph.
2. `WHERE toLower(n.name) = toLower($surface)` — match the name, ignoring case. `$surface` is a placeholder that Neo4j fills in safely at run time, which prevents injection attacks.
3. `RETURN ... labels(n) AS kg_labels` — return each match's id, name, and full list of labels. A node can have more than one label (like `["Entity", "City"]`), so we keep the whole list.

---

**The candidate retrieval function**

```python
def get_candidates(session, surface: str) -> List[Dict]:
    result = session.run(CANDIDATE_QUERY, surface=surface)
    return [dict(record) for record in result]
```

Wrapping the query in a function keeps the query string in one place instead of scattered everywhere. Passing `surface=surface` fills the `$surface` placeholder safely. The list comprehension pulls every row and turns each one into a plain dictionary, so callers never have to deal with Neo4j's own types.

---

**The parameterization requirement**

```python
# WRONG — f-string injection risk:
# bad_query = f"MATCH (n:Entity) WHERE n.name = '{surface}' RETURN n"
# session.run(bad_query)

# RIGHT — always pass values as keyword arguments to session.run():
# session.run(CANDIDATE_QUERY, surface=surface)
```

If you build the query with an f-string and `surface` were something like `"' OR 1=1 //"`, the query would match every node in the graph. The autograder checks your code's structure (not just whether it runs) and flags any `session.run` where the query is built with an f-string or `.format()`. So passing values as parameters isn't optional here — it's required.

---

### Full Section 3 Code Block

```python
CANDIDATE_QUERY = """
MATCH (n:Entity)
WHERE toLower(n.name) = toLower($surface)
RETURN n.id AS id, n.name AS name, labels(n) AS kg_labels
"""  # Cypher constant: matches nodes by name case-insensitively, returns id, name, and all node labels

def get_candidates(session, surface: str) -> List[Dict]:                             # Fetches every KG node whose name matches the given surface form
    result = session.run(CANDIDATE_QUERY, surface=surface)                           # Passes surface as a safe parameter — never interpolated into the query string
    return [dict(record) for record in result]                                       # Converts each Neo4j Record into a plain dict the caller can use freely

with driver.session() as session:                                                     # Borrows a connection from the pool; returned automatically when the block exits
    candidates = get_candidates(session, "Paris")                                    # Runs the lookup for the surface form "Paris"

print(f"Candidates for 'Paris' ({len(candidates)} found):")                          # Shows how many KG nodes matched before any type filtering
for c in candidates:                                                                  # Iterates over each candidate dictionary
    print(f"  {c}")                                                                   # Prints the candidate's id, name, and kg_labels fields

print("\nRule: session.run(query, param=value) — never f-strings in Cypher.")       # Reminds that parameterized queries prevent Cypher injection attacks
```

---

## Section 4 — NER → KG Type-Filter Disambiguation

### Explanation

**The `LinkResult` dataclass**

```python
@dataclass
class LinkResult:
    surface: str
    ner_label: str
    predicted_node_id: Optional[str]
    reason: str
    candidates: List[Dict] = field(default_factory=list)
```

A `@dataclass` writes the boilerplate (`__init__`, `__repr__`, `__eq__`) for us. A few notes on the fields:

- `predicted_node_id: Optional[str]` — either a node id or `None`. `Optional` makes the NIL case clear in the signature.
- `reason: str` — a fixed label like `"resolved-by-type"`. The autograder and diagnostics search for these exact strings, so a typo like `"resolved_by_type"` would make them count zero.
- `candidates: List[Dict] = field(default_factory=list)` — you can't write `= []` here, because all instances would then share one list. `field(default_factory=list)` gives each instance a fresh empty list.

---

**The NER/KG bridge dictionary**

```python
NER_LABEL_TO_KG_TYPE: Dict[str, Set[str]] = {
    "GPE":    {"City", "Country", "Region"},
    "ORG":    {"Organization", "Company"},
    "PERSON": {"Person"},
    "LOC":    {"Continent", "Ocean", "Mountain"},
    "PRODUCT": {"Product"},
}
```

This dictionary connects spaCy's entity labels to the graph's node labels. The values are sets because we'll check membership with set intersection. Keeping it as one top-level constant means you can support recipe types later (like `"CUISINE"` or `"INGREDIENT"`) by editing this one dictionary — no function changes needed.

---

**The type filter**

```python
def type_filter(candidates: List[Dict], ner_label: str) -> List[Dict]:
    allowed = NER_LABEL_TO_KG_TYPE.get(ner_label, set())
    return [c for c in candidates if set(c["kg_labels"]) & allowed]
```

`.get(ner_label, set())` returns an empty set for any label we haven't mapped. With an empty `allowed`, nothing matches, so the result is no candidates — which correctly leads to NIL instead of a random guess. The `&` is set intersection: if a candidate's labels overlap with `allowed`, it survives.

---

**Stage-1 disambiguation and reason tokens**

```python
if len(filtered) == 1:
    return LinkResult(..., reason="resolved-by-type", ...)
return LinkResult(..., reason="needs-hierarchy" if filtered else "nil-no-candidates", ...)
```

Three outcomes after filtering:

- **Exactly one left** → `"resolved-by-type"`, done.
- **More than one** → `"needs-hierarchy"`, move on to Section 5.
- **None left** → `"nil-no-candidates"`, no point continuing.

---

### Full Section 4 Code Block

```python
@dataclass                                                                            # Generates __init__, __repr__, and __eq__ automatically from the field list
class LinkResult:
    surface: str                                                                      # The original text as it appeared in the headline (e.g., "Paris")
    ner_label: str                                                                    # The NER category assigned by spaCy (e.g., "GPE", "ORG")
    predicted_node_id: Optional[str]                                                  # The matched KG node id, or None when the correct answer is NIL
    reason: str                                                                       # Fixed token explaining the outcome — must match the diagnostic contract exactly
    candidates: List[Dict] = field(default_factory=list)                             # All candidates returned before filtering; default_factory avoids shared-list bug

NER_LABEL_TO_KG_TYPE: Dict[str, Set[str]] = {                                       # Maps each spaCy NER label to the set of KG node labels that should match it
    "GPE":    {"City", "Country", "Region"},                                         # Geo-political entities map to place-type graph labels
    "ORG":    {"Organization", "Company"},                                            # Organizations map to company/org graph labels
    "PERSON": {"Person"},                                                             # Person entities map to the Person node label
    "LOC":    {"Continent", "Ocean", "Mountain"},                                    # Physical locations map to geographic feature labels
    "PRODUCT": {"Product"},                                                           # Product entities map to the Product node label
}

def type_filter(candidates: List[Dict], ner_label: str) -> List[Dict]:              # Removes candidates whose KG labels don't overlap with the NER type
    allowed = NER_LABEL_TO_KG_TYPE.get(ner_label, set())                            # Returns empty set for unmapped labels, which filters out all candidates
    return [c for c in candidates if set(c["kg_labels"]) & allowed]                 # Keeps only candidates whose label set intersects with the allowed types

def disambiguate_type(session, surface: str, ner_label: str) -> LinkResult:         # Stage-1 linker: resolves via type filter or signals the next step needed
    candidates = get_candidates(session, surface)                                     # Fetches all KG nodes whose name matches the surface form
    filtered   = type_filter(candidates, ner_label)                                  # Narrows to candidates compatible with the NER label

    if len(filtered) == 1:                                                            # Exactly one type-compatible candidate — unambiguously resolved
        return LinkResult(
            surface=surface, ner_label=ner_label,                                    # Echoes inputs so the caller can trace which mention was processed
            predicted_node_id=filtered[0]["id"],                                     # The single surviving candidate is the answer
            reason="resolved-by-type",                                               # Diagnostic token: type filter alone was sufficient
            candidates=candidates,                                                    # Full pre-filter list preserved for inspection and debugging
        )
    return LinkResult(
        surface=surface, ner_label=ner_label,                                        # Echoes inputs so the caller can trace which mention was processed
        predicted_node_id=None,                                                      # No single answer yet — hierarchy step needed, or no match exists
        reason="needs-hierarchy" if filtered else "nil-no-candidates",               # Distinguishes: ambiguous survivors vs. zero survivors after filtering
        candidates=candidates,                                                        # Full pre-filter list preserved for inspection and debugging
    )

with driver.session() as session:                                                     # Borrows a connection from the pool for the two demonstration calls
    print("Paris (GPE):", disambiguate_type(session, "Paris", "GPE"))               # Should resolve to the Paris city node via the type filter alone
    print("Apple (ORG):", disambiguate_type(session, "Apple", "ORG"))               # Should resolve to the Apple company node via the type filter alone
```

---

## Section 5 — Hierarchical Traversal with `[:SUBCLASS_OF*0..]`

### Explanation

**The hierarchy query**

```python
HIERARCHY_QUERY = """
MATCH (c:Entity {id: $cand_id})-[:SUBCLASS_OF*0..]->(ancestor:Entity)
RETURN collect(labels(ancestor)) AS ancestor_label_sets
"""
```

Breaking it down:

- `{id: $cand_id}` — start at one specific node by its id.
- `-[:SUBCLASS_OF*0..]->` — follow `SUBCLASS_OF` links upward. `*0..` means "zero or more hops," so it includes the node itself plus all its ancestors. This is what climbs the hierarchy.
- `collect(labels(ancestor))` — gather every ancestor's labels into one list-of-lists, returned as a single record.

---

**Flattening the ancestor label sets**

```python
def get_ancestor_labels(session, cand_id: str) -> Set[str]:
    result = session.run(HIERARCHY_QUERY, cand_id=cand_id)
    record = result.single()
    if record is None:
        return set()
    ancestor_labels: Set[str] = set()
    for label_list in record["ancestor_label_sets"]:
        ancestor_labels.update(label_list)
    return ancestor_labels
```

`result.single()` grabs the one record we expect. The `if record is None` check handles a node id that doesn't exist. The loop flattens the list-of-lists into one set. For Paris, that set ends up something like `{"Entity", "City", "Country", "Continent", "World"}` — all the labels found along the way up.

---

**The hierarchy disambiguation function**

```python
hierarchy_matches = [
    c for c in candidates
    if get_ancestor_labels(session, c["id"]) & allowed
]

if len(hierarchy_matches) == 1:
    return LinkResult(..., reason="resolved-by-hierarchy", ...)
```

For each candidate, we ask: does any of its ancestors have a type-compatible label? If so, keep it. The token here is `"resolved-by-hierarchy"`, kept separate from `"resolved-by-type"` so the diagnostics can tell how often each method does the work.

---

**Why this matters for the recipe assignment**

The geo chain `Paris → France → Europe → Continent` works just like the cuisine chain `Sichuan → Chinese → Asian → World`. In the recipe graph, a candidate might sit higher up the hierarchy than the NER label expects. The same `[:SUBCLASS_OF*0..]` query handles it — only the data in the graph changes, not the Cypher.

---

### Full Section 5 Code Block

```python
HIERARCHY_QUERY = """
MATCH (c:Entity {id: $cand_id})-[:SUBCLASS_OF*0..]->(ancestor:Entity)
RETURN collect(labels(ancestor)) AS ancestor_label_sets
"""  # Cypher constant: follows SUBCLASS_OF edges upward (*0.. includes the node itself) and collects all ancestor labels into one list

def get_ancestor_labels(session, cand_id: str) -> Set[str]:                          # Returns the flattened set of all labels found across a node's ancestors
    result = session.run(HIERARCHY_QUERY, cand_id=cand_id)                           # Runs the hierarchy query with the candidate's id passed as a safe parameter
    record = result.single()                                                           # Expects exactly one result row from the COLLECT aggregation
    if record is None:                                                                 # Guards against a cand_id that doesn't exist in the graph
        return set()                                                                   # Returns an empty set so the caller's intersection produces no matches
    ancestor_labels: Set[str] = set()                                                 # Accumulates all labels found across every ancestor node
    for label_list in record["ancestor_label_sets"]:                                  # Iterates over each ancestor's label list (the query returns a list of lists)
        ancestor_labels.update(label_list)                                            # Merges this ancestor's labels into the running set
    return ancestor_labels                                                             # Final set contains every label seen anywhere up the hierarchy chain

def disambiguate_hierarchy(session, surface: str, ner_label: str) -> LinkResult:     # Two-stage linker: type filter first, then hierarchy traversal if needed
    candidates = get_candidates(session, surface)                                     # Fetches all KG nodes whose name matches the surface form
    filtered   = type_filter(candidates, ner_label)                                  # Narrows to candidates compatible with the NER label

    if len(filtered) == 1:                                                            # Type filter alone resolved it — no hierarchy query needed
        return LinkResult(
            surface=surface, ner_label=ner_label,                                    # Echoes inputs so the caller can trace which mention was processed
            predicted_node_id=filtered[0]["id"],                                     # The single surviving candidate is the answer
            reason="resolved-by-type", candidates=candidates                         # Stage-1 token; full candidate list preserved for inspection
        )

    allowed = NER_LABEL_TO_KG_TYPE.get(ner_label, set())                            # Retrieves allowed types again for the hierarchy intersection check
    hierarchy_matches = [                                                             # Candidates whose ancestors include a type-compatible label
        c for c in candidates                                                         # Searches the full original candidate list for the hierarchy pass
        if get_ancestor_labels(session, c["id"]) & allowed                           # Keeps candidate if any ancestor label overlaps with the allowed types
    ]

    if len(hierarchy_matches) == 1:                                                   # Exactly one candidate survived the hierarchy check — unambiguously resolved
        return LinkResult(
            surface=surface, ner_label=ner_label,                                    # Echoes inputs so the caller can trace which mention was processed
            predicted_node_id=hierarchy_matches[0]["id"],                            # The single hierarchy-compatible candidate is the answer
            reason="resolved-by-hierarchy",                                          # Diagnostic token: hierarchy traversal was required to resolve
            candidates=candidates                                                      # Full pre-filter list preserved for inspection and debugging
        )

    return LinkResult(                                                                 # Neither stage could produce a unique answer
        surface=surface, ner_label=ner_label,                                        # Echoes inputs so the caller can trace which mention was processed
        predicted_node_id=None,                                                      # No single match found — return NIL
        reason="nil-ambiguous" if hierarchy_matches else "nil-no-candidates",        # Distinguishes: multiple survivors vs. zero survivors after all stages
        candidates=candidates                                                          # Full pre-filter list preserved for inspection and debugging
    )

with driver.session() as session:                                                     # Borrows a connection from the pool for the demonstration call
    result = disambiguate_hierarchy(session, "Paris", "LOC")                         # Tests the hierarchy path with LOC, which type filter alone won't resolve
    print("Paris (LOC via hierarchy):", result)                                       # Prints the full LinkResult so we can inspect reason and predicted id
```

---

## Section 6 — NIL Handling + Reason Plumbing

### Explanation

**Early exit for absent surface forms**

```python
candidates = get_candidates(session, surface)

if not candidates:
    return LinkResult(
        surface=surface, ner_label=ner_label,
        predicted_node_id=None,
        reason="nil-no-candidates",
        candidates=[]
    )
```

This is the earliest exit. If no node matches the name, there's nothing to disambiguate, so we return NIL right away. That skips the type filter and the hierarchy queries, which would only burn database round-trips for the same answer.

---

**Widening the pool for the hierarchy pass**

```python
pool = filtered if filtered else candidates
```

If the type filter wiped out every candidate (for example, the NER label isn't in our mapping yet, so `allowed` is empty), we fall back to the full candidate list for the hierarchy step. Without this, the hierarchy would have nothing to look at, and we'd report `nil-no-candidates` even though the entity is actually in the graph under a label we just haven't mapped.

---

**Distinguishing the two NIL cases**

```python
return LinkResult(
    ...,
    reason="nil-ambiguous" if hierarchy_matches else "nil-no-candidates",
    ...
)
```

`"nil-ambiguous"` means the entity is in the graph but we can't pick between several matches — we'd need a tie-breaker. `"nil-no-candidates"` means nothing matched at all, so the graph may be incomplete or the NER may have misfired.

These two are tracked separately by the diagnostics, with their own counts. Merging them into one token (or swapping hyphens for underscores) would lose that distinction and break the dashboard.

---

### Full Section 6 Code Block

```python
def link_entity(session, surface: str, ner_label: str) -> LinkResult:               # Complete three-stage linker: early NIL → type filter → hierarchy → NIL
    candidates = get_candidates(session, surface)                                     # Fetches all KG nodes whose name matches the surface form

    if not candidates:                                                                # Name absent from the graph entirely — skip all disambiguation stages
        return LinkResult(
            surface=surface, ner_label=ner_label,                                    # Echoes inputs so the caller can trace which mention was processed
            predicted_node_id=None,                                                  # Nothing in the graph matches this name
            reason="nil-no-candidates",                                              # Earliest possible NIL: surface form not found in the KG at all
            candidates=[]                                                             # Empty list; nothing to inspect
        )

    filtered = type_filter(candidates, ner_label)                                    # Narrows candidates to those whose KG labels are compatible with the NER type

    if len(filtered) == 1:                                                            # Type filter alone resolved it — no hierarchy query needed
        return LinkResult(
            surface=surface, ner_label=ner_label,                                    # Echoes inputs so the caller can trace which mention was processed
            predicted_node_id=filtered[0]["id"],                                     # The single surviving candidate is the answer
            reason="resolved-by-type",                                               # Stage-1 diagnostic token
            candidates=candidates                                                      # Full pre-filter list preserved for inspection and debugging
        )

    pool    = filtered if filtered else candidates                                    # Falls back to all candidates when the type filter returned nothing
    allowed = NER_LABEL_TO_KG_TYPE.get(ner_label, set())                            # Retrieves allowed types for the hierarchy intersection check
    hierarchy_matches = [                                                             # Candidates whose ancestors include a type-compatible label
        c for c in pool                                                               # Searches only the narrowed pool (filtered results, or all candidates)
        if get_ancestor_labels(session, c["id"]) & allowed                           # Keeps candidate if any ancestor label overlaps with the allowed types
    ]

    if len(hierarchy_matches) == 1:                                                   # Exactly one candidate survived the hierarchy check
        return LinkResult(
            surface=surface, ner_label=ner_label,                                    # Echoes inputs so the caller can trace which mention was processed
            predicted_node_id=hierarchy_matches[0]["id"],                            # The single hierarchy-compatible candidate is the answer
            reason="resolved-by-hierarchy",                                          # Stage-2 diagnostic token
            candidates=candidates                                                      # Full pre-filter list preserved for inspection and debugging
        )

    return LinkResult(                                                                 # All stages exhausted — return NIL with a reason that distinguishes the case
        surface=surface, ner_label=ner_label,                                        # Echoes inputs so the caller can trace which mention was processed
        predicted_node_id=None,                                                      # Could not identify a unique match
        reason="nil-ambiguous" if hierarchy_matches else "nil-no-candidates",        # Distinguishes: multiple survivors vs. zero survivors after all stages
        candidates=candidates                                                          # Full pre-filter list preserved for inspection and debugging
    )

nil_demos = [                                                                         # Test cases that should each produce a NIL result via different paths
    ("Zephyria",    "GPE", "Expected: nil-no-candidates (fictional city)"),          # Name not in the graph at all — hits the earliest exit
    ("Springfield", "GPE", "Expected: nil-ambiguous (many cities, no resolver)"),   # Name in the graph but too many matches to pick a single winner
]

with driver.session() as session:                                                     # Borrows a connection from the pool for the NIL demonstration cases
    for surface, ner_label, note in nil_demos:                                       # Unpacks each demo tuple into its three components
        result = link_entity(session, surface, ner_label)                            # Runs the full three-stage linker on this surface form
        print(f"\n{note}")                                                            # Prints the expected-outcome note before the actual result
        print(f"  reason={result.reason!r}  predicted={result.predicted_node_id!r}")  # Shows the reason token and predicted id (None for NIL)
```

---

## Section 7 — P/R/F1 over a 10-Headline Gold Set

### Explanation

**The gold standard**

```python
GOLD_SET = [
    ("Paris",         "GPE",  "Q90"),
    ("Zephyria",      "GPE",  None),
    ...
]
```

Each entry is `(surface_form, ner_label, correct_node_id)`. We use `None` for "the correct answer is NIL" rather than a special string, which keeps the comparison simple: `pred_id == gold_id` works for the NIL case too, since `None == None` is `True`.

---

**The four-case fixture**

```python
if gold_id is None and pred_id is None:
    tn_nil += 1          # Correct abstention — not penalized
elif gold_id is not None and pred_id == gold_id:
    tp += 1              # Correct non-NIL link
elif gold_id is not None and pred_id != gold_id:
    fn += 1              # Missed the correct entity
else:                    # pred is not None, gold is None
    fp += 1              # Linked when should have been NIL
```

There are four possible outcomes, and the autograder tests all of them. The common bug it catches is mixing up the first case (correctly returning NIL) with the third (missing a real entity). Counting a correct NIL as a miss inflates `fn`, drags down recall, and makes the linker look worse than it is. Note that `tn_nil` never enters the scoring formulas — correct abstentions simply aren't penalized.

---

**The P/R/F1 formulas**

```python
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
```

Standard scoring. **Precision**: of the links the system made, how many were right? **Recall**: of the links it should have made, how many did it find? **F1** balances the two, so you can't game it by guessing almost nothing (high precision) or guessing everything (high recall). The `if ... else 0.0` guards avoid dividing by zero when there's nothing to score.

---

### Full Section 7 Code Block

```python
GOLD_SET = [                                                                          # 10-entry evaluation set: (surface_form, ner_label, correct_node_id)
    ("Paris",         "GPE",  "Q90"),                                                # Paris city — correct answer is node Q90
    ("London",        "GPE",  "Q84"),                                                # London city — correct answer is node Q84
    ("Apple",         "ORG",  "Q312"),                                               # Apple Inc. — correct answer is node Q312
    ("Zephyria",      "GPE",  None),                                                 # Fictional city — correct answer is NIL
    ("Springfield",   "GPE",  None),                                                 # Too many matches to pick — correct answer is NIL
    ("Berlin",        "GPE",  "Q64"),                                                # Berlin city — correct answer is node Q64
    ("Tokyo",         "GPE",  "Q1490"),                                              # Tokyo city — correct answer is node Q1490
    ("FakeCorp",      "ORG",  None),                                                 # Non-existent company — correct answer is NIL
    ("Amazon",        "ORG",  "Q3884"),                                              # Amazon company — correct answer is node Q3884
    ("Mount Olympus", "LOC",  "Q182212"),                                            # Mountain — correct answer resolves via hierarchy to node Q182212
]

def score(gold_set, session) -> Dict[str, float]:                                    # Runs all gold entries through link_entity and computes P/R/F1 metrics
    tp = fp = fn = tn_nil = 0                                                        # Initializes all four outcome counters to zero before the evaluation loop

    for surface, ner_label, gold_id in gold_set:                                     # Unpacks each gold tuple into its three components for evaluation
        pred_id = link_entity(session, surface, ner_label).predicted_node_id         # Runs the linker and extracts just the predicted node id

        if gold_id is None and pred_id is None:                                      # Both gold and prediction are NIL — correct abstention
            tn_nil += 1                                                               # Counted separately; correct NILs are not penalized in the P/R formulas
        elif gold_id is not None and pred_id == gold_id:                             # Non-NIL gold, and the prediction matches exactly
            tp += 1                                                                   # True positive: found and correctly identified the right entity
        elif gold_id is not None and pred_id != gold_id:                             # Non-NIL gold, but prediction is wrong or NIL
            fn += 1                                                                   # False negative: a real entity was missed or linked to the wrong node
        else:                                                                         # Prediction is not None but gold is None
            fp += 1                                                                   # False positive: system linked when the correct answer was NIL

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0                           # Of all links the system made, what fraction were correct?
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0                           # Of all entities that should have been linked, what fraction were found?
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # Harmonic mean of precision and recall; guards against zero division

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN-NIL": tn_nil,                            # Raw counts for diagnosing individual failures in the gold set
        "Precision": round(precision, 4),                                             # Rounded to 4 decimal places for clean display
        "Recall":    round(recall, 4),                                                # Rounded to 4 decimal places for clean display
        "F1":        round(f1, 4),                                                    # Rounded to 4 decimal places for clean display
    }

with driver.session() as session:                                                     # Borrows a connection from the pool for running all 10 gold evaluations
    metrics = score(GOLD_SET, session)                                                # Executes the full scoring pass and collects all metric results

print("--- Evaluation results ---")                                                  # Section header before metric output
for k, v in metrics.items():                                                         # Iterates over each metric name/value pair in the results dict
    print(f"  {k}: {v}")                                                              # Prints each metric name and value, indented for readability
```

---

## Section 8 — Pivot to Assignment

### Explanation

**The concept mapping**

```python
mapping = {
    "Dataset":           "200-doc recipe corpus (not 10 headlines)",
    "KG node types":     "Recipe, Cuisine, Ingredient, Author, Technique",
    "Key NER labels":    "CUISINE, INGREDIENT, PERSON (author)",
    "Hierarchy example": "Sichuan → Chinese → Asian → World",
    "Cypher beat 3":     "[:SUBCLASS_OF*0..] — identical Cypher, different graph",
}
```

Everything in the demo has a recipe-domain twin. The key takeaway is in the last row: the `[:SUBCLASS_OF*0..]` query is exactly the same in your assignment — only the graph data is different. You don't rewrite the traversal. You just fill the recipe graph with the cuisine hierarchy and make sure `NER_LABEL_TO_KG_TYPE` maps the recipe NER labels to the right graph labels.

---

**The four TODO functions**

```python
todos = [
    "get_candidates(session, surface)          → parameterized Cypher",
    "type_filter(candidates, ner_label)        → extend NER_LABEL_TO_KG_TYPE for recipes",
    "get_ancestor_labels(session, cand_id)     → [:SUBCLASS_OF*0..] traversal",
    "disambiguate(session, surface, ner_label) → orchestrate all stages → LinkResult",
]
```

Functions 1 and 3 are basically the demo versions — copy them and update the connection details. Function 2 needs new recipe entries in `NER_LABEL_TO_KG_TYPE`. Function 4 ties it together, running the steps in order: candidates → type filter → hierarchy → NIL.

---

**Reason token contract**

```python
for token in ["resolved-by-type", "resolved-by-hierarchy", "nil-no-candidates", "nil-ambiguous"]:
    print(f"  {token!r}")
```

The integration repo runs `grep` for these exact strings to build its dashboards, and the autograder checks for them too. If you use underscores, abbreviate, or change them at all, the search finds nothing and the dashboard quietly shows zeros. Treat this list as a fixed contract — use these strings exactly.

---

**Two signals beyond flat type filtering (for your PR strategy paragraph)**

The assignment asks you to name two signals beyond the basic type filter:

1. **Hierarchy traversal** — already built. Walk `[:SUBCLASS_OF*0..]` to find type-compatible ancestors. This is the main one and the one the thresholds expect.
2. **A second signal** — pick one of:
   - *prior probability* (favor candidates that show up most often as correct answers in training)
   - *surface alias matching* (add an `ALIAS` link so `"Szechuan"` points to `:Sichuan`)
   - *contextual co-occurrence* (if the text mentions `"noodles"` or `"wok"`, lean toward Chinese cuisine)

Name whichever two you actually implement in your one-paragraph PR summary.

---

### Full Section 8 Code Block

```python
print("=" * 60)                                                                       # Prints a 60-character divider line for visual separation
print("ASSIGNMENT PIVOT SUMMARY")                                                     # Section title announcing the recipe-domain mapping
print("=" * 60)                                                                       # Closing divider to bracket the section title

mapping = {                                                                           # Maps each geo-demo concept to its recipe-domain equivalent
    "Dataset":           "200-doc recipe corpus (not 10 headlines)",                 # Scale up from 2 test headlines to 200 recipe documents
    "KG node types":     "Recipe, Cuisine, Ingredient, Author, Technique",           # Recipe graph labels replace the geo city/country/continent labels
    "Key NER labels":    "CUISINE, INGREDIENT, PERSON (author)",                     # New NER categories to add to NER_LABEL_TO_KG_TYPE for the assignment
    "Hierarchy example": "Sichuan → Chinese → Asian → World",                       # Cuisine hierarchy mirrors the geo city → country → continent chain
    "Cypher beat 3":     "[:SUBCLASS_OF*0..] — identical Cypher, different graph",  # The hierarchy traversal query is reused unchanged; only the data differs
}
for concept, equivalent in mapping.items():                                           # Iterates over each concept/equivalent pair in the mapping dict
    print(f"  {concept:20s}: {equivalent}")                                          # Prints concept padded to 20 chars, then its recipe-domain equivalent

print("\nTODO functions to implement in linker.py:")                                 # Heading for the four required function stubs in the assignment
todos = [                                                                             # Ordered list of functions to implement in linker.py
    "get_candidates(session, surface)          → parameterized Cypher",              # Beat 1: safe name lookup — same pattern as demo, update connection details
    "type_filter(candidates, ner_label)        → extend NER_LABEL_TO_KG_TYPE for recipes",  # Beat 2: add recipe NER labels (CUISINE, INGREDIENT, PERSON) to the mapping dict
    "get_ancestor_labels(session, cand_id)     → [:SUBCLASS_OF*0..] traversal",     # Beat 3: hierarchy climb — identical query to the demo, different graph data
    "disambiguate(session, surface, ner_label) → orchestrate all stages → LinkResult",  # Beat 4: ties all three stages together in order and returns a LinkResult
]
for i, todo in enumerate(todos, 1):                                                   # enumerate(..., 1) numbers items starting at 1 instead of 0
    print(f"  {i}. {todo}")                                                           # Prints each TODO with its step number for quick reference

print("\nTest-split thresholds (partial-cascade implementation passes all three):")  # Minimum scores the autograder requires to mark the submission passing
print("  Precision ≥ 0.80")                                                          # At least 80% of links the system makes must be correct
print("  Recall    ≥ 0.65")                                                          # At least 65% of real entities in the test set must be found
print("  F1        ≥ 0.72")                                                          # Harmonic mean of precision and recall must meet this floor

print("\nReason token vocabulary (integration diagnostics grep for these exactly):")  # Warns that these strings must appear verbatim — no spelling variations
for token in ["resolved-by-type", "resolved-by-hierarchy", "nil-no-candidates", "nil-ambiguous"]:  # The four exact reason tokens the diagnostics and autograder search for
    print(f"  {token!r}")                                                             # Prints each token with surrounding quotes to emphasize exact spelling required
```
