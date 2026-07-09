# SI Demo Walkthrough — Movie Knowledge Graph
## Module 9A Integrated Lab

**Domain:** Movies — 17 films with directors, genres (SKOS), years, lead actors, and optional Oscar wins.  
**Skills demonstrated:** Turtle authoring · Fuseki stand-up · SELECT+GROUP BY · OPTIONAL · CONSTRUCT · rdflib cross-check.

---

## Section 1 — Turtle Knowledge Graph (`movie_demo/movies.ttl`)

### 1.1 Namespace Prefix Declarations

```turtle
@prefix mv:   <http://example.org/movie/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
```

| Line | What it does |
|------|-------------|
| `@prefix mv: …` | Binds the short alias `mv:` to the base IRI for this dataset. Every movie, director, and genre node starts with `http://example.org/movie/`. |
| `@prefix skos: …` | Imports the SKOS vocabulary. SKOS gives us `skos:prefLabel` (canonical name) and `skos:altLabel` (synonyms). |
| `@prefix xsd: …` | Imports XML Schema Datatypes for typed literals like integers and booleans. |

The trailing space-dot (` .`) ends every Turtle directive. Missing it is the single most common parse error beginners make.

---

### 1.2 SKOS Genre Concepts

```turtle
mv:Action a skos:Concept ;
    skos:prefLabel "Action" ;
    skos:altLabel  "action film" , "thriller" .
```

| Line | What it does |
|------|-------------|
| `mv:Action` | Subject IRI — the unique identifier for this genre concept. |
| `a skos:Concept` | `a` is Turtle shorthand for `rdf:type`. Declares this node as a SKOS concept — a controlled vocabulary term. |
| `skos:prefLabel "Action"` | The **preferred** label. At most one per language. This is what a search UI would display. |
| `skos:altLabel "action film"` | An **alternative** label — a synonym or variant. A SPARQL query using a property path over both label predicates can match any of these terms. |
| `skos:altLabel "thriller"` | A second alternative. The `,` connects multiple objects to the same predicate. The closing ` .` ends this concept block. |

The same pattern repeats for `mv:Drama`, `mv:Comedy`, `mv:SciFi`, and `mv:Horror`.

---

### 1.3 Director Nodes

```turtle
mv:nolan a mv:Director ; mv:name "Christopher Nolan" .
```

| Part | What it does |
|------|-------------|
| `mv:nolan` | Subject IRI — unique identifier for this director. |
| `a mv:Director` | Type declaration. `mv:Director` is a class defined implicitly by usage; no OWL declaration is needed for SPARQL. |
| `mv:name "Christopher Nolan"` | A plain string literal. The `;` and ` .` on one line is compact Turtle — legal and preferred for simple nodes. |

Ten directors are defined: Kubrick, Spielberg, Nolan, Tarantino, Coppola, Scorsese, Lynch, Anderson, Villeneuve, Fincher.

---

### 1.4 Movie Instances

```turtle
mv:dunkirk a mv:Movie ;
    mv:title       "Dunkirk" ;
    mv:year        2017 ;
    mv:genre       mv:Drama ;
    mv:directedBy  mv:nolan ;
    mv:leadActor   "Tom Hardy" , "Cillian Murphy" ;
    mv:oscarWin    true .
```

| Line | What it does |
|------|-------------|
| `mv:dunkirk` | Subject IRI — unique identifier for this film. |
| `a mv:Movie` | Type declaration. |
| `mv:title "Dunkirk"` | String literal — the film title. |
| `mv:year 2017` | Integer literal. Turtle assigns `xsd:integer` automatically to unquoted whole numbers. |
| `mv:genre mv:Drama` | **Object property** — the value is the IRI `mv:Drama` defined above. A query traverses this link to reach the SKOS labels on `mv:Drama`. |
| `mv:directedBy mv:nolan` | Object property linking to the Director node. The GROUP BY query in Q1 follows this link to reach the director's name. |
| `mv:leadActor "Tom Hardy"` | String literal. Multiple actors listed with `,`. |
| `mv:oscarWin true` | Boolean literal. **Intentionally absent on many movies** — that absence is what makes the OPTIONAL query interesting. |

Movies without `mv:oscarWin`: Inception, Interstellar, Pulp Fiction, Goodfellas, The Wolf of Wall Street, Jurassic Park, Arrival, Fight Club, Mulholland Drive, 2001: A Space Odyssey, The Shining (11 movies).  
Movies with `mv:oscarWin true`: Dunkirk, Django Unchained, The Godfather, Schindler's List, Blade Runner 2049, The Grand Budapest Hotel (6 movies).

---

### Full `movies.ttl`

```turtle
@prefix mv:   <http://example.org/movie/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

# ── Genre concepts (SKOS controlled vocabulary) ───────────────────────────────

mv:Action a skos:Concept ;
    skos:prefLabel "Action" ;
    skos:altLabel  "action film" , "thriller" .

mv:Drama a skos:Concept ;
    skos:prefLabel "Drama" ;
    skos:altLabel  "dramatic film" , "character study" .

mv:Comedy a skos:Concept ;
    skos:prefLabel "Comedy" ;
    skos:altLabel  "comic film" , "humor" .

mv:SciFi a skos:Concept ;
    skos:prefLabel "Science Fiction" ;
    skos:altLabel  "sci-fi" , "speculative fiction" .

mv:Horror a skos:Concept ;
    skos:prefLabel "Horror" ;
    skos:altLabel  "horror film" , "psychological horror" .

# ── Directors ─────────────────────────────────────────────────────────────────

mv:kubrick    a mv:Director ; mv:name "Stanley Kubrick" .
mv:spielberg  a mv:Director ; mv:name "Steven Spielberg" .
mv:nolan      a mv:Director ; mv:name "Christopher Nolan" .
mv:tarantino  a mv:Director ; mv:name "Quentin Tarantino" .
mv:coppola    a mv:Director ; mv:name "Francis Ford Coppola" .
mv:scorsese   a mv:Director ; mv:name "Martin Scorsese" .
mv:lynch      a mv:Director ; mv:name "David Lynch" .
mv:anderson   a mv:Director ; mv:name "Wes Anderson" .
mv:villeneuve a mv:Director ; mv:name "Denis Villeneuve" .
mv:fincher    a mv:Director ; mv:name "David Fincher" .

# ── Movies (17 total) ─────────────────────────────────────────────────────────

mv:inception a mv:Movie ;
    mv:title "Inception" ; mv:year 2010 ; mv:genre mv:SciFi ;
    mv:directedBy mv:nolan ;
    mv:leadActor "Leonardo DiCaprio" , "Joseph Gordon-Levitt" .

mv:interstellar a mv:Movie ;
    mv:title "Interstellar" ; mv:year 2014 ; mv:genre mv:SciFi ;
    mv:directedBy mv:nolan ;
    mv:leadActor "Matthew McConaughey" , "Anne Hathaway" .

mv:dunkirk a mv:Movie ;
    mv:title "Dunkirk" ; mv:year 2017 ; mv:genre mv:Drama ;
    mv:directedBy mv:nolan ;
    mv:leadActor "Tom Hardy" , "Cillian Murphy" ;
    mv:oscarWin true .

mv:pulpfiction a mv:Movie ;
    mv:title "Pulp Fiction" ; mv:year 1994 ; mv:genre mv:Drama ;
    mv:directedBy mv:tarantino ;
    mv:leadActor "John Travolta" , "Samuel L. Jackson" .

mv:djangounchained a mv:Movie ;
    mv:title "Django Unchained" ; mv:year 2012 ; mv:genre mv:Drama ;
    mv:directedBy mv:tarantino ;
    mv:leadActor "Jamie Foxx" , "Leonardo DiCaprio" ;
    mv:oscarWin true .

mv:thegodfather a mv:Movie ;
    mv:title "The Godfather" ; mv:year 1972 ; mv:genre mv:Drama ;
    mv:directedBy mv:coppola ;
    mv:leadActor "Marlon Brando" , "Al Pacino" ;
    mv:oscarWin true .

mv:goodfellas a mv:Movie ;
    mv:title "Goodfellas" ; mv:year 1990 ; mv:genre mv:Drama ;
    mv:directedBy mv:scorsese ;
    mv:leadActor "Robert De Niro" , "Ray Liotta" .

mv:thewolfofwallstreet a mv:Movie ;
    mv:title "The Wolf of Wall Street" ; mv:year 2013 ; mv:genre mv:Drama ;
    mv:directedBy mv:scorsese ;
    mv:leadActor "Leonardo DiCaprio" , "Jonah Hill" .

mv:schindlerslist a mv:Movie ;
    mv:title "Schindler's List" ; mv:year 1993 ; mv:genre mv:Drama ;
    mv:directedBy mv:spielberg ;
    mv:leadActor "Liam Neeson" , "Ben Kingsley" ;
    mv:oscarWin true .

mv:jurassicpark a mv:Movie ;
    mv:title "Jurassic Park" ; mv:year 1993 ; mv:genre mv:SciFi ;
    mv:directedBy mv:spielberg ;
    mv:leadActor "Sam Neill" , "Jeff Goldblum" .

mv:bladerunner2049 a mv:Movie ;
    mv:title "Blade Runner 2049" ; mv:year 2017 ; mv:genre mv:SciFi ;
    mv:directedBy mv:villeneuve ;
    mv:leadActor "Ryan Gosling" , "Harrison Ford" ;
    mv:oscarWin true .

mv:arrival a mv:Movie ;
    mv:title "Arrival" ; mv:year 2016 ; mv:genre mv:SciFi ;
    mv:directedBy mv:villeneuve ;
    mv:leadActor "Amy Adams" , "Jeremy Renner" .

mv:grandbudapesthotel a mv:Movie ;
    mv:title "The Grand Budapest Hotel" ; mv:year 2014 ; mv:genre mv:Comedy ;
    mv:directedBy mv:anderson ;
    mv:leadActor "Ralph Fiennes" , "Tony Revolori" ;
    mv:oscarWin true .

mv:fightclub a mv:Movie ;
    mv:title "Fight Club" ; mv:year 1999 ; mv:genre mv:Drama ;
    mv:directedBy mv:fincher ;
    mv:leadActor "Brad Pitt" , "Edward Norton" .

mv:mulhollanddrive a mv:Movie ;
    mv:title "Mulholland Drive" ; mv:year 2001 ; mv:genre mv:Drama ;
    mv:directedBy mv:lynch ;
    mv:leadActor "Naomi Watts" , "Laura Harring" .

mv:twothousandodoneasodyssey a mv:Movie ;
    mv:title "2001: A Space Odyssey" ; mv:year 1968 ; mv:genre mv:SciFi ;
    mv:directedBy mv:kubrick ;
    mv:leadActor "Keir Dullea" , "Gary Lockwood" .

mv:theshining a mv:Movie ;
    mv:title "The Shining" ; mv:year 1980 ; mv:genre mv:Horror ;
    mv:directedBy mv:kubrick ;
    mv:leadActor "Jack Nicholson" , "Shelley Duvall" .
```

---

## Section 2 — Demo Query 1: SELECT with GROUP BY

### Line-by-line explanation

```sparql
PREFIX mv:   <http://example.org/movie/>
```
Declares the `mv:` prefix so the query engine can expand short IRIs. Must match the `@prefix` in the TTL.

```sparql
SELECT ?dirName (COUNT(?movie) AS ?movieCount) WHERE {
```
`SELECT` lists the output columns. `COUNT(?movie)` is an **aggregate function** that counts how many distinct bindings `?movie` has within each group. `AS ?movieCount` names the computed column.

```sparql
    ?movie     a mv:Movie ;
               mv:directedBy ?dir .
```
`?movie` binds to every IRI typed `mv:Movie`. The `;` continues assigning predicates to the same subject. `mv:directedBy ?dir` follows the object property to bind `?dir` to the Director node.

```sparql
    ?dir       mv:name ?dirName .
```
Traverses one more hop: from the Director node to its name literal. This two-hop traversal (`Movie → Director → name`) is what GROUP BY collapses.

```sparql
}
GROUP BY ?dirName
ORDER BY DESC(?movieCount) ?dirName
```
`GROUP BY ?dirName` collapses all rows that share the same director name into a single output row, with `COUNT(?movie)` summing the movies in that group.  
`ORDER BY DESC(?movieCount)` sorts with the most prolific director first.

**Expected result:** 10 rows (one per director). Nolan appears first with 3 movies; all others have 1–2.

---

### Full Q1 — SELECT with GROUP BY

```sparql
PREFIX mv:   <http://example.org/movie/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?dirName (COUNT(?movie) AS ?movieCount) WHERE {
    ?movie     a mv:Movie ;
               mv:directedBy ?dir .
    ?dir       mv:name ?dirName .
}
GROUP BY ?dirName
ORDER BY DESC(?movieCount) ?dirName
```

---

## Section 3 — Demo Query 2: SELECT with OPTIONAL

### Line-by-line explanation

```sparql
SELECT ?title ?year ?oscarWin WHERE {
```
Three projected variables — the third (`?oscarWin`) may be **unbound** for movies that lack the triple.

```sparql
    ?movie a mv:Movie ;
           mv:title ?title ;
           mv:year  ?year .
```
**Required** triple patterns — a movie must match all three to appear in results at all.

```sparql
    OPTIONAL { ?movie mv:oscarWin ?oscarWin }
```
`OPTIONAL { … }` attempts to match the inner pattern. If the match succeeds, `?oscarWin` is bound to `true`. **If the match fails** (the movie has no `mv:oscarWin` triple), the row is **still returned** — but `?oscarWin` is unbound (Python `None`, SPARQL JSON omits the key).

Without `OPTIONAL`, this line would become a **required** triple pattern and the 11 movies without an Oscar-win triple would simply disappear from the results. That is the most important distinction to understand: `OPTIONAL` keeps rows; a bare required pattern discards rows.

```sparql
}
ORDER BY ?year
```
Sorts all 17 rows chronologically.

**Expected result:** 17 rows. Six rows have `?oscarWin = true`; eleven rows have `?oscarWin` unbound.

---

### Full Q2 — SELECT with OPTIONAL

```sparql
PREFIX mv:   <http://example.org/movie/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?title ?year ?oscarWin WHERE {
    ?movie a mv:Movie ;
           mv:title ?title ;
           mv:year  ?year .
    OPTIONAL { ?movie mv:oscarWin ?oscarWin }
}
ORDER BY ?year
```

---

## Section 4 — Demo Query 3: CONSTRUCT

### Line-by-line explanation

```sparql
CONSTRUCT {
    ?movie mv:directedByName ?dirName .
}
```
The `CONSTRUCT` block is a **triple template**. For every solution produced by the `WHERE` clause, the engine instantiates this template with the bound variables, producing one output triple. The result is a *graph* (a set of triples), not a table.

Here we "flatten" the two-hop path `Movie → Director → name` into a single direct triple `Movie → directedByName → "name string"`. This is useful for export, visualization, or feeding a downstream service that doesn't need to know about the intermediate Director node.

```sparql
WHERE {
    ?movie a mv:Movie ;
           mv:directedBy ?dir .
    ?dir   mv:name ?dirName .
}
```
The WHERE clause is a standard SELECT-style graph pattern. `?movie` and `?dirName` are the only variables referenced in the CONSTRUCT template, so `?dir` is an internal join variable only.

**Expected result:** A graph of 17 triples — one per movie — each of the form:
```
<mv:inception>  mv:directedByName  "Christopher Nolan" .
```

In Python with rdflib, `graph.query(Q3_CONSTRUCT).graph` returns the constructed graph as an `rdflib.Graph` object you can iterate, serialize, or query further.

---

### Full Q3 — CONSTRUCT

```sparql
PREFIX mv:   <http://example.org/movie/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

CONSTRUCT {
    ?movie mv:directedByName ?dirName .
}
WHERE {
    ?movie a mv:Movie ;
           mv:directedBy ?dir .
    ?dir   mv:name ?dirName .
}
```

---

## Section 5 — Python Script (`movie_demo/movie_queries.py`)

### 5.1 Imports and constants

```python
from pathlib import Path
import requests
import rdflib
from SPARQLWrapper import JSON, SPARQLWrapper

DEMO_TTL    = Path(__file__).parent / "movies.ttl"
FUSEKI_BASE = "http://localhost:3030"
DATASET     = "movies"
ENDPOINT    = f"{FUSEKI_BASE}/{DATASET}/sparql"
```

| Symbol | Role |
|--------|------|
| `Path(__file__).parent` | Resolves to the directory containing this script. Prevents broken paths when the script is called from a different working directory. |
| `rdflib` | Python RDF library — parses Turtle and runs SPARQL locally with no server. |
| `SPARQLWrapper` | HTTP client wrapper for the SPARQL 1.1 Protocol. Sends queries to Fuseki and deserializes JSON results. |
| `ENDPOINT` | Fuseki's SPARQL query endpoint for the `movies` dataset. All remote queries target this URL. |

---

### 5.2 Query strings

```python
PREFIXES = """
PREFIX mv:   <http://example.org/movie/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""
```

Defined once and prepended to every query constant. Avoids repetition and guarantees consistency — a typo in a prefix would cause all queries to fail visibly rather than silently returning zero results.

`Q1_GROUP_BY`, `Q2_OPTIONAL`, `Q3_CONSTRUCT` are module-level string constants. Storing them at module level makes them testable and REPL-inspectable without running the full script.

---

### 5.3 Helper functions

```python
def load_local_graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(DEMO_TTL), format="turtle")
    return g
```

`rdflib.Graph()` — creates an empty in-memory triple store.  
`.parse(…, format="turtle")` — reads the TTL file and populates the store. Returns the graph so the caller can reuse it across multiple queries without re-parsing.

```python
def run_local(graph: rdflib.Graph, query: str) -> list:
    return list(graph.query(query))
```

`graph.query(query)` — runs SPARQL 1.1 directly against the in-memory graph using rdflib's built-in engine. No HTTP call, no Docker required. `list(…)` materializes the lazy result iterator so `.count()` and indexing work.

```python
def run_local_construct(graph: rdflib.Graph, query: str) -> rdflib.Graph:
    return graph.query(query).graph
```

For CONSTRUCT queries, rdflib wraps the result in a `Result` object. The `.graph` attribute holds the constructed `rdflib.Graph`. SELECT queries use `.bindings` or direct iteration instead.

```python
def run_fuseki_select(query: str) -> list[dict]:
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]
```

`SPARQLWrapper(ENDPOINT)` — points the wrapper at Fuseki's HTTP endpoint.  
`.setReturnFormat(JSON)` — requests SPARQL 1.1 JSON results.  
`.query().convert()` — sends the HTTP GET and parses the JSON body.  
`["results"]["bindings"]` — extracts the array of solution rows from the SPARQL JSON envelope.

---

### 5.4 Rendering Q2 OPTIONAL results

```python
for row in run_local(g, Q2_OPTIONAL):
    oscar = str(row.oscarWin) if row.oscarWin is not None else "(unbound)"
    print(f"  {str(row.title):<45} ({row.year})  {oscar}")
```

`row.oscarWin` is `None` when the OPTIONAL block failed to match (no `mv:oscarWin` triple on that movie). Checking `is not None` rather than truthiness is important here because the boolean value `false` is also falsy — though in this dataset all `mv:oscarWin` triples have `true`.

---

### 5.5 Rendering Q3 CONSTRUCT results

```python
result_graph = g.query(Q3_CONSTRUCT).graph
print(f"  Constructed {len(result_graph)} triples.")
for s, p, o in sorted(result_graph, key=lambda t: str(t[0])):
    movie_local = str(s).split("/")[-1]
    print(f"  <{movie_local}>  directedByName  \"{o}\"")
```

`len(result_graph)` — counts the triples in the constructed graph. Should equal 17 (one per movie).  
Iterating `result_graph` yields `(subject, predicate, object)` tuples. `str(s).split("/")[-1]` extracts the local name from the full IRI for compact display.

---

### Full `movie_queries.py`

```python
#!/usr/bin/env python3
"""
SI Demo — Movie Knowledge Graph
Demonstrates TTL authoring, Fuseki stand-up, and three SPARQL query forms:
  Q1: SELECT with GROUP BY (movies per director)
  Q2: SELECT with OPTIONAL (Oscar wins)
  Q3: CONSTRUCT (build a directed-by subgraph)
"""

from pathlib import Path

import requests
import rdflib
from SPARQLWrapper import JSON, SPARQLWrapper

DEMO_TTL    = Path(__file__).parent / "movies.ttl"
FUSEKI_BASE = "http://localhost:3030"
DATASET     = "movies"
ENDPOINT    = f"{FUSEKI_BASE}/{DATASET}/sparql"

PREFIXES = """
PREFIX mv:   <http://example.org/movie/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

Q1_GROUP_BY = PREFIXES + """
SELECT ?dirName (COUNT(?movie) AS ?movieCount) WHERE {
    ?movie     a mv:Movie ;
               mv:directedBy ?dir .
    ?dir       mv:name ?dirName .
}
GROUP BY ?dirName
ORDER BY DESC(?movieCount) ?dirName
"""

Q2_OPTIONAL = PREFIXES + """
SELECT ?title ?year ?oscarWin WHERE {
    ?movie a mv:Movie ;
           mv:title ?title ;
           mv:year  ?year .
    OPTIONAL { ?movie mv:oscarWin ?oscarWin }
}
ORDER BY ?year
"""

Q3_CONSTRUCT = PREFIXES + """
CONSTRUCT {
    ?movie mv:directedByName ?dirName .
}
WHERE {
    ?movie a mv:Movie ;
           mv:directedBy ?dir .
    ?dir   mv:name ?dirName .
}
"""


def load_local_graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(DEMO_TTL), format="turtle")
    return g


def run_local(graph: rdflib.Graph, query: str) -> list:
    return list(graph.query(query))


def run_local_construct(graph: rdflib.Graph, query: str) -> rdflib.Graph:
    return graph.query(query).graph


def _create_fuseki_dataset() -> None:
    r = requests.post(
        f"{FUSEKI_BASE}/$/datasets",
        data={"dbName": DATASET, "dbType": "tdb2"},
        auth=("admin", "admin"),
    )
    if r.status_code not in (200, 409):
        r.raise_for_status()


def load_into_fuseki() -> None:
    _create_fuseki_dataset()
    turtle = DEMO_TTL.read_text(encoding="utf-8")
    r = requests.post(
        f"{FUSEKI_BASE}/{DATASET}/data",
        data=turtle.encode("utf-8"),
        headers={"Content-Type": "text/turtle"},
        auth=("admin", "admin"),
    )
    r.raise_for_status()
    print(f"Loaded movies.ttl → Fuseki  (HTTP {r.status_code})")


def run_fuseki_select(query: str) -> list[dict]:
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]


def main() -> None:
    print("Parsing movies.ttl with rdflib ...")
    g = load_local_graph()

    print("\n=== Q1: Movies per Director (GROUP BY) ===")
    rows = run_local(g, Q1_GROUP_BY)
    for row in rows:
        print(f"  {str(row.dirName):<25}  {row.movieCount} movie(s)")

    print("\n=== Q2: All Movies with Optional Oscar Win ===")
    rows = run_local(g, Q2_OPTIONAL)
    for row in rows:
        oscar = "Oscar winner" if row.oscarWin else "—"
        print(f"  {str(row.title):<45}  ({row.year})  {oscar}")

    print("\n=== Q3: CONSTRUCT — Directed-By Subgraph ===")
    result_graph = g.query(Q3_CONSTRUCT).graph
    print(f"  Constructed {len(result_graph)} triples.")
    for s, p, o in sorted(result_graph, key=lambda t: str(t[0])):
        movie_local = str(s).split("/")[-1]
        print(f"  <{movie_local}>  directedByName  \"{o}\"")

    print("\n─── Fuseki Cross-check ───")
    load_into_fuseki()
    local_count  = len(run_local(g, Q1_GROUP_BY))
    fuseki_count = len(run_fuseki_select(Q1_GROUP_BY))
    print(f"  rdflib  → {local_count} director rows")
    print(f"  Fuseki  → {fuseki_count} director rows")
    if local_count == fuseki_count:
        print("  PASS — counts match.")
    else:
        print("  FAIL — count mismatch, check the data load.")


if __name__ == "__main__":
    main()
```

---

## Section 6 — Publication Lab Queries (`queries.py`)

### 6.1 Shared infrastructure

```python
ENDPOINT = "http://localhost:3030/publications/sparql"

_PREFIXES = """
PREFIX pub:  <http://example.org/pub/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

def _run(query: str) -> list[dict]:
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(_PREFIXES + query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]
```

`_PREFIXES` is concatenated onto every query string so individual query functions don't need to repeat prefix declarations. The leading `_` marks `_run` as a module-internal helper not intended for direct use by test code.

---

### 6.2 Q1 — All Papers

```python
def q1_all_papers() -> list[dict]:
    return _run("""
        SELECT ?paper ?title ?year WHERE {
            ?paper a pub:Paper ;
                   pub:title ?title ;
                   pub:year  ?year .
        }
        ORDER BY ?year ?title
    """)
```

`?paper a pub:Paper` — binds `?paper` to every IRI typed as `pub:Paper`. There are exactly **80** such IRIs.  
`pub:title ?title` and `pub:year ?year` — read two data properties off each paper. Both are required (not OPTIONAL), so papers missing either would be silently excluded — but all 80 papers in this dataset have both.  
`ORDER BY ?year ?title` — double sort: by year first, then alphabetically within each year.

**Expected: 80 rows.**

---

### 6.3 Q2 — All Authors

```python
def q2_all_authors() -> list[dict]:
    return _run("""
        SELECT ?author ?name WHERE {
            ?author a pub:Author ;
                    pub:name ?name .
        }
        ORDER BY ?name
    """)
```

Follows the same pattern as Q1 but targets `pub:Author` nodes. There are **120** Author IRIs in the dataset — 20 prominent authors (auth001–auth020) and 100 secondary authors (auth021–auth120).

**Expected: 120 rows.**

---

### 6.4 Q3 — Papers per Year (GROUP BY)

```python
def q3_papers_by_year() -> list[dict]:
    return _run("""
        SELECT ?year (COUNT(?paper) AS ?count) WHERE {
            ?paper a pub:Paper ;
                   pub:year ?year .
        }
        GROUP BY ?year
        ORDER BY ?year
    """)
```

`COUNT(?paper)` counts how many papers have each distinct year value.  
`GROUP BY ?year` collapses all paper rows into one row per distinct year.  
The dataset spans years 2018–2024 inclusive — **7 distinct values**.  
All 7 year-group counts sum to 80.

**Expected: 7 rows.**

---

### 6.5 Q4 — Prolific Authors (HAVING)

```python
def q4_prolific_authors() -> list[dict]:
    return _run("""
        SELECT ?name (COUNT(?paper) AS ?count) WHERE {
            ?paper a pub:Paper ;
                   pub:hasAuthor ?author .
            ?author pub:name ?name .
        }
        GROUP BY ?name
        HAVING (COUNT(?paper) >= 3)
        ORDER BY DESC(?count) ?name
    """)
```

This query traverses **two hops** before aggregating: Paper → Author → name.  
`pub:hasAuthor` is an object property linking papers to Author nodes.  
`GROUP BY ?name` groups all paper–author pairs by author name.  
`HAVING (COUNT(?paper) >= 3)` filters **after** aggregation — only keeps groups where the author has ≥ 3 papers.  
The 20 prominent authors (auth001–auth020) each appear in exactly 3 papers; no secondary author appears in more than 2.

**Expected: 20 rows.**

---

### 6.6 Q5 — Papers with OPTIONAL DOI

```python
def q5_papers_with_doi() -> list[dict]:
    return _run("""
        SELECT ?title ?year ?doi WHERE {
            ?paper a pub:Paper ;
                   pub:title ?title ;
                   pub:year  ?year .
            OPTIONAL { ?paper pub:doi ?doi }
        }
        ORDER BY ?year ?title
    """)
```

`OPTIONAL { ?paper pub:doi ?doi }` — the engine tries to match `pub:doi` on every paper. Papers p001–p032 have a `pub:doi` triple; papers p033–p080 do not. When the match fails, `?doi` is simply unbound — the row is still returned with `?doi` absent from the JSON binding map.

In Python, test code checks `"doi" in row` to distinguish the two cases rather than comparing `row["doi"]` to `None` (which would raise a `KeyError`).

**Expected: 80 rows** (40 with `doi`, 40 without).

---

### 6.7 Q6 — Semantic Web Papers via SKOS

```python
def q6_semantic_web_papers() -> list[dict]:
    return _run("""
        SELECT DISTINCT ?title ?year WHERE {
            ?paper a pub:Paper ;
                   pub:title    ?title ;
                   pub:year     ?year ;
                   pub:hasTopic ?topic .
            ?topic (skos:prefLabel|skos:altLabel) ?label .
            FILTER(CONTAINS(LCASE(STR(?label)), "semantic"))
        }
        ORDER BY ?year ?title
    """)
```

`pub:hasTopic ?topic` — links the paper to a SKOS concept node.  
`?topic (skos:prefLabel|skos:altLabel) ?label` — the `|` operator is a **SPARQL 1.1 property path**. It tries both predicates on `?topic` and binds `?label` to every label found. A single paper whose topic has both a `prefLabel` and two `altLabel`s would produce three rows before `DISTINCT` collapses them.  
`FILTER(CONTAINS(LCASE(STR(?label)), "semantic"))` — case-insensitive substring match. `STR(…)` strips the RDF datatype annotation; `LCASE(…)` lowercases; `CONTAINS(…, "semantic")` checks for the substring.

Only `pub:topic_sw` has `prefLabel "Semantic Web"` — the word "semantic" appears there. Ten papers carry `pub:topic_sw` as a topic.

**Expected: 10 DISTINCT rows.**

---

### 6.8 Q7 — Recent Papers (FILTER on integer)

```python
def q7_recent_papers() -> list[dict]:
    return _run("""
        SELECT ?title ?year WHERE {
            ?paper a pub:Paper ;
                   pub:title ?title ;
                   pub:year  ?year .
            FILTER(?year > 2022)
        }
        ORDER BY ?year ?title
    """)
```

`pub:year` is stored as `xsd:integer` (Turtle assigns this automatically to unquoted whole numbers). `FILTER(?year > 2022)` uses SPARQL's built-in numeric comparison — no string parsing or casting required.

Papers from 2023: p063–p074 (12 papers). Papers from 2024: p075–p080 (6 papers). Total: **18 rows.**

---

### 6.9 Q8 — Conference Papers

```python
def q8_conference_papers() -> list[dict]:
    return _run("""
        SELECT ?title ?vname WHERE {
            ?paper a pub:Paper ;
                   pub:title       ?title ;
                   pub:publishedIn ?venue .
            ?venue pub:name ?vname ;
                   pub:type "conference" .
        }
        ORDER BY ?vname ?title
    """)
```

Two-hop traversal: `Paper → Venue → name/type`. The `pub:type "conference"` triple pattern filters venue nodes — only those with `pub:type` bound to the literal string `"conference"` match. Papers p001–p040 are at the 20 conference venues (2 papers per venue).

**Expected: 40 rows.**

---

## Section 7 — Test Suite (`tests/test_queries.py`)

### 7.1 conftest.py — session fixture

```python
@pytest.fixture(scope="session", autouse=True)
def require_fuseki():
    try:
        requests.get(FUSEKI_PING, timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip("Fuseki is not running on localhost:3030 ...")
```

`scope="session"` — runs once for the entire pytest session, not once per test.  
`autouse=True` — automatically applied to every test without needing `@pytest.mark.usefixtures`.  
`pytest.skip(…)` — when Fuseki is unreachable, marks the entire session as skipped with a descriptive message rather than failing with a cryptic connection error.

---

### 7.2 Representative test patterns

```python
def test_q1_returns_80_papers():
    results = q1_all_papers()
    assert len(results) == 80, f"Expected 80 papers, got {len(results)}"
```

The f-string in the `assert` message is the actual count — this is critical for debugging because pytest only displays the message when the assertion fails.

```python
def test_q3_total_counts_to_80():
    results = q3_papers_by_year()
    total = sum(int(r["count"]["value"]) for r in results)
    assert total == 80
```

`r["count"]["value"]` — the SPARQL JSON envelope wraps every binding as `{"type": "...", "value": "..."}`. Aggregate results like `COUNT` come back as strings and must be cast to `int`.

```python
def test_q5_some_dois_present_some_absent():
    results = q5_papers_with_doi()
    with_doi    = [r for r in results if "doi" in r]
    without_doi = [r for r in results if "doi" not in r]
    assert len(with_doi) > 0    # OPTIONAL matched at least once
    assert len(without_doi) > 0 # OPTIONAL also missed at least once
```

This test verifies the OPTIONAL semantics are actually exercised — not just that the query runs without error.

---

### Full `tests/conftest.py`

```python
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

FUSEKI_PING = "http://localhost:3030/$/ping"


@pytest.fixture(scope="session", autouse=True)
def require_fuseki():
    """Skip the test session when Fuseki is not reachable."""
    try:
        requests.get(FUSEKI_PING, timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip(
            "Fuseki is not running on localhost:3030 — "
            "start it with `docker compose up -d` and run `python load_dataset.py` first."
        )
```

---

## Section 8 — Infrastructure Files

### 8.1 `docker-compose.yml`

```yaml
version: '3.8'
services:
  fuseki:
    image: stain/jena-fuseki:5.0.0
    ports:
      - "3030:3030"
    environment:
      - ADMIN_PASSWORD=admin
    volumes:
      - fuseki_data:/fuseki
volumes:
  fuseki_data:
```

| Key | What it does |
|-----|-------------|
| `image: stain/jena-fuseki:5.0.0` | Pulls the pinned Fuseki image (matches Lab 9A Applied). Pinning the version prevents silent upgrades. |
| `ports: "3030:3030"` | Maps container port 3030 to host port 3030. All Python scripts target `localhost:3030`. |
| `ADMIN_PASSWORD=admin` | Sets the admin credential. All `requests.post(…, auth=("admin","admin"))` calls in Python use this. |
| `volumes: fuseki_data:/fuseki` | Named Docker volume persists the TDB2 triple store between container restarts. Without this, all data is lost on `docker compose down`. |

---

### 8.2 `load_dataset.py`

```python
def create_dataset() -> None:
    response = requests.post(
        f"{FUSEKI_BASE}/$/datasets",
        data={"dbName": DATASET_NAME, "dbType": "tdb2"},
        auth=(ADMIN_USER, ADMIN_PASS),
    )
    if response.status_code == 409:
        print(f"Dataset '{DATASET_NAME}' already exists — skipping creation.")
    elif response.status_code == 200:
        print(f"Dataset '{DATASET_NAME}' created.")
    else:
        response.raise_for_status()
```

`POST /$/datasets` — Fuseki's admin REST endpoint for dataset management. The `$` prefix is the Fuseki convention for admin operations.  
`dbType: "tdb2"` — TDB2 is the recommended persistent triple store backend. TDB (v1) is legacy.  
Status 409 (Conflict) means the dataset already exists — treated as success to make the script idempotent.  
`response.raise_for_status()` — converts any other 4xx/5xx into a Python exception with the HTTP status in the message.

```python
def load_data() -> requests.Response:
    url    = f"{FUSEKI_BASE}/{DATASET_NAME}/data"
    turtle = TTL_PATH.read_text(encoding="utf-8")
    response = requests.post(
        url,
        data=turtle.encode("utf-8"),
        headers={"Content-Type": "text/turtle"},
        auth=(ADMIN_USER, ADMIN_PASS),
    )
    response.raise_for_status()
    return response
```

`/publications/data` — the **Graph Store Protocol (GSP)** endpoint. GSP is the W3C standard HTTP interface for uploading RDF into a named graph or the default graph.  
`Content-Type: text/turtle` — tells Fuseki how to parse the body. Using the wrong content type causes a 400 error.  
`.encode("utf-8")` — converts the Python string to bytes before POST. Fuseki expects raw bytes, not a Unicode string object.

---

## Quick-Start Checklist

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Fuseki
docker compose up -d

# 3. Load the publications dataset
python load_dataset.py

# 4. Run the test suite
pytest tests/ -v

# 5. Open the notebook
jupyter notebook lab_9a_integrated.ipynb

# 6. Run the SI demo (movie domain)
cd movie_demo
python movie_queries.py
```

### Expected test output

```
tests/test_queries.py::test_q1_returns_80_papers          PASSED
tests/test_queries.py::test_q1_includes_known_paper       PASSED
tests/test_queries.py::test_q2_returns_120_authors        PASSED
tests/test_queries.py::test_q2_includes_prominent_authors PASSED
tests/test_queries.py::test_q3_returns_7_year_groups      PASSED
tests/test_queries.py::test_q3_years_are_2018_to_2024     PASSED
tests/test_queries.py::test_q3_total_counts_to_80         PASSED
tests/test_queries.py::test_q4_returns_20_prolific_authors PASSED
tests/test_queries.py::test_q4_every_result_has_at_least_3_papers PASSED
tests/test_queries.py::test_q5_returns_80_rows            PASSED
tests/test_queries.py::test_q5_some_dois_present_some_absent PASSED
tests/test_queries.py::test_q6_returns_10_semantic_web_papers PASSED
tests/test_queries.py::test_q6_known_paper_included       PASSED
tests/test_queries.py::test_q7_returns_18_recent_papers   PASSED
tests/test_queries.py::test_q7_all_results_after_2022     PASSED
tests/test_queries.py::test_q8_returns_40_conference_papers PASSED
tests/test_queries.py::test_q8_neurips_papers_present     PASSED

17 passed in X.XXs
```
