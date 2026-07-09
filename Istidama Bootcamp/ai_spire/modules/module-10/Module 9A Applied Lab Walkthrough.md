# SI Demo Walkthrough — Music Knowledge Graph

> **Annotated edition.** This version keeps the original walkthrough intact and adds extra explanation throughout. Wherever a concept tends to surprise people arriving from a SQL / relational-database background, you'll find a callout like this:
>
> > **From the relational world** — short note mapping the idea back to tables, rows, keys, and JOINs.
>
> If you've never touched RDF before, read **Section 0** first — it's the mental model the rest of the document assumes.

**Domain:** Music — 15 albums with artist, genre, release year, and record label.
**Skills demonstrated:** Turtle syntax · Fuseki stand-up · SPARQL with SKOS disambiguation · year FILTER · rdflib-vs-Fuseki cross-check.

---

## Section 0 — Mental Model: From Tables to Triples

If you think in tables, the single most useful idea to absorb is this: **RDF has no tables.** There is exactly one data structure — the **triple** — and everything is built from it.

A triple is a three-part fact:

```
subject   predicate   object
mu:nevermind   mu:artist   "Nirvana"
```

Read it as a sentence: *"Nevermind — has artist — Nirvana."* A whole dataset is just a big pile of these sentences. The "graph" is what you get when triples share subjects and objects: the album *Nevermind* points to the genre *Rock*, *Rock* points to its labels, and so a network of nodes and edges forms. There are no rows, no columns, no table boundaries — just nodes connected by labeled edges.

### The translation table

Here is how the relational vocabulary maps onto RDF. Keep this handy; the rest of the document leans on it.

| Relational concept | RDF / graph equivalent | The catch |
|---|---|---|
| Table (e.g. `albums`) | A `rdf:type` class, e.g. `mu:Album` | Nothing declares the table. A node "is an Album" simply because a triple says so. |
| Row | All the triples that share one subject | Rows aren't fixed-width. Two albums can carry different predicates and nothing complains. |
| Column | A **predicate** (e.g. `mu:title`) | Predicates are global, not owned by a table. The same predicate can describe albums, artists, anything. |
| Cell value | An **object** | An object is either a literal (`"Nirvana"`, `1991`) **or** a link (IRI) to another node. |
| Primary key | The subject **IRI** (`mu:nevermind`) | It's a globally-unique URL, not a local auto-increment integer. |
| Foreign key | An IRI-valued predicate ("object property") | The link *is* the data. There's no separate `JOIN ... ON`; you just follow the edge. |
| `JOIN` | A shared variable across triple patterns | You describe the shape you want; the engine finds the connections. |
| Schema / DDL (`CREATE TABLE`) | **Optional** — RDFS, OWL, or SHACL | RDF is schema-*optional*. Data exists and is queryable with no schema at all. |
| `NULL` | An absent triple | There is no NULL. Missing data is simply a fact you never stated. |
| Lookup / reference table | A SKOS concept scheme | Comes with built-in support for synonyms (`skos:altLabel`). |
| SQL | **SPARQL** | Same `SELECT … WHERE … ORDER BY` skeleton, but the `WHERE` matches graph patterns. |
| The RDBMS engine | A **triplestore** (here, Apache Jena Fuseki) | Stores triples; answers SPARQL over HTTP. |
| Recursive CTE / self-join | A **property path** (`|`, `/`, `*`, `+`) | Multi-hop traversal is a first-class operator in the query language. |

### Three ideas that trip people up

**1. Schema-optional.** In a relational database you must `CREATE TABLE` before you can insert a single row; the schema is a gatekeeper. In RDF the data comes first and any schema is a layer you *choose* to add on top for validation or inference. In this demo there is no schema — the walkthrough notes that `mu:Album` is "a class defined implicitly by use." If you misspell a predicate as `mu:titel`, nothing errors; you just get an empty result. The discipline a relational `CREATE TABLE` enforces for you is, in RDF, your own responsibility (or SHACL's).

**2. Open-world assumption.** A relational database is *closed-world*: if a row isn't there, the thing is treated as false or non-existent. RDF is *open-world*: the absence of a triple means "not stated," **not** "false." If an album has no `mu:label` triple, RDF doesn't conclude the album has no label — only that this dataset doesn't happen to record one. This matters the moment you start reasoning over data or merging datasets.

**3. Identifiers are URLs, on purpose.** A relational primary key (`id = 42`) is meaningful only inside its own table. An RDF subject is an IRI like `http://example.org/music/nevermind` — globally unique by design, so two independently-built datasets can refer to the same thing and *automatically* line up when merged. That global-merge property is the whole reason the Web of Data uses URLs as keys. (See `@prefix` below for how Turtle keeps these from cluttering the page.)

With that model in place, the rest of the document is just detail.

---

## Section 1 — Turtle Knowledge Graph (`music_demo.ttl`)

Turtle is **not a schema and not a database** — it's a *file format*. It's a way to write triples as readable text, sitting at the same conceptual level as a `.csv` file or a SQL dump. The same triples could be written in JSON-LD or RDF/XML and mean exactly the same thing; Turtle is just the most human-friendly spelling.

> **From the relational world** — Think of a `.ttl` file as closer to the `INSERT` statements in a database dump than to the `CREATE TABLE` statements. It carries *data* (and, optionally, vocabulary), written in a notation. Parsing it produces the graph, the way running a dump populates a database.

### 1.1 Namespace Prefix Declarations

```
@prefix mu:   <http://example.org/music/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
```

| Line | What it does |
|------|-------------|
| `@prefix mu: …` | Binds the short alias `mu:` to the base IRI for this dataset. Every album and genre you define will start with `http://example.org/music/`. |
| `@prefix skos: …` | Imports the SKOS vocabulary namespace. SKOS (Simple Knowledge Organization System) is a W3C standard for controlled vocabularies — it gives us `skos:prefLabel` and `skos:altLabel`. |
| `@prefix xsd: …` | Imports XML Schema Datatypes so we can tag literals with types like `xsd:integer` and `xsd:date`. |

The trailing space-dot (` .`) ends every Turtle directive and triple.

> **From the relational world** — There's no SQL equivalent of a prefix because relational identifiers are local strings, not URLs. The closest analogy is a schema/namespace qualifier like `dbo.` or `sales.` in front of a table name — a short alias that expands to something longer. `mu:nevermind` is purely shorthand for `http://example.org/music/nevermind`; the colon is the expansion point, not a folder separator.

### 1.2 SKOS Genre Concepts

```
mu:Rock a skos:Concept ;
    skos:prefLabel "rock" ;
    skos:altLabel  "rock and roll" ,
                   "rock music" .
```

| Line | What it does |
|------|-------------|
| `mu:Rock` | The subject — a new IRI `http://example.org/music/Rock`. |
| `a skos:Concept` | RDF shorthand for `rdf:type`. Declares this node as a SKOS concept (a term in a controlled vocabulary). The `;` continues listing predicates for the same subject. |
| `skos:prefLabel "rock"` | The **preferred** human-readable label. There must be at most one prefLabel per language. This is the canonical term. |
| `skos:altLabel "rock and roll"` | An **alternative** label — a synonym, abbreviation, or variant spelling. You can have as many altLabels as you need. The `,` connects multiple objects for the same predicate. |
| `skos:altLabel "rock music"` | A second altLabel. The closing ` .` ends the block for `mu:Rock`. |

The same pattern repeats for `mu:HipHop` (with altLabels `"rap"` and `"hip hop"`), `mu:Jazz`, `mu:Electronic`, and `mu:RnB`. SKOS lets a single query match any of these synonym labels — that is the key technique the lab exercises.

> **From the relational world** — A SKOS concept is what you'd build with a *lookup table* plus a *synonyms table*. Imagine `genres(id, canonical_name)` joined to `genre_synonyms(genre_id, synonym)`. `skos:prefLabel` is the `canonical_name` column; each `skos:altLabel` is a row in the synonyms table. The difference is that SKOS is a published W3C standard, so any tool that understands SKOS already knows what "preferred label" and "alternative label" mean — you're not inventing a private table design that only your app understands.
>
> Two punctuation marks do the work of multiple `INSERT`s here: `;` means "same subject, new predicate" (still describing `mu:Rock`), and `,` means "same subject *and* predicate, another value" (another altLabel for the same concept).

### 1.3 Album Instances

```
mu:nevermind a mu:Album ;
    mu:title       "Nevermind" ;
    mu:artist      "Nirvana" ;
    mu:genre       mu:Rock ;
    mu:releaseYear 1991 ;
    mu:label       "DGC Records" .
```

| Line | What it does |
|------|-------------|
| `mu:nevermind` | Subject IRI — a unique identifier for this album. |
| `a mu:Album` | Declares the type. `mu:Album` is a class defined implicitly by use; no OWL declaration is needed for basic SPARQL queries. |
| `mu:title "Nevermind"` | A plain string literal property. |
| `mu:artist "Nirvana"` | Another string literal. (In a richer ontology this would link to an Artist node; string literals are simpler for this demo.) |
| `mu:genre mu:Rock` | An **object property** — the value is not a literal string but the IRI `mu:Rock` defined above. This is the link a SPARQL query traverses to reach the SKOS labels. |
| `mu:releaseYear 1991` | An integer literal. Turtle assigns `xsd:integer` automatically to unquoted whole numbers. |
| `mu:label "DGC Records"` | The record label — a plain string. |

This same block is repeated 14 more times with different genres (linking to `mu:HipHop`, `mu:Jazz`, `mu:Electronic`, `mu:RnB`) and release years ranging from 1959 to 2015.

> **From the relational world** — This block is one "row" of an albums table, but notice three departures:
>
> 1. **The type is a fact, not a table.** `a mu:Album` is itself a triple (`mu:nevermind rdf:type mu:Album`). Membership in the "Album" set is data you can query and even add later — not a structural property fixed at creation.
> 2. **Literal vs. link is the literal/object-property distinction.** `mu:artist "Nirvana"` stores a value directly (like a `VARCHAR` cell). `mu:genre mu:Rock` stores a *foreign key to another node* — except there's no separate key column and no `REFERENCES` constraint; the IRI on the right simply *is* the connection. To "join" to the genre's labels later, you follow this edge.
> 3. **Datatypes are inferred and attached to the value, not the column.** `1991` becomes an `xsd:integer` automatically. In SQL the column declares the type for every row; in RDF each literal carries its own type tag, so in principle a predicate could hold an integer on one node and a string on another. Discipline, not the engine, keeps that consistent here.

### Full `music_demo.ttl`

```turtle
@prefix mu:   <http://example.org/music/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

# ── Genre concepts (SKOS controlled vocabulary) ───────────────────────────────

mu:Rock a skos:Concept ;
    skos:prefLabel "rock" ;
    skos:altLabel  "rock and roll" ,
                   "rock music" .

mu:HipHop a skos:Concept ;
    skos:prefLabel "hip-hop" ;
    skos:altLabel  "rap" ,
                   "hip hop" .

mu:Jazz a skos:Concept ;
    skos:prefLabel "jazz" ;
    skos:altLabel  "jazz music" .

mu:Electronic a skos:Concept ;
    skos:prefLabel "electronic" ;
    skos:altLabel  "EDM" ,
                   "electronica" .

mu:RnB a skos:Concept ;
    skos:prefLabel "R&B" ;
    skos:altLabel  "rhythm and blues" ,
                   "soul" .

# ── Albums ────────────────────────────────────────────────────────────────────

mu:nevermind a mu:Album ;
    mu:title       "Nevermind" ;
    mu:artist      "Nirvana" ;
    mu:genre       mu:Rock ;
    mu:releaseYear 1991 ;
    mu:label       "DGC Records" .

mu:darkside a mu:Album ;
    mu:title       "The Dark Side of the Moon" ;
    mu:artist      "Pink Floyd" ;
    mu:genre       mu:Rock ;
    mu:releaseYear 1973 ;
    mu:label       "Harvest Records" .

mu:okcomputer a mu:Album ;
    mu:title       "OK Computer" ;
    mu:artist      "Radiohead" ;
    mu:genre       mu:Rock ;
    mu:releaseYear 1997 ;
    mu:label       "Parlophone" .

mu:abbeyroad a mu:Album ;
    mu:title       "Abbey Road" ;
    mu:artist      "The Beatles" ;
    mu:genre       mu:Rock ;
    mu:releaseYear 1969 ;
    mu:label       "Apple Records" .

mu:ledzeppeliniv a mu:Album ;
    mu:title       "Led Zeppelin IV" ;
    mu:artist      "Led Zeppelin" ;
    mu:genre       mu:Rock ;
    mu:releaseYear 1971 ;
    mu:label       "Atlantic Records" .

mu:thechronic a mu:Album ;
    mu:title       "The Chronic" ;
    mu:artist      "Dr. Dre" ;
    mu:genre       mu:HipHop ;
    mu:releaseYear 1992 ;
    mu:label       "Death Row Records" .

mu:topimpabutterfly a mu:Album ;
    mu:title       "To Pimp a Butterfly" ;
    mu:artist      "Kendrick Lamar" ;
    mu:genre       mu:HipHop ;
    mu:releaseYear 2015 ;
    mu:label       "Top Dawg Entertainment" .

mu:takecare a mu:Album ;
    mu:title       "Take Care" ;
    mu:artist      "Drake" ;
    mu:genre       mu:HipHop ;
    mu:releaseYear 2011 ;
    mu:label       "Young Money / Cash Money" .

mu:theblueprint a mu:Album ;
    mu:title       "The Blueprint" ;
    mu:artist      "Jay-Z" ;
    mu:genre       mu:HipHop ;
    mu:releaseYear 2001 ;
    mu:label       "Roc-A-Fella Records" .

mu:kindofblue a mu:Album ;
    mu:title       "Kind of Blue" ;
    mu:artist      "Miles Davis" ;
    mu:genre       mu:Jazz ;
    mu:releaseYear 1959 ;
    mu:label       "Columbia Records" .

mu:alovesupreme a mu:Album ;
    mu:title       "A Love Supreme" ;
    mu:artist      "John Coltrane" ;
    mu:genre       mu:Jazz ;
    mu:releaseYear 1965 ;
    mu:label       "Impulse! Records" .

mu:randomaccess a mu:Album ;
    mu:title       "Random Access Memories" ;
    mu:artist      "Daft Punk" ;
    mu:genre       mu:Electronic ;
    mu:releaseYear 2013 ;
    mu:label       "Columbia Records" .

mu:discovery a mu:Album ;
    mu:title       "Discovery" ;
    mu:artist      "Daft Punk" ;
    mu:genre       mu:Electronic ;
    mu:releaseYear 2001 ;
    mu:label       "Virgin Records" .

mu:thriller a mu:Album ;
    mu:title       "Thriller" ;
    mu:artist      "Michael Jackson" ;
    mu:genre       mu:RnB ;
    mu:releaseYear 1982 ;
    mu:label       "Epic Records" .

mu:purplerain a mu:Album ;
    mu:title       "Purple Rain" ;
    mu:artist      "Prince" ;
    mu:genre       mu:RnB ;
    mu:releaseYear 1984 ;
    mu:label       "Warner Bros. Records" .
```

> **From the relational world** — Scan the file and notice what's *absent*: there is no `CREATE TABLE albums (...)`, no column list, no `NOT NULL`, no `PRIMARY KEY` declaration, no `FOREIGN KEY ... REFERENCES genres`. The data is the whole file. The structure you'd normally pin down in DDL is here only as a *convention the author followed consistently* — every album happens to have the same five predicates, but nothing enforced that. This is the freedom (and the footgun) of schema-optional data.

---

## Section 2 — Query 1: All Albums

> **From the relational world** — SPARQL keeps SQL's outer shape — `SELECT … WHERE … ORDER BY` — so the skeleton feels familiar. What changes is the `WHERE` clause. In SQL the `WHERE` filters rows from a named table. In SPARQL the `WHERE` is a **graph pattern**: a little template of triples with variables in the blanks, and the engine returns every way the data can fill those blanks. You don't name a table; you describe the shape of the facts you want.

### Line-by-line explanation

```sparql
PREFIX mu:   <http://example.org/music/>
```
Tells the SPARQL engine to expand `mu:` to the full IRI. Must match the `@prefix` in the TTL exactly.

> **Note** — In Turtle the directive is `@prefix` *with* a trailing dot; in SPARQL it's `PREFIX` *without* one. Same idea, slightly different spelling — an easy thing to mix up.

```sparql
SELECT ?title ?artist ?year WHERE {
```
`SELECT` lists the variables to return — SPARQL columns. `WHERE {` opens the graph pattern block.

> **From the relational world** — A `?variable` is a blank to be filled, and it doubles as the column header in the output. The crucial trick: **reusing the same variable name in two places is the join.** There's no `ON a.genre_id = g.id`; you just write `?genre` in both spots and matching values are forced to line up (you'll see this in Q2).

```sparql
    ?album a mu:Album ;
```
`?album` is a variable that will bind to every IRI typed as `mu:Album`. The `;` continues adding predicates for the same subject without repeating `?album`.

> **From the relational world** — This single line is your `FROM albums`. "The table" is reconstructed on the fly as "every subject that has the fact `a mu:Album`." Because membership is just a triple, your `FROM` is really a `WHERE` in disguise.

```sparql
           mu:title       ?title ;
           mu:artist      ?artist ;
           mu:releaseYear ?year .
```
Each line reads one property off `?album` into a result variable. The `.` ends the triple pattern group.

> **From the relational world** — Here's a subtlety with no SQL parallel: each of these lines is a *required* pattern. If some album were missing a `mu:releaseYear` triple, that album would **drop out of the results entirely** — not appear with a NULL year. A missing triple removes the whole match. To get SQL-style "show it anyway, with a blank," you'd wrap the optional part in `OPTIONAL { … }`, which is SPARQL's `LEFT JOIN`.

```sparql
}
ORDER BY ?year
```
Sorts results by release year (ascending). SPARQL `ORDER BY` works on any bound variable.

**Expected result:** 15 rows, one per album, sorted oldest-first.

### Full Q1

```sparql
PREFIX mu:   <http://example.org/music/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT ?title ?artist ?year WHERE {
    ?album a mu:Album ;
           mu:title       ?title ;
           mu:artist      ?artist ;
           mu:releaseYear ?year .
}
ORDER BY ?year
```

---

## Section 3 — Query 2: Rock Albums via SKOS Disambiguation

> **From the relational world** — This is the query that shows why anyone bothers with a graph. The goal: find rock albums *whether the genre was tagged "rock," "rock and roll," or "rock music."* In a relational schema you'd join `albums → genres → genre_synonyms` and search the synonym column. SPARQL does the same join, but the multi-table hop collapses into one tidy property-path line.

### Line-by-line explanation

```sparql
SELECT DISTINCT ?title ?label WHERE {
```
`DISTINCT` removes duplicate rows. Without it, an album whose genre matches both `prefLabel "rock"` and `altLabel "rock music"` would appear twice — once per matching label. `?label` is kept in the projection so the instructor can show *which* label triggered the match.

> **From the relational world** — `DISTINCT` behaves exactly like SQL's. It's needed here for the same reason a one-to-many join inflates row counts: one album related to three matching labels yields three rows.

```sparql
    ?album a mu:Album ;
           mu:title ?title ;
           mu:genre ?genre .
```
Binds `?genre` to the genre concept IRI (e.g., `mu:Rock`).

> **From the relational world** — `?genre` is the join key. It's bound to the genre *node* (`mu:Rock`), and reusing it on the next line is what connects an album to that genre's labels — the equivalent of `JOIN genres ON albums.genre_id = genres.id`.

```sparql
    ?genre (skos:prefLabel|skos:altLabel) ?label .
```
This is a **SPARQL 1.1 property path**. The `|` operator means "either predicate." A single triple pattern traverses both `skos:prefLabel` and `skos:altLabel` — the engine returns a row for every label it finds on `?genre`. This is the key mechanism that lets one query find albums whether the genre was tagged with the preferred term or any synonym.

> **From the relational world** — A property path is the part with no clean SQL twin. `(skos:prefLabel|skos:altLabel)` says "follow an edge that is *either* of these predicates." It's like `UNION`-ing two joins (one to the preferred-label column, one to the synonyms table) into a single step. Property paths also do things SQL needs recursive CTEs for: `foaf:knows+` would mean "one or more `knows` hops" (friends-of-friends to any depth). Here we only need the simple "either/or" form.

```sparql
    FILTER(CONTAINS(LCASE(STR(?label)), "rock"))
```
`STR(…)` strips the datatype tag to get a plain string.
`LCASE(…)` lowercases it so the comparison is case-insensitive.
`CONTAINS(…, "rock")` returns `true` if the string contains the substring `"rock"`.
Together this matches `"rock"`, `"rock and roll"`, and `"rock music"`.

> **From the relational world** — `FILTER` is the row-level half of SQL's `WHERE` (the part that compares values, as opposed to the part that joins). This nested call is just `WHERE LOWER(label) LIKE '%rock%'`. The extra `STR(...)` exists because an RDF literal carries a datatype/language tag; `STR` peels that off to leave a bare string to test.

**Expected result:** 15 rows — the 5 rock albums (Nevermind, The Dark Side of the Moon, OK Computer, Abbey Road, Led Zeppelin IV) each appear **three** times, once per matching SKOS label ("rock", "rock and roll", "rock music"). `DISTINCT` removes same-row duplicates but keeps distinct *(title, label)* pairs. This is the key teaching moment: the same album is findable whether the user searches "rock" or "rock and roll" — SKOS makes those synonyms transparent.

### Full Q2

```sparql
PREFIX mu:   <http://example.org/music/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?title ?label WHERE {
    ?album a mu:Album ;
           mu:title ?title ;
           mu:genre ?genre .
    ?genre (skos:prefLabel|skos:altLabel) ?label .
    FILTER(CONTAINS(LCASE(STR(?label)), "rock"))
}
ORDER BY ?title
```

---

## Section 4 — Query 3: Post-2000 Hip-Hop / Rap Albums

> **From the relational world** — Same join-via-property-path as Q2, now with two filters stacked: a numeric date condition and a synonym match. The point of interest is how naturally a typed integer comparison sits next to a string search.

### Line-by-line explanation

```sparql
SELECT DISTINCT ?title ?year WHERE {
```
`DISTINCT` again prevents duplicates when a genre concept has multiple matching labels.

```sparql
    ?album a mu:Album ;
           mu:title       ?title ;
           mu:genre       ?genre ;
           mu:releaseYear ?year .
```
Binds all four variables from the album node.

```sparql
    ?genre (skos:prefLabel|skos:altLabel) ?label .
```
Same property path as Q2 — traverse both label predicates to collect every SKOS label for the genre.

```sparql
    FILTER(?year >= 2000)
```
**Year filter.** `?year` is bound to an `xsd:integer` literal. The comparison `>= 2000` uses SPARQL's built-in numeric ordering. No casting or string parsing needed.

> **From the relational world** — This works *because* the Turtle author wrote `1991` without quotes, so it parsed as `xsd:integer` rather than `xsd:string`. Had the years been quoted (`"1991"`), `?year >= 2000` would do string comparison and give wrong answers — the RDF echo of storing a date in a `VARCHAR` column. Datatypes still matter; they're just attached to each value instead of declared once per column.

```sparql
    FILTER(
        CONTAINS(LCASE(STR(?label)), "hip") ||
        CONTAINS(LCASE(STR(?label)), "rap")
    )
```
**Genre filter with OR.** The `||` operator is SPARQL's logical OR. This matches any label containing `"hip"` (catches `"hip-hop"` and `"hip hop"`) or `"rap"`. Either condition alone is sufficient.

> **From the relational world** — Plain boolean `OR` inside a `WHERE`, identical in spirit to `WHERE (label LIKE '%hip%' OR label LIKE '%rap%')`. Stacking two separate `FILTER` lines (the year one and this one) is an implicit `AND` — multiple filters in the same pattern block all must hold.

**Expected result:** 4 hip-hop albums released from 2000 onward (The Blueprint 2001, Take Care 2011, To Pimp a Butterfly 2015, and one more depending on ordering). *The Chronic* (1992) is correctly excluded by the year filter.

### Full Q3

```sparql
PREFIX mu:   <http://example.org/music/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?title ?year WHERE {
    ?album a mu:Album ;
           mu:title       ?title ;
           mu:genre       ?genre ;
           mu:releaseYear ?year .
    ?genre (skos:prefLabel|skos:altLabel) ?label .
    FILTER(?year >= 2000)
    FILTER(
        CONTAINS(LCASE(STR(?label)), "hip") ||
        CONTAINS(LCASE(STR(?label)), "rap")
    )
}
ORDER BY ?year
```

---

## Section 5 — Python Script (`music_demo_queries.py`)

> **From the relational world** — There are two ways to run SPARQL here, and the split mirrors "embedded database" vs. "client/server":
>
> - **rdflib** parses the Turtle into an *in-memory* graph and queries it in-process — like SQLite running inside your app, no server.
> - **Fuseki** is a standalone triplestore you talk to over HTTP — like Postgres or MySQL listening on a port. `SPARQLWrapper` is the client driver.
>
> Running the same query both ways is the cross-check in Section 6.

### 5.1 Imports and constants

```python
from pathlib import Path
import requests
import rdflib
from SPARQLWrapper import JSON, SPARQLWrapper
```

| Import | Role |
|--------|------|
| `pathlib.Path` | Cross-platform file paths. `Path(__file__).parent` always resolves to the directory containing this script, regardless of where you run it from. |
| `requests` | HTTP client used to call the Fuseki admin API (create dataset, POST data). |
| `rdflib` | Python RDF library for parsing Turtle locally and running SPARQL without a server. |
| `SPARQLWrapper, JSON` | Thin wrapper around the SPARQL Protocol. Sends queries to Fuseki's HTTP endpoint and returns results as Python dicts. |

```python
DEMO_TTL    = Path(__file__).parent / "music_demo.ttl"
FUSEKI_BASE = "http://localhost:3030"
DATASET     = "music"
ENDPOINT    = f"{FUSEKI_BASE}/{DATASET}/sparql"
```

`ENDPOINT` is the SPARQL 1.1 query endpoint that Fuseki exposes for the `music` dataset. All `SPARQLWrapper` calls go here.

> **From the relational world** — `ENDPOINT` is the rough equivalent of a database connection string / DSN: it's the address the client sends queries to. The difference is that it's an ordinary HTTP URL, and SPARQL endpoints are commonly exposed on the open web — there's a whole ecosystem of public ones (e.g. Wikidata's), which is far less common for raw SQL ports.

### 5.2 Query strings

```python
PREFIXES = """
PREFIX mu:   <http://example.org/music/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""
```

Defined once and prepended to every query so the namespace expansions are never repeated.

`Q1`, `Q2`, `Q3` are plain multi-line strings. Storing queries as module-level constants makes them easy to inspect in a REPL and reuse across both the `rdflib` and `SPARQLWrapper` execution paths.

### 5.3 Helper functions

```python
def load_local_graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(DEMO_TTL), format="turtle")
    return g
```

`rdflib.Graph()` creates an empty in-memory graph.
`.parse(…, format="turtle")` reads the TTL file and populates the graph.
Returns the populated graph for reuse — parsing is the slow step, so we do it once.

> **From the relational world** — `.parse()` is the moment the text file becomes a queryable structure — like loading a dump into a fresh database. After this call the graph is a live set of triples in RAM, and crucially, **if the Turtle is malformed, this is where it fails** — before any network call. That early failure is exactly what makes rdflib useful as a syntax validator in the cross-check.

```python
def run_local(graph: rdflib.Graph, query: str) -> list:
    return list(graph.query(query))
```

`graph.query(query)` runs SPARQL directly against the in-memory graph using rdflib's built-in SPARQL 1.1 engine. No network call. `list(…)` materializes the lazy result iterator into a Python list so we can measure its length.

```python
def _create_fuseki_dataset() -> None:
    r = requests.post(
        f"{FUSEKI_BASE}/$/datasets",
        data={"dbName": DATASET, "dbType": "tdb2"},
        auth=("admin", "admin"),
    )
    if r.status_code not in (200, 409):
        r.raise_for_status()
```

> **From the relational world** — This is `CREATE DATABASE` done over HTTP. `tdb2` is Fuseki's persistent on-disk store (its storage engine, loosely like InnoDB). The `409` check means "already exists, that's fine" — an idempotent create, so re-running the script doesn't error.

```python
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
```

`_create_fuseki_dataset()` — calls `POST /$/datasets` to create the TDB2 persistent store (silently skips a 409 if it already exists).
`DEMO_TTL.read_text(…)` — reads the entire TTL file into a string.
`requests.post(…, headers={"Content-Type": "text/turtle"})` — sends the Turtle body to Fuseki's **Graph Store Protocol** (GSP) endpoint. GSP is the standard HTTP interface for loading RDF into a named graph or the default graph.
`.raise_for_status()` — converts any 4xx/5xx HTTP error into a Python exception.

> **From the relational world** — This is the bulk load — the `LOAD DATA INFILE` / `COPY` step. The **Graph Store Protocol** is a W3C-standardized HTTP way to push RDF into a store; because it's a standard, the same `POST` works against any compliant triplestore, not just Fuseki. Contrast that with bulk-load syntax, which is vendor-specific in the SQL world.

```python
def run_fuseki(query: str) -> list[dict]:
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]
```

`SPARQLWrapper(ENDPOINT)` — points the wrapper at Fuseki's SPARQL endpoint.
`.setQuery(query)` — stages the query string.
`.setReturnFormat(JSON)` — asks Fuseki to reply with SPARQL JSON results.
`.query().convert()` — sends the HTTP GET and parses the JSON response.
`["results"]["bindings"]` — digs into the SPARQL JSON envelope to get the array of result rows.

> **From the relational world** — `bindings` is the result set; each entry is one "row," but shaped as a dict of `{variable: {value, type, datatype}}` rather than a flat tuple. The extra nesting is there precisely because an RDF value isn't just a string — it knows whether it's a literal, an IRI, what datatype it carries, and so on.

### 5.4 Main demo

```python
def main() -> None:
    g = load_local_graph()

    print("\n=== Q1: All Albums ===")
    for row in run_local(g, Q1):
        print(f"  {row.title:<35}  {row.artist:<25}  ({row.year})")
```

`run_local` returns a list of `rdflib.query.ResultRow` objects. Attribute access (`row.title`) works because rdflib binds SPARQL variable names as Python attributes. The `:<35` format specifier left-pads the column to 35 characters for readable output.

```python
    print("\n=== Q2: Rock / Rock-and-Roll Albums (SKOS) ===")
    for row in run_local(g, Q2):
        print(f"  {row.title:<35}  matched label: '{row.label}'")
```

Printing `row.label` shows *which* SKOS label triggered the match — critical for the demo because it makes the disambiguation visible ("rock and roll" vs "rock").

```python
    local_count  = len(run_local(g, Q1))
    fuseki_count = len(run_fuseki(Q1))
    if local_count == fuseki_count:
        print("  PASS — counts match, cross-check successful.")
```

The cross-check runs Q1 against both engines and compares row counts. If they differ, the data load failed or the Turtle was malformed. This pattern is the quickest sanity-check before a demo.

### Full `music_demo_queries.py`

```python
#!/usr/bin/env python3
"""
SI Demo — Music Knowledge Graph
Demonstrates TTL authoring, Fuseki stand-up, three SPARQL queries with
SKOS disambiguation and a year FILTER, and an rdflib-vs-Fuseki cross-check.
"""

from pathlib import Path

import requests
import rdflib
from SPARQLWrapper import JSON, SPARQLWrapper

DEMO_TTL    = Path(__file__).parent / "music_demo.ttl"
FUSEKI_BASE = "http://localhost:3030"
DATASET     = "music"
ENDPOINT    = f"{FUSEKI_BASE}/{DATASET}/sparql"

PREFIXES = """
PREFIX mu:   <http://example.org/music/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

Q1 = PREFIXES + """
SELECT ?title ?artist ?year WHERE {
    ?album a mu:Album ;
           mu:title       ?title ;
           mu:artist      ?artist ;
           mu:releaseYear ?year .
}
ORDER BY ?year
"""

Q2 = PREFIXES + """
SELECT DISTINCT ?title ?label WHERE {
    ?album a mu:Album ;
           mu:title ?title ;
           mu:genre ?genre .
    ?genre (skos:prefLabel|skos:altLabel) ?label .
    FILTER(CONTAINS(LCASE(STR(?label)), "rock"))
}
ORDER BY ?title
"""

Q3 = PREFIXES + """
SELECT DISTINCT ?title ?year WHERE {
    ?album a mu:Album ;
           mu:title       ?title ;
           mu:genre       ?genre ;
           mu:releaseYear ?year .
    ?genre (skos:prefLabel|skos:altLabel) ?label .
    FILTER(?year >= 2000)
    FILTER(
        CONTAINS(LCASE(STR(?label)), "hip") ||
        CONTAINS(LCASE(STR(?label)), "rap")
    )
}
ORDER BY ?year
"""


def load_local_graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(DEMO_TTL), format="turtle")
    return g


def run_local(graph: rdflib.Graph, query: str) -> list:
    return list(graph.query(query))


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
    print(f"Loaded music_demo.ttl → Fuseki  (HTTP {r.status_code})")


def run_fuseki(query: str) -> list[dict]:
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]


def main() -> None:
    print("Parsing music_demo.ttl with rdflib ...")
    g = load_local_graph()

    print("\n=== Q1: All Albums ===")
    for row in run_local(g, Q1):
        print(f"  {row.title:<35}  {row.artist:<25}  ({row.year})")

    print("\n=== Q2: Rock / Rock-and-Roll Albums (SKOS) ===")
    for row in run_local(g, Q2):
        print(f"  {row.title:<35}  matched label: '{row.label}'")

    print("\n=== Q3: Hip-Hop / Rap Albums Released >= 2000 ===")
    for row in run_local(g, Q3):
        print(f"  {row.title:<35}  ({row.year})")

    print("\n─── Fuseki Cross-check ───")
    load_into_fuseki()

    local_count  = len(run_local(g, Q1))
    fuseki_count = len(run_fuseki(Q1))
    print(f"  rdflib  → {local_count} albums")
    print(f"  Fuseki  → {fuseki_count} albums")

    if local_count == fuseki_count:
        print("  PASS — counts match, cross-check successful.")
    else:
        print("  FAIL — count mismatch, check the data load.")


if __name__ == "__main__":
    main()
```

---

## Section 6 — rdflib vs. Fuseki Cross-Check

### Why cross-check?

| Engine | What it validates |
|--------|-------------------|
| **rdflib** | The Turtle file itself — if rdflib can parse and query it, the syntax is correct. |
| **Fuseki** | The full pipeline — Docker container running, admin API reachable, GSP upload successful, SPARQL endpoint returning results. |

Running the same query through both engines and comparing row counts catches three failure modes early:
1. **Bad TTL** — rdflib parse error before any network call.
2. **Load failure** — Fuseki returns 200 but the data wasn't actually stored (e.g., wrong endpoint or content-type).
3. **Query engine divergence** — an edge case in property-path handling that one engine resolves differently (rare, but good to rule out).

> **From the relational world** — There's no everyday SQL habit quite like this, because you normally have *one* engine. The closest parallel is running your test suite against SQLite locally and Postgres in CI to catch dialect differences. RDF/SPARQL are W3C standards, so two independent engines (rdflib and Fuseki) *should* agree exactly — and when they don't, the disagreement is a real signal worth chasing.

### Cross-check pattern

```python
# 1. Load TTL into rdflib in-memory graph
g = load_local_graph()

# 2. Run Q1 locally — no Fuseki required
local_results = run_local(g, Q1)

# 3. Load the same TTL into Fuseki
load_into_fuseki()

# 4. Run the identical Q1 against Fuseki's SPARQL endpoint
fuseki_results = run_fuseki(Q1)

# 5. Compare
assert len(local_results) == len(fuseki_results), (
    f"Mismatch: rdflib={len(local_results)}, Fuseki={len(fuseki_results)}"
)
```

If the assert passes you have confidence that both the file and the server pipeline are healthy. If it fails, start by checking `r.status_code` in `load_into_fuseki` and confirming the Fuseki admin UI at `http://localhost:3030` shows the dataset with the expected triple count.

---

## Quick-Start Checklist

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Fuseki
docker compose up -d

# 3. Load the recipes dataset
python load_dataset.py

# 4. Run the test suite
pytest tests/ -v

# 5. Run the SI demo (music domain)
cd music_demo
python music_demo_queries.py
```

---

## Appendix A — Glossary for Relational Developers

A one-line "if you already know SQL" gloss for each new term.

| Term | One-line meaning | Nearest SQL idea |
|------|------------------|------------------|
| **Triple** | A single fact: subject–predicate–object. | One cell, expressed as a standalone sentence. |
| **Graph** | The whole set of triples; nodes joined by labeled edges. | The entire database, but with no table walls. |
| **RDF** | The data model triples belong to. | The relational model — but graph-shaped. |
| **Turtle (`.ttl`)** | A text format for writing RDF. | An `INSERT`-style dump, not the `CREATE TABLE` part. |
| **IRI** | A globally-unique URL used as an identifier. | A primary key that's a URL and unique worldwide. |
| **`@prefix` / `PREFIX`** | A short alias for a long IRI base. | A schema qualifier like `dbo.`, just expandable. |
| **Predicate** | The relationship/property in a triple. | A column name — but global, not table-scoped. |
| **Literal** | A plain value (string, number, date). | A scalar cell value. |
| **Object property** | A predicate whose value is a link to another node. | A foreign key — except the link *is* the value. |
| **Class / `a` / `rdf:type`** | A node's declared type, stated as a triple. | Table membership, but assertable as data. |
| **Datatype (`xsd:integer`…)** | A type tag attached to a literal. | A column type, but per-value instead of per-column. |
| **Triplestore** | A database that stores triples. | The RDBMS engine (here, Fuseki). |
| **SPARQL** | The query language for RDF. | SQL. |
| **Graph pattern** | The triple template inside `WHERE {}`. | The `FROM` + `JOIN` + join-`WHERE`, fused. |
| **`?variable`** | A blank to bind; reused names = a join. | A column alias that also performs the join. |
| **Property path** | Multi-predicate / multi-hop edge traversal. | `UNION`ed joins, or a recursive CTE. |
| **`OPTIONAL`** | Pattern that may be absent without dropping the row. | `LEFT JOIN`. |
| **`FILTER`** | Row-level value test inside a pattern. | The comparison part of `WHERE`. |
| **SKOS** | W3C vocabulary for terms + synonyms. | A lookup table joined to a synonyms table. |
| **`skos:prefLabel`** | The one canonical name of a concept. | The canonical-name column. |
| **`skos:altLabel`** | A synonym/variant of a concept. | A row in the synonyms table. |
| **Graph Store Protocol (GSP)** | Standard HTTP way to load RDF into a store. | `LOAD DATA` / `COPY`, but vendor-neutral. |
| **Open-world assumption** | "Not stated" ≠ "false." | The opposite of SQL's closed-world default. |
| **Schema-optional** | Data is valid and queryable with no schema. | The opposite of mandatory `CREATE TABLE` first. |
