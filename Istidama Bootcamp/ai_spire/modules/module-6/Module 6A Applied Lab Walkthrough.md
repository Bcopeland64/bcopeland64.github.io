# Module 6A Applied Lab — Code Walkthrough

This document walks every line of [module6_ner_lab.ipynb](module6_ner_lab.ipynb). Each section starts with a line-by-line explanation, then shows the full code block at the end so you can copy-paste it cleanly.

The lab extracts structured entity data from ten unstructured public-health advisory excerpts (WHO, CDC, UNICEF, and ministry-level bulletins) using five different NER systems — spaCy, Hugging Face BERT, NLTK's classical MaxEnt chunker, a Regex + Gazetteer baseline, and Flair — then scores every system against a hand-annotated gold standard using strict Precision / Recall / F1. Read alongside the notebook.

---

## 1. Load Data and Orient

### Why this section exists
Every downstream step operates on this same list of ten raw strings. Loading it first, and printing one example, gives learners a chance to read the text and manually spot entities before the machine tries — a baseline for comparison once the automated systems run (the notebook's own "pause and reflect" prompt asks exactly this).

### Line-by-line

- `texts = [ ... ]` — a Python list of 10 raw, unmodified sentences describing health bulletins (WHO/H5N1 in Southeast Asia, Jordan's vaccination campaign, King Hussein Medical Center, dengue surveillance, UNICEF polio vaccine distribution, a Global Fund malaria grant, an RSV vaccine trial, Sheikh Khalifa Medical City, and EU measles figures). The strings are left completely unmodified — original casing, punctuation, and numerals intact — because that is the raw material every NER system in this lab will consume directly. Each sentence was deliberately written to be dense with multiple entity types (`ORG`, `GPE`, `DATE`, `CARDINAL`, `MONEY`, `PERSON`, `FACILITY`) so that every system has enough surface area to succeed or fail visibly.
- `print(f"Loaded {len(texts)} texts.\n")` — confirms the list loaded with the expected count (10) and prints a blank line via the `\n` inside the f-string for visual spacing before the next print.
- `print("First text:\n")` — a small section label so the printed sentence below it is clearly identified in the cell output.
- `print(texts[0])` — prints the first sentence (the WHO/H5N1 sentence) in full. This is the sentence the rest of Sections 2–4 use repeatedly as the running example, since a single sentence viewed through multiple lenses (preprocessing stages, spaCy, HF BERT) makes it easy to compare outputs directly.

### Full code block

```python
texts = [
    "The WHO declared a public health emergency of international concern following reports of 2,400 confirmed cases of avian influenza H5N1 in Southeast Asia between January and March 2025.",
    "The Ministry of Health in Amman announced that Jordan's national vaccination campaign reached 3.2 million residents across all twelve governorates by the end of Q2.",
    "Dr. Al-Rashidi, acting director of the Eastern Mediterranean Regional Office, confirmed that the CDC and WHO are coordinating a joint response team to be deployed to affected areas by next Friday.",
    "Royal Medical Services reported that King Hussein Medical Center in Amman admitted 340 patients with respiratory symptoms in a single week, exceeding normal capacity by 40 percent.",
    "The Pan American Health Organization issued updated guidelines for dengue surveillance in the Caribbean after a 60 percent year-over-year increase in reported cases across 12 countries.",
    "UNICEF distributed 500,000 doses of oral polio vaccine to health workers in northern Syria, targeting children under five in Aleppo and Idlib governorates.",
    "The Global Fund approved a USD 4.2 billion allocation for malaria prevention programs across sub-Saharan Africa, with implementation beginning in Senegal, Nigeria, and the Democratic Republic of Congo.",
    "The National Institute of Allergy and Infectious Diseases published interim results from a Phase III clinical trial of the novel RSV vaccine, showing 78 percent efficacy in adults over 60.",
    "Abu Dhabi's Department of Health confirmed that the new Sheikh Khalifa Medical City expansion will add 200 beds and a dedicated infectious disease unit by late 2026.",
    "The European Centre for Disease Prevention and Control reported 15,000 cases of measles across the EU in 2025, with Romania, Italy, and France accounting for 70 percent of cases.",
]

print(f"Loaded {len(texts)} texts.\n")
print("First text:\n")
print(texts[0])
```

---

## 2. Preprocess a Single Text

### Why this section exists
The notebook makes a deliberate, non-obvious point here: **preprocessing does not precede NER.** NER needs original casing and punctuation to recognize proper nouns; preprocessing (normalization, lowercasing, lemmatization) is only useful for separate text-analysis tasks like word-frequency counts. This section walks through three preprocessing stages on one sentence specifically to demonstrate *why* running NER on preprocessed text would be a mistake, before Section 3 runs NER on the untouched original.

### 2a. Setup — imports and spaCy model

#### Why this section exists
Both the preprocessing demo and the NER pass in Section 3 need a loaded spaCy pipeline, so it's loaded once here and reused throughout the notebook.

#### Line-by-line

- `import unicodedata` — Python's standard-library Unicode database module. It supplies `unicodedata.normalize()`, used in the next cell to canonicalize Unicode text before any further processing.
- `import spacy` — the spaCy NLP library's entry point, used to load a pretrained pipeline and to process text into `Doc` objects.
- `nlp = spacy.load("en_core_web_sm")` — loads spaCy's small English pipeline. `en_core_web_sm` bundles a tokenizer, part-of-speech tagger, dependency parser, lemmatizer, and a statistical NER component, and is described as "CPU-friendly" because at ~12 MB it runs comfortably without a GPU. This call must find the model already installed (`python -m spacy download en_core_web_sm`) or it raises an `OSError`; the resulting `nlp` object is callable — `nlp(text)` — and is reused everywhere else in the notebook that needs spaCy.

#### Full code block

```python
import unicodedata
import spacy

# Load the spaCy English model (small, CPU-friendly)
nlp = spacy.load("en_core_web_sm")
```

### 2b. Stage 1 — NFC normalization

#### Why this section exists
Shows the first, most invisible preprocessing stage: Unicode canonicalization. It matters little for clean ASCII text but is essential for real-world text copied from PDFs or non-Latin scripts, which this bootcamp's learners will likely encounter.

#### Line-by-line

- `text = texts[0]` — pulls the running example sentence into a short-named local variable for the rest of this section.
- `print("Original text:")` / `print(text)` — displays the untouched sentence as a baseline for comparison against the normalized version below.
- `text_norm = unicodedata.normalize("NFC", text)` — applies **Normalization Form C (Canonical Composition)**: it converts any character that is spread across a base codepoint plus a separate combining diacritic mark into a single, precomposed codepoint (when one exists). Two strings that look visually identical can have different underlying byte sequences if one uses composed characters and the other uses decomposed ones; NFC collapses that discrepancy so string comparisons and searches behave consistently. This is invisible for plain ASCII English text (as this sentence is) but becomes critical for text with diacritics or Arabic-script ligatures pasted from Word or PDF documents, where inconsistent normalization silently breaks exact-match lookups (including the gazetteer matching used later in Section 9).
- `print("\nAfter NFC normalization:")` / `print(text_norm)` — prints the normalized string; for this ASCII sentence it will look byte-for-byte identical to the original, which is itself the teaching point — the transformation is real but not visually detectable on this input.
- The trailing comments explain that the effect only becomes visible on text with composed/decomposed diacritics or ligatures, which this example intentionally does not contain.

#### Full code block

```python
# --- Stage 1: Unicode NFC normalization ---
# NFC collapses visually identical characters that have different byte representations.
# Critical for text copied from PDFs or Arabic-script documents.

text = texts[0]
print("Original text:")
print(text)

text_norm = unicodedata.normalize("NFC", text)
print("\nAfter NFC normalization:")
print(text_norm)

# For clean ASCII English text you will see no visible difference.
# The difference matters when text contains composed/decomposed diacritics or
# Arabic ligatures copied from Word documents or PDFs.
```

### 2c. Stage 2 — Lowercase

#### Why this section exists
This is the section that makes the "preprocessing destroys NER signal" argument concrete: lowercasing collapses `WHO` (an organization) and `who` (a pronoun) into the same string, deleting the exact signal spaCy's statistical NER model relies on to recognize proper nouns.

#### Line-by-line

- `text_lower = text_norm.lower()` — Python's built-in `str.lower()`, applied to the NFC-normalized text from the previous stage. Lowercasing is a standard step for bag-of-words style analysis (word counts, vocabulary stats) because it treats `"WHO"` and `"who"` as the same token — but that exact behavior is disastrous for NER, since capitalization is one of the strongest signals a statistical or rule-based tagger uses to spot proper nouns.
- `print("Lowercased:")` / `print(text_lower)` — displays the fully lowercased sentence so learners can visually see every acronym and proper noun flattened to lowercase.
- The two trailing `print` calls spell out the specific failure mode: `'WHO'` becomes `'who'`, and if NER ran on this lowercased string, the model would lose the capitalization cue that lets it distinguish the World Health Organization from the pronoun "who." This is the direct justification for why Section 3 runs NER on `texts[0]` (the original), never on `text_lower`.

#### Full code block

```python
# --- Stage 2: Lowercase ---
# Standard for bag-of-words analysis, but DESTROYS casing signals for NER.
# 'WHO' → 'who' (pronoun vs. organization — the model can no longer distinguish)

text_lower = text_norm.lower()
print("Lowercased:")
print(text_lower)

print("\n>>> Note: 'WHO' becomes 'who'. If we ran NER on this, the model loses")
print("    the capitalization signal it uses to identify proper nouns. This is")
print("    why preprocessing order matters.")
```

### 2d. Stage 3 — Tokenize and lemmatize

#### Why this section exists
Shows the legitimate use case for the lowercased text: it's fine — even desirable — for downstream tasks like word-frequency counts or topic modeling, just not for NER. This closes the preprocessing detour before Section 3 switches back to the original text.

#### Line-by-line

- `doc_lower = nlp(text_lower)` — runs the *entire* spaCy pipeline (tokenizer, tagger, parser, lemmatizer, and even the NER component, though its output is unused here) on the **lowercased** text. This is intentional: the point of this cell is lemmatization for text-analysis purposes, not entity extraction.
- `tokens = [token.lemma_ for token in doc_lower if token.is_alpha]` — a list comprehension over every `Token` in the `Doc`. `token.lemma_` is the token's dictionary root form as a string (e.g., `"declared"` → `"declare"`); `token.is_alpha` is a boolean filter that keeps only tokens made entirely of alphabetic characters, dropping punctuation, digits, and whitespace tokens from the result.
- `print("Lemmatized alpha tokens (first 20):")` / `print(tokens[:20])` — displays a slice of the first 20 lemmas so the printed output stays compact.
- The trailing comments give worked examples (`'declared' → 'declare'`, `'cases' → 'case'`, `'confirmed' → 'confirm'`) and state the intended use: clean tokens for frequency analysis or topic modeling — explicitly *not* for NER, reinforcing the section's central lesson one more time before moving on.

#### Full code block

```python
# --- Stage 3: Tokenize and lemmatize with spaCy ---
# We tokenize the LOWERCASED text here purely for analysis purposes.
# spaCy's lemmatizer maps inflected forms to their dictionary root.

doc_lower = nlp(text_lower)
tokens = [token.lemma_ for token in doc_lower if token.is_alpha]

print("Lemmatized alpha tokens (first 20):")
print(tokens[:20])

# 'declared' → 'declare', 'cases' → 'case', 'confirmed' → 'confirm'
# This gives you clean tokens for frequency analysis, topic modelling, etc.
```

---

## 3. spaCy NER with displacy Visualization

### Why this section exists
This is the payoff of Section 2's warning: NER runs on the **original, unmodified** text, never the lowercased version. It also introduces `displacy`, spaCy's built-in visual renderer, which serves as a fast visual QA pass before any programmatic counting or scoring happens later.

### 3a. Run NER and print the entity table

#### Line-by-line

- `from spacy import displacy` — imports spaCy's visualization submodule, used in the next cell.
- `doc = nlp(texts[0])` — runs the spaCy pipeline on the **original** `texts[0]` string (not `text_lower`). This is the `Doc` object every subsequent spaCy call in the notebook is built from the same way.
- `print("Entities found in texts[0]:")` and the header `print` — build a fixed-width column header (`Entity Text`, `Label`, `Span`) using format specifiers (`{'Entity Text':<35}`) so the printed rows below line up visually.
- `print("-" * 60)` — a horizontal rule under the header, purely cosmetic.
- `for ent in doc.ents:` — `doc.ents` is a tuple of `Span` objects, one per entity spaCy's statistical NER component recognized in the text.
- `print(f"{ent.text:<35} {ent.label_:<12} ({ent.start_char}:{ent.end_char})")` — for each entity, prints the entity's surface text (`ent.text`), its label (`ent.label_`, e.g. `ORG`, `GPE`, `DATE`), and its character-offset span (`ent.start_char`, `ent.end_char`) — the index range within the original string where the entity occurs. Left-aligned fixed-width fields (`<35`, `<12`) again keep the table columns readable.

#### Full code block

```python
from spacy import displacy

# Run NER on the original (not preprocessed) text
doc = nlp(texts[0])

print("Entities found in texts[0]:")
print(f"{'Entity Text':<35} {'Label':<12} {'Span'}")
print("-" * 60)
for ent in doc.ents:
    print(f"{ent.text:<35} {ent.label_:<12} ({ent.start_char}:{ent.end_char})")
```

### 3b. displacy render

#### Line-by-line

- The leading comment explains the purpose: `displacy` renders entities inline as colored, labeled highlights directly in the notebook, meant as a quick **visual QA pass** before doing any programmatic counting (like the table above, or the evaluation metrics in Section 7).
- `displacy.render(doc, style="ent", jupyter=True)` — `style="ent"` selects the entity-highlighting visualization mode (as opposed to `style="dep"`, which would draw a dependency-parse tree). `jupyter=True` tells `displacy` to render the HTML directly into the current notebook cell's output rather than returning a raw HTML string that would need to be wrapped separately.

#### Full code block

```python
# displacy renders entities inline in the notebook — use this as your visual QA pass
# before doing any programmatic counting.

displacy.render(doc, style="ent", jupyter=True)
```

### 3c. Loop over more texts

#### Line-by-line

- The leading comment states the goal: process a few more texts to build intuition for how the model behaves across different sentence structures, not just the one running example.
- `for i in [1, 2, 3]:` — iterates over a hard-coded list of three more indices (Jordan's vaccination sentence, the Dr. Al-Rashidi sentence, and the King Hussein Medical Center sentence) — deliberately chosen because each one stresses a different entity type spaCy might get wrong.
- `print(f"\n=== texts[{i}] ===")` — a labeled separator so each text's rendering is clearly demarcated in the output.
- `doc_i = nlp(texts[i])` — runs the pipeline on that text, producing a fresh `Doc`.
- `displacy.render(doc_i, style="ent", jupyter=True)` — renders that text's entities inline, same as 3b.
- `print()` — a blank line for visual spacing between iterations.
- The trailing comment block lists four **observation prompts** for learners to check by eye while reading the rendered output: whether `King Hussein Medical Center` was tagged `FACILITY`, whether `Q2` was tagged `DATE`, whether `Dr. Al-Rashidi` was captured as `PERSON`, and what label spaCy assigned to `Amman` (answer given inline: `GPE`, geopolitical entity). These are the exact edge cases the gold-standard evaluation in Section 7 will later score numerically.

#### Full code block

```python
# Process a few more texts to build intuition for model behavior

for i in [1, 2, 3]:
    print(f"\n=== texts[{i}] ===")
    doc_i = nlp(texts[i])
    displacy.render(doc_i, style="ent", jupyter=True)
    print()

# Observation prompts:
#   - Did the model correctly tag 'King Hussein Medical Center' as a FACILITY?
#   - Did it handle 'Q2' as a DATE?
#   - Did it capture 'Dr. Al-Rashidi' as a PERSON?
#   - What entity type did spaCy assign to 'Amman'? (GPE = geopolitical entity)
```

---

## 4. Hugging Face BERT-NER Pipeline

### Why this section exists
Introduces a second, transformer-based NER system — `dslim/bert-base-NER` — as a contrast to spaCy. This model uses a different label scheme (BIO tags) and a WordPiece tokenizer that can split unfamiliar words into subword pieces, both of which require explicit handling before the output can be compared apples-to-apples with spaCy or scored against the gold standard.

### 4a. Load the pipeline

#### Line-by-line

- `from transformers import pipeline` — imports Hugging Face's high-level `pipeline` factory function, which bundles model loading, tokenization, inference, and (optionally) post-processing behind one callable object.
- The comment explains the key argument choice: `aggregation_strategy="none"` returns **raw, token-level** predictions rather than pre-merged entity spans, because the notebook wants to demonstrate — and handle — subword splitting itself rather than let the pipeline hide it.
- `ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="none")` — `"ner"` selects the token-classification task type. `model="dslim/bert-base-NER"` is a BERT checkpoint fine-tuned on the CoNLL-2003 NER benchmark; on first use it downloads the model weights and tokenizer files from the Hugging Face Hub and caches them locally (subsequent runs load from cache). The returned `ner_pipeline` object is callable — `ner_pipeline(text)` — and, with `aggregation_strategy="none"`, still automatically drops tokens the model tags `"O"` (outside any entity) via the pipeline's default `ignore_labels=["O"]` behavior, so only entity-bearing tokens are returned.
- `print("Pipeline loaded successfully.")` — confirms the (potentially slow, network-dependent) model load completed without error before moving on.

#### Full code block

```python
from transformers import pipeline

# aggregation_strategy="none" gives us raw token-level output so we can
# inspect and handle subword splitting ourselves.
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="none")

print("Pipeline loaded successfully.")
```

### 4b. Run on texts[0]

#### Line-by-line

- `results = ner_pipeline(texts[0])` — runs the loaded pipeline on the same running-example sentence used throughout the notebook. It returns a list of dicts, one per recognized token, each with keys like `word`, `entity` (a BIO-tagged label such as `B-ORG`), `score` (the model's confidence, 0–1), `start`, and `end` (character offsets).
- The `print` header builds another fixed-width column table (`word`, `entity`, `score`, `span`).
- `for r in results:` iterates the raw token dicts and prints each one: `r['word']` (the raw WordPiece token — possibly a subword fragment), `r['entity']` (the BIO label), `r['score']:.3f` (confidence formatted to three decimal places), and the `(start:end)` character span. Because `aggregation_strategy="none"` was used, some of these rows may be subword continuation pieces (like `##F`), which is exactly what the next cell demonstrates explicitly.

#### Full code block

```python
# Run the HF pipeline on texts[0]
results = ner_pipeline(texts[0])

print("Raw HF output for texts[0]:")
print(f"{'word':<20} {'entity':<12} {'score':<8} span")
print("-" * 55)
for r in results:
    print(f"{r['word']:<20} {r['entity']:<12} {r['score']:.3f}    ({r['start']}:{r['end']})")
```

### 4c. Subword splitting demonstration

#### Line-by-line

- `demo_sentence = "The IPCC released its Sixth Assessment Report in Geneva."` — a short sentence chosen deliberately because `IPCC` is an uncommon acronym unlikely to appear whole in BERT's WordPiece vocabulary, making it likely to be split into subword fragments — the exact phenomenon this cell wants to surface.
- `demo_results = ner_pipeline(demo_sentence)` — runs the same pipeline on this new sentence.
- The `print` calls show the input sentence and then loop over `demo_results`, printing each token's `word`, `entity`, and `score` with fixed-width alignment.
- The trailing comment sets the expectation: you should see tokens like `'IP'` followed by `'##CC'` — the `##` prefix is WordPiece's marker for "this piece continues the previous token," not a standalone word. This is the concrete example the next cell's merge function is built to fix.

#### Full code block

```python
# Demonstrate the subword problem explicitly with a short sentence
demo_sentence = "The IPCC released its Sixth Assessment Report in Geneva."
demo_results = ner_pipeline(demo_sentence)

print("Subword splitting demonstration:")
print(f"Input: {demo_sentence}\n")
for r in demo_results:
    print(f"  word={r['word']:<15} entity={r['entity']:<10} score={r['score']:.3f}")

# You should see tokens like 'IP', '##CC' — continuation pieces beginning with ##
```

### 4d. `merge_subword_entities` function

#### Line-by-line

- `def merge_subword_entities(results):` — declares a function that takes the raw list-of-dicts output from the HF pipeline. Its docstring documents the args (`results`: list of dicts from a HF NER pipeline) and return value (list of dicts with `'##'` tokens merged back into the previous entry).
- `merged = []` — the accumulator list that will hold the final, subword-merged entries.
- `for r in results:` — iterates every raw token dict in order (order matters here — WordPiece continuation pieces always immediately follow the token they extend).
- `if r["word"].startswith("##"):` — WordPiece's convention: any token string beginning with `##` is a continuation fragment of the immediately preceding token, not a new word.
  - `if merged:` — a safety guard against a `##`-prefixed token appearing as the very first result (which shouldn't normally happen, but would otherwise cause an `IndexError` on `merged[-1]`).
    - `merged[-1]["word"] += r["word"][2:]` — appends the continuation piece, with its leading `##` stripped off (`r["word"][2:]` slices off the first two characters), onto the `word` field of the most recently added merged entry — reconstructing the full token (e.g., `"IP"` + `"CC"` → `"IPCC"`).
    - `merged[-1]["end"] = r["end"]` — extends the previous entry's end character offset to include this piece, so the merged entry's span still correctly covers the whole reconstructed word.
- `else:` — for any token that is *not* a continuation piece:
  - `merged.append(dict(r))` — appends a **copy** of the token dict (`dict(r)` constructs a shallow copy) rather than the original dict reference. This matters because the function is about to mutate the `word` and `end` fields in place on subsequent iterations, and mutating the original pipeline output would be a surprising side effect for any caller still holding a reference to `results`.
- `return merged` — the final list of whole-word entity dicts, ready for display or evaluation.
- The cell then calls `merged_demo = merge_subword_entities(demo_results)` on the IPCC example and prints each merged entry's `word` and `entity`, so learners can confirm `IP` + `##CC` collapsed back into a single `IPCC` entry.

#### Full code block

```python
def merge_subword_entities(results):
    """
    Merge WordPiece continuation tokens (those starting with '##') back
    into the preceding token entry.

    Args:
        results: list of dicts from a HF NER pipeline

    Returns:
        list of dicts with '##' tokens merged
    """
    merged = []
    for r in results:
        if r["word"].startswith("##"):
            if merged:
                # Append the subword piece (strip the leading ##)
                merged[-1]["word"] += r["word"][2:]
                # Extend the span end to include this piece
                merged[-1]["end"] = r["end"]
        else:
            # Copy to avoid mutating the original pipeline output
            merged.append(dict(r))
    return merged


# Test it on the IPCC demonstration
merged_demo = merge_subword_entities(demo_results)
print("After merging subwords:")
for m in merged_demo:
    print(f"  {m['word']:<20} {m['entity']:<10}")
```

### 4e. Apply merging to texts[0]

#### Line-by-line

- `hf_merged_0 = merge_subword_entities(results)` — applies the merge function (defined in 4d) to the raw `results` captured back in 4b for the running-example sentence, producing whole-word entity entries.
- The `print` calls build another fixed-width table and loop over `hf_merged_0`, printing `word`, `entity`, and `score` for each merged entry — this is the cleaned-up, human-readable version of the raw table from 4b, and the form used for every comparison and evaluation from here forward.

#### Full code block

```python
# Apply merging to texts[0]
hf_merged_0 = merge_subword_entities(results)

print("HF entities for texts[0] after subword merging:")
print(f"{'word':<25} {'entity':<12} {'score'}")
print("-" * 50)
for m in hf_merged_0:
    print(f"{m['word']:<25} {m['entity']:<12} {m['score']:.3f}")
```

---

## 5. Side-by-Side Comparison

### Why this section exists
Neither system's raw output format matches the other's (spaCy uses `Span` objects with labels like `ORG`; HF uses merged dicts with BIO labels like `B-ORG`). This section builds one shared comparison function that normalizes both into `(text, label)` tuples and prints them in parallel columns, so the label-scheme differences shown in the preceding markdown table (`ORG` vs `B-ORG`, `GPE` vs `B-LOC`, etc.) become visible side by side across five example sentences.

### Line-by-line

- `def compare_ner_outputs(text, nlp_model, hf_pipeline):` — a function parameterized by the raw text and both already-loaded model objects, so it can be reused across every text without re-loading either model. Its docstring documents all three arguments.
- `doc = nlp_model(text)` — runs spaCy on the text.
- `spacy_ents = [(ent.text, ent.label_) for ent in doc.ents]` — converts spaCy's `Span` objects into a plain list of `(text, label)` tuples — the same shape used for HF results below, enabling a fair side-by-side print.
- `hf_raw = hf_pipeline(text)` — runs the HF pipeline on the same text, returning raw (possibly subword-split) token dicts.
- `hf_merged = merge_subword_entities(hf_raw)` — reuses the Section 4d helper to reassemble whole-word entities.
- `hf_ents = [(m["word"], m["entity"]) for m in hf_merged]` — converts the merged HF dicts into the same `(text, label)` tuple shape as `spacy_ents`.
- `max_rows = max(len(spacy_ents), len(hf_ents))` — the two systems will almost never find the exact same number of entities in a sentence, so the print loop needs to iterate up to whichever list is longer, without an `IndexError` on the shorter one.
- The `print` header builds two fixed-width columns, `spaCy entity` and `HF entity`.
- `for i in range(max_rows):` — iterates row indices up to `max_rows`.
  - `sp = f"{spacy_ents[i][0]} [{spacy_ents[i][1]}]" if i < len(spacy_ents) else ""` — a conditional expression: if this row index is still within bounds of `spacy_ents`, format the entity as `"text [LABEL]"`; otherwise leave the column blank (an empty string) — this is what keeps mismatched entity counts from crashing the loop.
  - `hf = ...` — the same pattern applied to `hf_ents`.
  - `print(f"{sp:<35} {hf:<35}")` — prints both columns side by side, left-aligned to 35 characters each, so a reader can visually scan for agreement or disagreement row by row.
- `for idx in range(5):` — the outer driver loop, applying the comparison to the first five texts (not all ten, to keep the notebook's output manageable).
  - `print(f"\n{'='*70}")` / a truncated preview of the sentence (`texts[idx][:80]`) / another `'='*70` line — a clearly delimited banner separating each text's comparison block.
  - `compare_ner_outputs(texts[idx], nlp, ner_pipeline)` — calls the function with the two already-loaded model objects (`nlp` from Section 2a, `ner_pipeline` from Section 4a).

### Full code block

```python
def compare_ner_outputs(text, nlp_model, hf_pipeline):
    """
    Run both NER systems on the same text and print a side-by-side comparison.

    Args:
        text:        raw input string
        nlp_model:   loaded spaCy language model
        hf_pipeline: loaded HF NER pipeline
    """
    # spaCy
    doc = nlp_model(text)
    spacy_ents = [(ent.text, ent.label_) for ent in doc.ents]

    # Hugging Face (merge subwords)
    hf_raw = hf_pipeline(text)
    hf_merged = merge_subword_entities(hf_raw)
    hf_ents = [(m["word"], m["entity"]) for m in hf_merged]

    # Print
    max_rows = max(len(spacy_ents), len(hf_ents))
    print(f"{'spaCy entity':<35} {'HF entity':<35}")
    print("-" * 70)
    for i in range(max_rows):
        sp = f"{spacy_ents[i][0]} [{spacy_ents[i][1]}]" if i < len(spacy_ents) else ""
        hf = f"{hf_ents[i][0]} [{hf_ents[i][1]}]" if i < len(hf_ents) else ""
        print(f"{sp:<35} {hf:<35}")


for idx in range(5):
    print(f"\n{'='*70}")
    print(f"texts[{idx}]: {texts[idx][:80]}...")
    print('='*70)
    compare_ner_outputs(texts[idx], nlp, ner_pipeline)
```

---

## 6. Tool Selection Discussion

### Why this section exists
This is a discussion-only step — a markdown cell with no accompanying code — inserted right after learners have seen both systems' raw output side by side, while the trade-offs are still fresh. It converts the visual impressions from Section 5 into an explicit decision framework learners can apply outside this lab.

### Discussion

The notebook presents a trade-off table contrasting spaCy `en_core_web_sm` and HF `dslim/bert-base-NER` across five criteria:

- **Speed** — spaCy's rule + ML pipeline is fast; the HF model requires a full transformer forward pass and is slower.
- **Accuracy** — spaCy is good for common entities; BERT-NER tends to be higher-accuracy on unusual names or domain-specific terms.
- **Label coverage** — spaCy covers `DATE`, `CARDINAL`, `MONEY`, `FAC`, `GPE`, and more; BERT-NER's CoNLL-2003 training only covers `ORG`, `PER`, `LOC`, and `MISC`.
- **Subword handling** — not needed for spaCy; required post-processing for BERT-NER (the Section 4 merge function).
- **Best for** — spaCy suits batch pipelines needing broad entity types; BERT-NER suits high-stakes extraction of rare entities.

The notebook's decision rule of thumb: reach for **spaCy** when you have a large corpus, need fast turnaround, and need `DATE`/`MONEY` fields; reach for **BERT-NER** when you have a small, high-value set and need accurate person/organization boundaries; and when you need both, either ensemble the two systems or route different entity types to whichever system covers them. This framing directly foreshadows Sections 8–10, which add three more systems (NLTK, a rule-based gazetteer, and Flair) to the same trade-off space and score all five quantitatively in Section 9c and 10d.

---

## 7. Evaluation Against a Gold Standard

### Why this section exists
Visual inspection (Sections 3, 5, 6) only goes so far — this section replaces "does it look right?" with a numeric, reproducible metric. It hand-annotates the first three texts as ground truth and defines a **strict exact-match** scoring function so every NER system in the rest of the notebook can be compared on the same objective scale.

### 7a. Gold standard annotations

#### Line-by-line

- The leading comment frames `gold_standard` as representing real human annotation work — the kind of labor-intensive manual labeling a production NER evaluation would require at scale.
- `gold_standard = {0: [...], 1: [...], 2: [...]}` — a dict keyed by text index (only `texts[0]`, `texts[1]`, and `texts[2]` are annotated; annotating all ten would be excessive for a teaching lab), where each value is a list of `{"entity_text": ..., "entity_label": ...}` dicts representing one correct entity.
  - For `texts[0]`: `WHO` (`ORG`), `2,400` (`CARDINAL`), `Southeast Asia` (`LOC` — note: `LOC`, not `GPE`; the accompanying markdown warns that a system tagging this `GPE` instead scores zero under strict matching, since span-correct-but-label-wrong still counts as a miss), `January` (`DATE`), and `March 2025` (`DATE`, annotated as one combined span rather than two separate tokens).
  - For `texts[1]`: `Ministry of Health` (`ORG`), `Amman` and `Jordan` (both `GPE`), `3.2 million` and `twelve` (both `CARDINAL`), and `the end of Q2` (`DATE`, annotated as the full phrase, not just `Q2`).
  - For `texts[2]`: `Al-Rashidi` (`PERSON` — note: just the surname, not `"Dr. Al-Rashidi"`; this specific annotation choice matters later in Section 9a, where the gazetteer must match the same shorter span), `Eastern Mediterranean Regional Office` and `CDC` and `WHO` (all `ORG`), and `next Friday` (`DATE`).
- `print(f"Gold standard covers {len(gold_standard)} texts.")` — confirms the dict has exactly 3 entries.

#### Full code block

```python
# Gold standard — manually annotated (represents real human annotation work)
gold_standard = {
    0: [
        {"entity_text": "WHO",            "entity_label": "ORG"},
        {"entity_text": "2,400",          "entity_label": "CARDINAL"},
        {"entity_text": "Southeast Asia", "entity_label": "LOC"},
        {"entity_text": "January",        "entity_label": "DATE"},
        {"entity_text": "March 2025",     "entity_label": "DATE"},
    ],
    1: [
        {"entity_text": "Ministry of Health",   "entity_label": "ORG"},
        {"entity_text": "Amman",                "entity_label": "GPE"},
        {"entity_text": "Jordan",               "entity_label": "GPE"},
        {"entity_text": "3.2 million",          "entity_label": "CARDINAL"},
        {"entity_text": "twelve",               "entity_label": "CARDINAL"},
        {"entity_text": "the end of Q2",        "entity_label": "DATE"},
    ],
    2: [
        {"entity_text": "Al-Rashidi",                          "entity_label": "PERSON"},
        {"entity_text": "Eastern Mediterranean Regional Office", "entity_label": "ORG"},
        {"entity_text": "CDC",                                "entity_label": "ORG"},
        {"entity_text": "WHO",                                "entity_label": "ORG"},
        {"entity_text": "next Friday",                        "entity_label": "DATE"},
    ],
}

print(f"Gold standard covers {len(gold_standard)} texts.")
```

### 7b. `evaluate_ner` function

#### Line-by-line

- `def evaluate_ner(predicted_ents, gold_ents):` — the core scoring function every system in the rest of the notebook calls. Its docstring specifies that an entity is a true positive only when **both** the text span and the label match a gold entry exactly, and documents the expected argument shapes and return keys.
- `gold_set = {(g["entity_text"], g["entity_label"]) for g in gold_ents}` — a set comprehension that converts the list of gold dicts into a set of `(text, label)` tuples, matching the tuple shape every system's predictions are normalized into elsewhere in the notebook. Using a `set` (not a list) is what makes exact-match comparison a simple, fast set operation instead of nested loops.
- `pred_set = set(predicted_ents)` — converts the predicted entity list (already a list of `(text, label)` tuples by convention) into a set the same way.
- `tp = len(pred_set & gold_set)` — the `&` operator computes set intersection: entities present in **both** sets, i.e., predictions that exactly match a gold entry — true positives.
- `fp = len(pred_set - gold_set)` — set difference: predicted entities that are **not** in the gold set — false positives (things the system claimed were entities but shouldn't have, or got right span/wrong label, or vice versa).
- `fn = len(gold_set - pred_set)` — the reverse difference: gold entities the system **missed entirely** — false negatives.
- `precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0` — precision is "of everything the system predicted, what fraction was correct?" The ternary guards against a `ZeroDivisionError` when the system predicted nothing at all (`tp + fp == 0`), returning `0.0` in that edge case instead of crashing.
- `recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0` — recall is "of everything that should have been found, what fraction was found?" Same zero-division guard, for the case where the gold set is empty.
- `f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)` — F1 is the harmonic mean of precision and recall, chosen (over a simple average) because it penalizes systems that only do well on one of the two metrics; again guarded against division by zero when both precision and recall are `0.0`.
- `return {"precision": ..., "recall": ..., "f1": ..., "tp": ..., "fp": ..., "fn": ...}` — returns all six computed values in a single dict, giving callers both the headline metrics and the raw counts they came from (useful for debugging why a score is low).
- `print("evaluate_ner() defined.")` — confirms the function was defined without a syntax error before it gets called repeatedly in later cells.

#### Full code block

```python
def evaluate_ner(predicted_ents, gold_ents):
    """
    Compute entity-level Precision, Recall, and F1 using strict exact match.

    An entity is a true positive only when BOTH the text span and the label
    match a gold entry exactly.

    Args:
        predicted_ents: list of (text, label) tuples from the NER system
        gold_ents:      list of {"entity_text": str, "entity_label": str} dicts

    Returns:
        dict with keys precision, recall, f1, tp, fp, fn
    """
    gold_set = {(g["entity_text"], g["entity_label"]) for g in gold_ents}
    pred_set = set(predicted_ents)

    tp = len(pred_set & gold_set)          # correct entities
    fp = len(pred_set - gold_set)          # predicted but not in gold
    fn = len(gold_set - pred_set)          # in gold but not predicted

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


print("evaluate_ner() defined.")
```

### 7c. Evaluate spaCy against the gold standard

#### Line-by-line

- The `print` calls build a results-table header with columns `Text`, `P`, `R`, `F1`, `TP`, `FP`, `FN`, right-aligned for numeric readability.
- `spacy_scores = []` — accumulates one score dict per gold-annotated text.
- `for idx, gold in gold_standard.items():` — iterates the gold dict's `(index, gold_list)` pairs, so this loop only touches the three annotated texts, not all ten.
  - `doc_i = nlp(texts[idx])` — runs spaCy fresh on the text at this specific index (re-running rather than reusing an earlier `doc`, since earlier cells may have processed different texts).
  - `pred = [(ent.text, ent.label_) for ent in doc_i.ents]` — converts spaCy's entities into the same `(text, label)` tuple shape `evaluate_ner` expects.
  - `scores = evaluate_ner(pred, gold)` — scores this text's predictions against its gold annotations.
  - `spacy_scores.append(scores)` — accumulates for the macro-average calculation below.
  - The `print` call formats one row per text using `:>6.2f` for the P/R/F1 floats (six-character right-aligned, two decimal places) and `:>4d` for the integer counts.
- `avg_p = sum(s["precision"] for s in spacy_scores) / len(spacy_scores)` — a **macro average**: it averages the per-text precision scores directly (as opposed to a micro average, which would pool all TP/FP/FN counts globally before computing one precision). Macro averaging treats every text as equally important regardless of how many entities it contains.
- `avg_r` and `avg_f1` — the same macro-averaging pattern applied to recall and F1.
- The final `print` shows the three macro-averaged numbers as a summary row.

#### Full code block

```python
# Evaluate spaCy across all gold-annotated texts

print("spaCy evaluation (strict exact match)\n")
print(f"{'Text':<8} {'P':>6} {'R':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
print("-" * 45)

spacy_scores = []
for idx, gold in gold_standard.items():
    doc_i = nlp(texts[idx])
    pred = [(ent.text, ent.label_) for ent in doc_i.ents]
    scores = evaluate_ner(pred, gold)
    spacy_scores.append(scores)
    print(f"texts[{idx}]  {scores['precision']:>6.2f} {scores['recall']:>6.2f} "
          f"{scores['f1']:>6.2f}  {scores['tp']:>4d} {scores['fp']:>4d} {scores['fn']:>4d}")

# Macro averages
avg_p  = sum(s["precision"] for s in spacy_scores) / len(spacy_scores)
avg_r  = sum(s["recall"]    for s in spacy_scores) / len(spacy_scores)
avg_f1 = sum(s["f1"]        for s in spacy_scores) / len(spacy_scores)
print(f"\nMacro avg  {avg_p:>6.2f} {avg_r:>6.2f} {avg_f1:>6.2f}")
```

### 7d. Evaluate HF BERT-NER against the gold standard

#### Line-by-line

- The leading comment flags the key incompatibility this cell must solve: HF's pipeline outputs BIO-tagged labels like `B-ORG` / `I-ORG`, while the gold standard uses plain `ORG`. Comparing them directly would score every correct HF prediction as a false positive.
- `def normalize_hf_label(label):` — a small helper with a one-line docstring (`'B-ORG', 'I-ORG' → 'ORG'; 'B-LOC' → 'LOC'; etc.`).
  - `if "-" in label: return label.split("-", 1)[1]` — `str.split("-", 1)` splits on the first hyphen only (the `1` argument caps it at one split), returning a two-element list like `['B', 'ORG']`; index `[1]` takes the part after the prefix — i.e., strips `B-`/`I-` regardless of which one it was.
  - `return label` — labels without a hyphen pass through unchanged (a defensive fallback, though in practice every entity label this pipeline returns has a `B-`/`I-` prefix since `"O"` tokens are already filtered out by the pipeline itself).
- The `print` header matches the same table format used for spaCy in 7c, for direct visual comparison.
- `hf_scores = []` — accumulator, parallel to `spacy_scores`.
- `for idx, gold in gold_standard.items():` — same three-text loop as before.
  - `hf_raw = ner_pipeline(texts[idx])` — runs the HF pipeline fresh on this text.
  - `hf_m = merge_subword_entities(hf_raw)` — reuses the Section 4d helper to reassemble whole-word entities.
  - `pred_hf = [(m["word"], normalize_hf_label(m["entity"])) for m in hf_m]` — builds `(text, label)` tuples, applying the BIO-prefix-stripping normalization to each label so `B-ORG` becomes `ORG` and can match the gold standard's `ORG` entries.
  - `scores = evaluate_ner(pred_hf, gold)` — same scoring function used for every other system in this notebook, guaranteeing an apples-to-apples comparison.
  - Accumulate and print, identical pattern to 7c.
- The macro-average block at the end mirrors 7c exactly, just over `hf_scores`.

#### Full code block

```python
# Evaluate HF BERT-NER across all gold-annotated texts
# Note: HF uses B-ORG / I-ORG BIO tags; gold uses ORG.
# We strip the B-/I- prefix to normalize labels before comparison.

def normalize_hf_label(label):
    """Convert 'B-ORG', 'I-ORG' → 'ORG'; 'B-LOC' → 'LOC'; etc."""
    if "-" in label:
        return label.split("-", 1)[1]
    return label


print("HF BERT-NER evaluation (strict exact match, BIO labels normalized)\n")
print(f"{'Text':<8} {'P':>6} {'R':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
print("-" * 45)

hf_scores = []
for idx, gold in gold_standard.items():
    hf_raw   = ner_pipeline(texts[idx])
    hf_m     = merge_subword_entities(hf_raw)
    pred_hf  = [(m["word"], normalize_hf_label(m["entity"])) for m in hf_m]
    scores   = evaluate_ner(pred_hf, gold)
    hf_scores.append(scores)
    print(f"texts[{idx}]  {scores['precision']:>6.2f} {scores['recall']:>6.2f} "
          f"{scores['f1']:>6.2f}  {scores['tp']:>4d} {scores['fp']:>4d} {scores['fn']:>4d}")

avg_p  = sum(s["precision"] for s in hf_scores) / len(hf_scores)
avg_r  = sum(s["recall"]    for s in hf_scores) / len(hf_scores)
avg_f1 = sum(s["f1"]        for s in hf_scores) / len(hf_scores)
print(f"\nMacro avg  {avg_p:>6.2f} {avg_r:>6.2f} {avg_f1:>6.2f}")
```

### 7e. Head-to-head summary chart

#### Line-by-line

- `print("=" * 50)` / title / `print("=" * 50)` — a banner framing the summary as the "wrap-up" of the two-system evaluation before the notebook moves on to adding more systems.
- The header `print` lays out columns for `Metric`, `spaCy`, and `BERT-NER`.
- `sp_avg = {"P": sum(...)/len(...), "R": ..., "F1": ...}` — a dict literal built with three inline generator-expression averages, computing the same macro-averaged precision/recall/F1 already computed in 7c but repackaged into a dict keyed by short metric names (`"P"`, `"R"`, `"F1"`) for convenient lookup in the printing loop below.
- `hf_avg = {...}` — the same pattern applied to `hf_scores`.
- `for metric in ["P", "R", "F1"]: print(f"{metric:<12} {sp_avg[metric]:>8.2f} {hf_avg[metric]:>10.2f}")` — iterates the three metric names and prints one row per metric, pulling the corresponding value out of each dict — a compact way to lay out a small comparison table without hard-coding three separate print statements.

#### Full code block

```python
# Head-to-head summary chart
print("=" * 50)
print("HEAD-TO-HEAD: spaCy vs. HF BERT-NER (macro avg)")
print("=" * 50)
print(f"{'Metric':<12} {'spaCy':>8} {'BERT-NER':>10}")
print("-" * 32)

sp_avg  = {"P": sum(s["precision"] for s in spacy_scores)/len(spacy_scores),
           "R": sum(s["recall"]    for s in spacy_scores)/len(spacy_scores),
           "F1":sum(s["f1"]        for s in spacy_scores)/len(spacy_scores)}
hf_avg  = {"P": sum(s["precision"] for s in hf_scores)/len(hf_scores),
           "R": sum(s["recall"]    for s in hf_scores)/len(hf_scores),
           "F1":sum(s["f1"]        for s in hf_scores)/len(hf_scores)}

for metric in ["P", "R", "F1"]:
    print(f"{metric:<12} {sp_avg[metric]:>8.2f} {hf_avg[metric]:>10.2f}")
```

---

## 8. NLTK Classical NER (MaxEnt Chunker)

### Why this section exists
So far both systems have been neural (spaCy's statistical pipeline, HF's transformer). This section introduces a **classical, pre-neural** approach — NLTK's tokenize → POS-tag → chunk pipeline — to answer whether an older statistical technique still holds up on this domain, and to widen the trade-off space from Section 6 to include speed/simplicity vs. label coverage.

### 8a. Setup, label map, and `run_nltk_ner`

#### Line-by-line

- `import nltk` — the Natural Language Toolkit library.
- The comment explains the download loop's purpose and notes `quiet=True` suppresses NLTK's normally noisy per-package progress output.
- `for pkg in [...]: nltk.download(pkg, quiet=True)` — downloads seven data packages, listed in both old and new naming forms for cross-version compatibility (NLTK renamed some packages, e.g. adding a `_tab` suffix, in newer releases): `punkt`/`punkt_tab` (sentence/word tokenizer models), `averaged_perceptron_tagger`/`averaged_perceptron_tagger_eng` (the pretrained POS-tagging model), `maxent_ne_chunker`/`maxent_ne_chunker_tab` (the pretrained Maximum Entropy named-entity chunker), and `words` (a corpus of known English words the chunker consults). Each `nltk.download` call is a no-op if the package is already cached locally, so re-running this cell is cheap after the first time.
- `_NLTK_LABEL_MAP = {...}` — a dict translating NLTK's own entity label vocabulary into the label scheme used by `gold_standard`: `ORGANIZATION`→`ORG`, `PERSON`→`PERSON` (unchanged), `GPE`→`GPE` (unchanged), `LOCATION`→`LOC`, `FACILITY`→`FAC`, and `GSP`→`GPE` (`GSP`, "geo-social-political," is an NLTK label variant sometimes emitted for entities that are ambiguously political/geographic — the notebook folds it into `GPE` for scoring purposes).
- `def run_nltk_ner(text):` — wraps the three-stage classical pipeline, with a docstring documenting the stages and explicitly noting the function does **not** detect `CARDINAL` or `DATE` entities (NLTK's chunker has no concept of numbers or dates — a structural limitation, not a bug, that will visibly cap this system's recall in 8c).
  - `tokens = nltk.word_tokenize(text)` — splits the raw text into tokens using the Penn Treebank tokenizer, which handles punctuation and contractions according to Penn Treebank conventions (e.g., splitting off trailing periods and commas as separate tokens).
  - `pos_tags = nltk.pos_tag(tokens)` — runs the Averaged Perceptron tagger over the token list, returning a list of `(word, POS_tag)` tuples (e.g., `("WHO", "NNP")` for a proper noun).
  - `tree = nltk.ne_chunk(pos_tags, binary=False)` — runs the pretrained MaxEnt chunker over the POS-tagged tokens, grouping contiguous tagged tokens into named-entity chunks. `binary=False` is the key argument: it keeps the fine-grained entity type distinctions (`PERSON`, `ORGANIZATION`, `GPE`, etc.) in the output tree, rather than collapsing every detected entity into one generic `NE` label (which is what `binary=True` would do).
  - `entities = []` — accumulator for the flattened `(text, label)` results.
  - `for subtree in tree:` — `tree` behaves like a flat sequence where each element is either a plain `(word, POS)` tuple (an ordinary, non-entity token) or an `nltk.Tree` object (a recognized entity chunk containing one or more leaves).
  - `if isinstance(subtree, nltk.Tree):` — filters for entity chunks only, skipping ordinary tokens.
    - `span = " ".join(word for word, _ in subtree.leaves())` — `subtree.leaves()` returns the list of `(word, POS)` pairs contained in this chunk; the generator expression pulls just the words (discarding POS tags via `_`) and joins them with spaces to reconstruct the entity's full text (e.g., a chunk spanning "Ministry", "of", "Health" becomes `"Ministry of Health"`).
    - `label = _NLTK_LABEL_MAP.get(subtree.label(), subtree.label())` — `subtree.label()` returns NLTK's raw entity-type string for this chunk (e.g., `"ORGANIZATION"`); `.get(key, key)` looks it up in the normalization map, falling back to the raw label unchanged if it isn't in the map (a defensive default for any label variant the map doesn't anticipate).
    - `entities.append((span, label))` — accumulates the normalized `(text, label)` tuple.
  - `return entities` — the final list, in the same tuple shape every other system in this notebook produces, ready to feed straight into `evaluate_ner`.
- `print("run_nltk_ner() defined.")` — confirms successful definition.

#### Full code block

```python
import nltk

# Download required NLTK data packages (quiet=True suppresses progress bars).
# Both old and new package names included for compatibility across NLTK versions.
for pkg in [
    "punkt", "punkt_tab",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab",
    "words",
]:
    nltk.download(pkg, quiet=True)

# NLTK entity labels differ from gold standard labels — map them.
_NLTK_LABEL_MAP = {
    "ORGANIZATION": "ORG",
    "PERSON":       "PERSON",
    "GPE":          "GPE",
    "LOCATION":     "LOC",
    "FACILITY":     "FAC",
    "GSP":          "GPE",
}


def run_nltk_ner(text):
    """
    Classical NLTK NER pipeline:
        word_tokenize  ->  pos_tag (PerceptronTagger)  ->  ne_chunk (MaxEnt)

    Returns a list of (entity_text, normalized_label) tuples.
    Does NOT detect CARDINAL or DATE entities.
    """
    tokens   = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tree     = nltk.ne_chunk(pos_tags, binary=False)

    entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            span  = " ".join(word for word, _ in subtree.leaves())
            label = _NLTK_LABEL_MAP.get(subtree.label(), subtree.label())
            entities.append((span, label))
    return entities


print("run_nltk_ner() defined.")
```

### 8b. Demo on texts[0]

#### Line-by-line

- `nltk_ents_0 = run_nltk_ner(texts[0])` — runs the new pipeline on the same running-example sentence used since Section 1.
- The first `print` block builds a two-column table (`Entity Text`, `Label`) and loops over `nltk_ents_0` to display each detected entity.
- The second block prints the gold-standard entries for `texts[0]` directly underneath, so a reader can visually diff NLTK's output against ground truth without scrolling back to Section 7a.
- The final `print` states explicitly: because NLTK has no `CARDINAL` or `DATE` labels, the gold entries `2,400` (`CARDINAL`), `January` (`DATE`), and `March 2025` (`DATE`) will **always** register as false negatives for this system — a structural ceiling on recall, not a fixable bug, which the evaluation in 8c will confirm numerically.

#### Full code block

```python
# Demo: run NLTK NER on texts[0] and compare with gold standard
nltk_ents_0 = run_nltk_ner(texts[0])

print("NLTK entities for texts[0]:")
print(f"{'Entity Text':<35} {'Label'}")
print("-" * 50)
for span, label in nltk_ents_0:
    print(f"{span:<35} {label}")

print("\nGold standard for texts[0]:")
for g in gold_standard[0]:
    print(f"  {g['entity_text']:<35} {g['entity_label']}")

print(
    "\nNote: NLTK has no CARDINAL or DATE labels, so those gold entries will"
    " always be false negatives regardless of span detection ability."
)
```

### 8c. Evaluate NLTK + 3-way comparison

#### Line-by-line

- The header `print` calls build the same table format used in Sections 7c/7d (`Text`, `P`, `R`, `F1`, `TP`, `FP`, `FN`), for direct visual consistency across all systems.
- `nltk_scores = []` — accumulator.
- `for idx, gold in gold_standard.items():` — the same three-text evaluation loop pattern used for spaCy and HF.
  - `pred = run_nltk_ner(texts[idx])` — runs the classical pipeline (no separate "raw vs. merged" step needed here, since `run_nltk_ner` already returns whole-token spans).
  - `scores = evaluate_ner(pred, gold)` — the exact same scoring function used for every other system, guaranteeing comparability.
  - Accumulate and print a formatted row, identical pattern to 7c/7d.
- The macro-average block computes `avg_p`, `avg_r`, `avg_f1` the same way as prior sections.
- `print()` — blank line separator before the comparison banner.
- The `"3-WAY COMPARISON"` banner and header row set up a table with four columns: `Metric`, `spaCy`, `BERT-NER`, `NLTK`.
- `for lbl, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:` — iterates three `(display label, dict key)` pairs so the loop body can print a human-readable label (`"Precision"`) while indexing into the score dicts with the machine key (`"precision"`).
  - `sp = sum(s[key] for s in spacy_scores) / len(spacy_scores)` — recomputes spaCy's macro average for this metric inline (redundant with 7c's `avg_p`/`avg_r`/`avg_f1`, but self-contained so this cell doesn't depend on which variables survived from earlier cells).
  - `hf` and `nt` — the same pattern applied to `hf_scores` and the newly computed `nltk_scores`.
  - `print(f"{lbl:<12} {sp:>8.2f} {hf:>10.2f} {nt:>8.2f}")` — prints one row per metric, all three systems' values aligned in columns.
- The closing `print` interprets the numbers in prose: NLTK's missing `CARDINAL`/`DATE` labels cap its recall; for the entity types it does cover (`ORG`, `PERSON`, `GPE`, `LOC`) its precision can be competitive with the neural systems, but the structural label gap keeps its overall F1 lower.

#### Full code block

```python
# Evaluate NLTK NER across all gold-annotated texts
print("NLTK evaluation (strict exact match)\n")
print(f"{'Text':<8} {'P':>6} {'R':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
print("-" * 45)

nltk_scores = []
for idx, gold in gold_standard.items():
    pred   = run_nltk_ner(texts[idx])
    scores = evaluate_ner(pred, gold)
    nltk_scores.append(scores)
    print(f"texts[{idx}]  {scores['precision']:>6.2f} {scores['recall']:>6.2f} "
          f"{scores['f1']:>6.2f}  {scores['tp']:>4d} {scores['fp']:>4d} {scores['fn']:>4d}")

avg_p  = sum(s['precision'] for s in nltk_scores) / len(nltk_scores)
avg_r  = sum(s['recall']    for s in nltk_scores) / len(nltk_scores)
avg_f1 = sum(s['f1']        for s in nltk_scores) / len(nltk_scores)
print(f"\nMacro avg  {avg_p:>6.2f} {avg_r:>6.2f} {avg_f1:>6.2f}")

# --- 3-way comparison: spaCy vs. BERT-NER vs. NLTK ---
print()
print("=" * 58)
print("3-WAY COMPARISON: spaCy vs. BERT-NER vs. NLTK (macro avg)")
print("=" * 58)
print(f"{'Metric':<12} {'spaCy':>8} {'BERT-NER':>10} {'NLTK':>8}")
print("-" * 42)

for lbl, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
    sp = sum(s[key] for s in spacy_scores) / len(spacy_scores)
    hf = sum(s[key] for s in hf_scores)    / len(hf_scores)
    nt = sum(s[key] for s in nltk_scores)  / len(nltk_scores)
    print(f"{lbl:<12} {sp:>8.2f} {hf:>10.2f} {nt:>8.2f}")

print(
    "\nInterpretation: NLTK's missing CARDINAL/DATE labels cap its recall."
    "\nFor the entity types it does cover (ORG, PERSON, GPE, LOC),"
    " precision can be competitive — but the label gap keeps overall F1 low."
)
```

---

## 9. Rule-Based NER (Regex + Gazetteer)

### Why this section exists
The fourth technique is deliberately the simplest possible baseline: no machine learning at all, just a curated lookup table (**gazetteer**) plus regular expressions for numbers and dates. It establishes a lower/upper interpretability bound against which the three ML-based systems can be judged, and it directly answers the practical question: *how much does adding ML actually buy you over a hand-crafted domain list?*

### 9a. Gazetteers, regex patterns, and `run_gazetteer_ner`

#### Line-by-line

- `import re` — Python's standard-library regular-expression module.
- The comment above `GAZETTEERS` claims entries are ordered longest-first within each label "so that a longer match is preferred over a shorter one during overlap resolution." In practice, this ordering is not what actually enforces longest-match behavior — the `candidates.sort(...)` call further down re-sorts every candidate by start position and span length regardless of the gazetteer's own list order, so the comment describes the *intent* rather than a strict requirement of the current implementation; it's worth reading past this comment to the actual overlap-resolution logic to see how longest-match is really guaranteed.
- `GAZETTEERS = {...}` — a dict mapping four label names to lists of known entity strings, hand-curated from the ten source texts:
  - `"ORG"` — 12 organization names, from long multi-word names (`"Eastern Mediterranean Regional Office"`) down to short acronyms (`"UNICEF"`, `"WHO"`, `"CDC"`).
  - `"GPE"` — 10 country/city names (`"Democratic Republic of Congo"`, `"Abu Dhabi"`, `"Amman"`, `"Jordan"`, `"Aleppo"`, `"Idlib"`, `"Senegal"`, `"Nigeria"`, `"Romania"`, `"Italy"`, `"France"`).
  - `"LOC"` — 5 broader geographic regions (`"sub-Saharan Africa"`, `"Southeast Asia"`, `"Caribbean"`, `"northern Syria"`, `"EU"`).
  - `"PERSON"` — a single entry, `"Al-Rashidi"`. The inline comment explains a subtle, deliberate design decision: the gold standard (Section 7a) annotates only the surname `"Al-Rashidi"`, not the full `"Dr. Al-Rashidi"`. If the gazetteer listed the longer form instead, the longest-match resolution logic (below) would select that longer span and permanently block the shorter gold-standard span from ever being found — turning a potential true positive into a guaranteed miss. This is exactly the kind of gazetteer-vs-gold-annotation mismatch that makes rule-based systems brittle in practice.
- `_CARDINAL_RE = [...]` — three regex patterns, tried independently:
  - `r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"` — matches comma-grouped large numbers like `2,400` or `15,000`. `\b` is a word boundary; `\d{1,3}` matches 1–3 leading digits; `(?:,\d{3})+` is a non-capturing group requiring one or more repetitions of a comma followed by exactly three digits (the standard thousands-separator pattern); `(?:\.\d+)?` optionally allows a decimal tail.
  - `r"\b\d+\.\d+\s+(?:million|billion|thousand)\b"` — matches decimal-plus-scale-word phrases like `4.2 billion`.
  - The third pattern is a `\b(?:one|two|...|billion)\b` alternation matching spelled-out number words from `one` through `twelve`, plus round-number words like `hundred`, `thousand`, `million`, `billion`.
- `_DATE_RE = [...]` — six patterns, also tried independently: `"the end of Q2"`-style phrases, `"next Friday"`-style weekday references, `"March 2025"`-style month+year pairs, bare month names (`"January"`), `"late/early/mid 2026"`-style approximate-year phrases, and bare fiscal-quarter references (`"Q2"`) as a catch-all fallback for quarter mentions not wrapped in an "end of" phrase.
- `def run_gazetteer_ner(text):` — the main function, with a docstring summarizing the approach (gazetteer exact-match plus regex, overlaps resolved by longest-match).
  - `candidates = []` — accumulator for every raw match found, before overlap resolution.
  - The nested loop `for label, entries in GAZETTEERS.items(): for entry in entries: for m in re.finditer(re.escape(entry), text):` — for every gazetteer entry, `re.escape(entry)` escapes any regex-special characters in the entity string (defensive — none of the current entries need it, but it protects against future entries containing characters like `.` or `(`) before passing it to `re.finditer`, which returns every non-overlapping occurrence of that literal string in the text. Each match contributes `(start, end, matched_text, label)` to `candidates`.
  - `for pat in _CARDINAL_RE: for m in re.finditer(pat, text, re.IGNORECASE): candidates.append((m.start(), m.end(), m.group(), "CARDINAL"))` — runs each cardinal-number pattern against the text case-insensitively, appending every match with label `"CARDINAL"`.
  - The equivalent loop for `_DATE_RE` appends matches labeled `"DATE"`.
  - `candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))` — sorts all candidates (gazetteer hits and regex hits mixed together) by a two-part key: first by start position ascending (`c[0]`), then, for candidates starting at the same position, by span length descending (`-(c[1] - c[0])`, negated so the longest span sorts first). This is the line that actually implements the "longest-match" guarantee described in the leading comment — regardless of which order entries appeared in `GAZETTEERS` or in the regex lists.
  - `accepted, occupied = [], set()` — `accepted` collects the final non-overlapping entities; `occupied` is a set of character indices already claimed by an accepted match.
  - `for start, end, span_text, label in candidates:` — walks the sorted candidates in order (left-to-right, longest-first at ties).
    - `positions = set(range(start, end))` — the set of character indices this candidate would occupy.
    - `if not positions & occupied:` — accepts the candidate only if none of its character positions overlap an already-claimed span; this is the greedy longest-match-first, left-to-right overlap resolution the docstring promises.
      - `accepted.append((span_text, label))` / `occupied |= positions` — records the accepted entity and marks its character range as claimed so a shorter, overlapping candidate later in the sorted list gets rejected.
  - `return accepted` — the final list of `(text, label)` tuples, same shape as every other system's output.
- `print("run_gazetteer_ner() defined.")` and `print(f"Gazetteer covers {sum(len(v) for v in GAZETTEERS.values())} named entity entries.")` — the second line uses a generator expression summing the list lengths across all four gazetteer categories to report the total entry count (28 entries: 12 + 10 + 5 + 1).

#### Full code block

```python
import re

# --- Gazetteer: curated entity lookup tables ---
# Entries are ordered longest-first within each label so that a longer match
# is preferred over a shorter one during overlap resolution.
GAZETTEERS = {
    "ORG": [
        "Eastern Mediterranean Regional Office",
        "National Institute of Allergy and Infectious Diseases",
        "European Centre for Disease Prevention and Control",
        "Pan American Health Organization",
        "King Hussein Medical Center",
        "Sheikh Khalifa Medical City",
        "Royal Medical Services",
        "Ministry of Health",
        "Department of Health",
        "Global Fund",
        "UNICEF", "WHO", "CDC",
    ],
    "GPE": [
        "Democratic Republic of Congo",
        "Abu Dhabi", "Amman", "Jordan", "Aleppo", "Idlib",
        "Senegal", "Nigeria", "Romania", "Italy", "France",
    ],
    "LOC": [
        "sub-Saharan Africa", "Southeast Asia",
        "Caribbean", "northern Syria", "EU",
    ],
    # Gold standard annotates the surname only ("Al-Rashidi"), not the full title form.
    # Listing "Dr. Al-Rashidi" here would cause a false positive — the longer span
    # would be selected by longest-match, blocking the shorter gold-standard span.
    "PERSON": ["Al-Rashidi"],
}

# --- Regex patterns for numerals and temporal expressions ---
_CARDINAL_RE = [
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",          # 2,400 / 15,000
    r"\b\d+\.\d+\s+(?:million|billion|thousand)\b", # 4.2 billion
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|"
    r"twenty|thirty|forty|fifty|hundred|thousand|million|billion)\b",
]

_DATE_RE = [
    r"\bthe\s+end\s+of\s+Q[1-4]\b",                # the end of Q2
    r"\bnext\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b",  # March 2025
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b",             # January
    r"\b(?:late|early|mid)\s+\d{4}\b",
    r"\bQ[1-4]\b",
]


def run_gazetteer_ner(text):
    """
    Pure rule-based NER: gazetteer exact-match lookup + regex patterns.
    Resolves overlapping candidates by keeping the longest span (longest-match).
    """
    candidates = []

    for label, entries in GAZETTEERS.items():
        for entry in entries:
            for m in re.finditer(re.escape(entry), text):
                candidates.append((m.start(), m.end(), m.group(), label))

    for pat in _CARDINAL_RE:
        for m in re.finditer(pat, text, re.IGNORECASE):
            candidates.append((m.start(), m.end(), m.group(), "CARDINAL"))

    for pat in _DATE_RE:
        for m in re.finditer(pat, text, re.IGNORECASE):
            candidates.append((m.start(), m.end(), m.group(), "DATE"))

    # Resolve overlaps: prefer longer spans; on ties prefer earlier start.
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    accepted, occupied = [], set()
    for start, end, span_text, label in candidates:
        positions = set(range(start, end))
        if not positions & occupied:
            accepted.append((span_text, label))
            occupied |= positions

    return accepted


print("run_gazetteer_ner() defined.")
print(f"Gazetteer covers {sum(len(v) for v in GAZETTEERS.values())} named entity entries.")
```

### 9b. Demo on texts[0]

#### Line-by-line

- `gaz_ents_0 = run_gazetteer_ner(texts[0])` — runs the rule-based function on the running-example sentence, following the same demo pattern established in 8b.
- The first `print` block builds a two-column table and displays every entity the gazetteer/regex approach found.
- The second block prints the gold-standard entries for `texts[0]` underneath for immediate visual comparison — the same layout used in 8b, letting a reader flip between this section and Section 8b to compare NLTK's and the gazetteer's misses side by side.

#### Full code block

```python
# Demo: run Gazetteer NER on texts[0] and compare with gold standard
gaz_ents_0 = run_gazetteer_ner(texts[0])

print("Gazetteer NER entities for texts[0]:")
print(f"{'Entity Text':<35} {'Label'}")
print("-" * 50)
for span, label in gaz_ents_0:
    print(f"{span:<35} {label}")

print("\nGold standard for texts[0]:")
for g in gold_standard[0]:
    print(f"  {g['entity_text']:<35} {g['entity_label']}")
```

### 9c. Evaluate + final 4-model comparison + bar chart

#### Line-by-line

- `import matplotlib.pyplot as plt` / `import numpy as np` — imports the plotting library and NumPy, needed for the bar chart at the end of this cell (the first chart in the notebook).
- The `print` header and evaluation loop follow the exact same pattern as Sections 7c/7d/8c: `gaz_scores = []`, loop over `gold_standard.items()`, call `run_gazetteer_ner(texts[idx])`, score with `evaluate_ner`, accumulate, and print a formatted row.
- The macro-average block computes `avg_p`, `avg_r`, `avg_f1` for the gazetteer system the same way as every prior system.
- `models = ["spaCy", "BERT-NER", "NLTK", "Gazetteer"]` — the four system names, in display order, used to drive both the printed table and the chart's x-axis labels.
- `score_lists = [spacy_scores, hf_scores, nltk_scores, gaz_scores]` — the corresponding list of per-text score-dict lists, in the same order as `models`.
- `summary = {model: {"P": ..., "R": ..., "F1": ...} for model, sl in zip(models, score_lists)}` — a dict comprehension that, for each `(model, score_list)` pair produced by `zip`, computes the macro average of `precision`/`recall`/`f1` across that system's per-text scores (via inline `sum(...)/len(...)` generator expressions), building a nested dict keyed first by model name, then by metric abbreviation.
- The `print` banner and header row lay out a table with one column per model.
- `for display_lbl, key in [("Precision", "P"), ("Recall", "R"), ("F1", "F1")]:` — iterates the three metrics.
  - `row = f"{display_lbl:<12}"` — starts the row string with the metric's display label.
  - `for m in models: row += f" {summary[m][key]:>10.2f}"` — appends one right-aligned value per model onto the row string using `+=` string concatenation in a loop (rather than building a single f-string, since the number of models is data-driven, not fixed).
  - `print(row)` — prints the completed row.
- **Bar chart:**
  - `x = np.arange(len(models))` — produces the array `[0, 1, 2, 3]`, one integer position per model, used as the base x-coordinate for each group of bars.
  - `width = 0.25` — the width of each individual bar within a group; three bars per group (P, R, F1) times `0.25` leaves room for all three to sit side by side without overlapping.
  - `fig, ax = plt.subplots(figsize=(9, 5))` — creates a new figure and axes sized 9×5 inches.
  - `ax.bar(x - width, [...], width, label='Precision', color='#4C72B0')` — draws the precision bars, shifted left by one bar-width from each model's base position `x`, so the three metrics' bars sit adjacent rather than stacked.
  - `ax.bar(x, [...], width, label='Recall', color='#DD8452')` — the recall bars, centered exactly at each model's base position.
  - `ax.bar(x + width, [...], width, label='F1', color='#55A868')` — the F1 bars, shifted right by one bar-width.
  - `ax.set_xlabel(...)`, `ax.set_ylabel(...)`, `ax.set_title(...)` — axis and title labels.
  - `ax.set_xticks(x)` / `ax.set_xticklabels(models)` — replaces the default numeric x-axis ticks (`0, 1, 2, 3`) with the actual model names, so the chart reads as a categorical comparison rather than a numeric plot.
  - `ax.set_ylim(0, 1.0)` — fixes the y-axis range to the full possible score range (0 to 1), so bar heights are directly comparable and not distorted by auto-scaling to the tallest bar.
  - `ax.legend()` — displays the Precision/Recall/F1 color legend, using the `label=` arguments passed to each `ax.bar` call.
  - `ax.grid(axis='y', alpha=0.3)` — adds faint (30% opacity) horizontal gridlines only, making it easier to read bar heights without cluttering the chart with vertical lines.
  - `plt.tight_layout()` — automatically adjusts subplot spacing so axis labels and the title don't get clipped by the figure edges.
  - `plt.show()` — renders the figure inline in the notebook output.
- The closing `print` block lists key takeaways in prose: spaCy is strongest overall because its broad label coverage (`DATE`, `CARDINAL`) boosts recall; BERT-NER scores lowest under strict matching because token-boundary mismatches and missing `DATE`/`CARDINAL` labels hurt it; NLTK is competitive on the entity types it covers but the label gap drags its F1 down; and the gazetteer's precision reflects how complete its lookup list is, with its regex component filling the `DATE`/`CARDINAL` gap the ML systems' label schemes create for NLTK/BERT — and expanding the gazetteer's entry list would directly raise its recall.

#### Full code block

```python
import matplotlib.pyplot as plt
import numpy as np

# --- Evaluate Gazetteer NER across all gold-annotated texts ---
print("Gazetteer NER evaluation (strict exact match)\n")
print(f"{'Text':<8} {'P':>6} {'R':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
print("-" * 45)

gaz_scores = []
for idx, gold in gold_standard.items():
    pred   = run_gazetteer_ner(texts[idx])
    scores = evaluate_ner(pred, gold)
    gaz_scores.append(scores)
    print(f"texts[{idx}]  {scores['precision']:>6.2f} {scores['recall']:>6.2f} "
          f"{scores['f1']:>6.2f}  {scores['tp']:>4d} {scores['fp']:>4d} {scores['fn']:>4d}")

avg_p  = sum(s['precision'] for s in gaz_scores) / len(gaz_scores)
avg_r  = sum(s['recall']    for s in gaz_scores) / len(gaz_scores)
avg_f1 = sum(s['f1']        for s in gaz_scores) / len(gaz_scores)
print(f"\nMacro avg  {avg_p:>6.2f} {avg_r:>6.2f} {avg_f1:>6.2f}")

# --- Final 4-model comparison table ---
models      = ["spaCy", "BERT-NER", "NLTK", "Gazetteer"]
score_lists = [spacy_scores, hf_scores, nltk_scores, gaz_scores]

summary = {
    model: {
        "P":  sum(s["precision"] for s in sl) / len(sl),
        "R":  sum(s["recall"]    for s in sl) / len(sl),
        "F1": sum(s["f1"]        for s in sl) / len(sl),
    }
    for model, sl in zip(models, score_lists)
}

print()
print("=" * 68)
print("FINAL 4-MODEL COMPARISON vs. Gold Standard (macro avg, strict exact match)")
print("=" * 68)
print(f"{'Metric':<12} {'spaCy':>8} {'BERT-NER':>10} {'NLTK':>8} {'Gazetteer':>11}")
print("-" * 52)
for display_lbl, key in [("Precision", "P"), ("Recall", "R"), ("F1", "F1")]:
    row = f"{display_lbl:<12}"
    for m in models:
        row += f" {summary[m][key]:>10.2f}"
    print(row)

# --- Bar chart ---
x     = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, [summary[m]['P']  for m in models], width, label='Precision', color='#4C72B0')
ax.bar(x,         [summary[m]['R']  for m in models], width, label='Recall',    color='#DD8452')
ax.bar(x + width, [summary[m]['F1'] for m in models], width, label='F1',        color='#55A868')

ax.set_xlabel('NER System')
ax.set_ylabel('Score (macro avg, 0–1)')
ax.set_title('NER System Comparison — Precision / Recall / F1\nvs. Gold Standard (strict exact match)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print()
print('Key takeaways:')
print('  spaCy     — strongest overall: broad label coverage (DATE, CARDINAL) boosts recall')
print('  BERT-NER  — low F1 under strict match: token boundaries & missing DATE/CARDINAL hurt')
print('  NLTK      — competitive on entity types it covers; blind to DATE/CARDINAL drags F1 down')
print('  Gazetteer — precision reflects gazetteer completeness; recall bounded by list coverage;')
print('              regex fills DATE/CARDINAL gap; expanding the list directly raises recall')
```

---

## 10. Flair NER (Contextual String Embeddings)

### Why this section exists
The fifth and final system introduces a different neural architecture entirely: Flair's character-level, bidirectional language model with a CRF decoder on top, popularized for **contextual string embeddings**. Unlike BERT's WordPiece tokenizer, Flair never fragments words into subword pieces, which makes it a useful contrast to the subword-handling problem Section 4 spent an entire function solving — and it closes the lab with a full five-way leaderboard.

### 10a. Load the tagger, `normalize_flair_label`, and `run_flair_ner`

#### Line-by-line

- `from flair.data import Sentence` — imports Flair's `Sentence` class, which wraps a raw string into Flair's internal tokenized representation before tagging.
- `from flair.models import SequenceTagger` — imports the pretrained sequence-tagging model wrapper.
- The comment notes the first run downloads ~120 MB of model weights and caches them locally, similar to the HF pipeline's caching behavior in Section 4a.
- `flair_tagger = SequenceTagger.load("flair/ner-english")` — loads Flair's standard 4-class English NER tagger from the Flair model hub (or local cache on subsequent runs). This model was trained with a **bidirectional character-level language model** (reading text left-to-right and right-to-left, concatenating the character states at word boundaries to form word embeddings) feeding into a **CRF (Conditional Random Field)** decoder, which predicts the most globally consistent tag sequence for the whole sentence rather than tagging each token independently.
- `def normalize_flair_label(label):` — a one-line helper with a docstring explaining the mapping.
  - `return "PERSON" if label == "PER" else label` — a conditional expression: Flair's CoNLL-2003-derived label scheme uses `PER` for persons (matching the notebook's earlier observation that HF's raw BIO tags used `B-PER` too), while the gold standard uses `PERSON`; this ternary normalizes just that one label and passes everything else (`ORG`, `LOC`, `MISC`) through unchanged.
- `def run_flair_ner(text):` — wraps prediction for a single text, with a docstring noting no subword post-processing is needed since Flair always returns whole-token spans.
  - `sentence = Sentence(text)` — constructs a Flair `Sentence` object, which tokenizes the raw text internally.
  - `flair_tagger.predict(sentence)` — runs the tagger's inference **in place**: rather than returning a new object, it mutates `sentence`, attaching predicted NER labels onto it. This is a different calling convention from spaCy's `nlp(text)` or the HF pipeline's `ner_pipeline(text)`, both of which return a fresh result rather than mutating an input.
  - `return [(span.text, normalize_flair_label(span.get_label("ner").value)) for span in sentence.get_spans("ner")]` — `sentence.get_spans("ner")` retrieves the predicted entity spans from the "ner" annotation layer, already merged into whole-token `Span` objects (no `##`-style fragments to reassemble, unlike Section 4). For each span, `span.get_label("ner")` retrieves the `Label` object holding the predicted tag for that layer, and `.value` extracts the plain string label (e.g., `"PER"`); the label is normalized through `normalize_flair_label` before pairing with `span.text` into the same `(text, label)` tuple shape used throughout the notebook.
- `print("Flair tagger loaded and run_flair_ner() defined.")` — confirms both the (potentially slow) model load and the function definition succeeded.

#### Full code block

```python
from flair.data import Sentence
from flair.models import SequenceTagger

# Load the standard 4-class English NER tagger.
# First run downloads the model weights (~120 MB) and caches them locally.
flair_tagger = SequenceTagger.load("flair/ner-english")

# Flair uses PER for persons; gold standard uses PERSON — normalize at evaluation time.
def normalize_flair_label(label):
    """Map Flair's 'PER' to 'PERSON'; all other labels pass through unchanged."""
    return "PERSON" if label == "PER" else label


def run_flair_ner(text):
    """
    Run Flair NER on a single text string.

    Returns a list of (entity_text, normalized_label) tuples.
    No subword post-processing needed — Flair always returns whole-token spans.
    """
    sentence = Sentence(text)
    flair_tagger.predict(sentence)
    return [
        (span.text, normalize_flair_label(span.get_label("ner").value))
        for span in sentence.get_spans("ner")
    ]


print("Flair tagger loaded and run_flair_ner() defined.")
```

### 10b. Demo on texts[0]

#### Line-by-line

- `flair_ents_0 = run_flair_ner(texts[0])` — runs Flair on the running-example sentence, matching the same demo pattern used for NLTK (8b) and the gazetteer (9b).
- The first `print` block displays Flair's detected entities in a two-column table.
- The second block prints the gold standard for `texts[0]` directly beneath, for the same side-by-side comparison used in the two previous demo sections.
- The final `print` states two structural facts up front: Flair has no `CARDINAL`/`DATE` labels, so those gold entries will always be false negatives — the same structural gap BERT-NER has — but *unlike* BERT-NER, Flair never fragments a word into subword tokens, so its entity spans are always clean, whole-token strings (contrasting directly with Section 4's `merge_subword_entities` workaround).

#### Full code block

```python
# Demo: run Flair NER on texts[0] and compare with gold standard
flair_ents_0 = run_flair_ner(texts[0])

print("Flair entities for texts[0]:")
print(f"{'Entity Text':<35} {'Label'}")
print("-" * 50)
for span, label in flair_ents_0:
    print(f"{span:<35} {label}")

print("\nGold standard for texts[0]:")
for g in gold_standard[0]:
    print(f"  {g['entity_text']:<35} {g['entity_label']}")

print(
    "\nNote: Flair has no CARDINAL or DATE labels, so those gold entries"
    " are always false negatives — same structural gap as BERT-NER."
    "\nHowever, unlike BERT-NER, Flair never produces fragmented subword"
    " tokens, so entity spans are always clean whole-token strings."
)
```

### 10c. Evaluate Flair against the gold standard

#### Line-by-line

- The header `print` calls build the same table format used for every prior system's evaluation.
- `flair_scores = []` — accumulator, following the established pattern.
- `for idx, gold in gold_standard.items():` — the same three-text loop.
  - `pred = run_flair_ner(texts[idx])` — runs Flair on this text.
  - `scores = evaluate_ner(pred, gold)` — scores with the shared metric function.
  - Accumulate and print a formatted row, identical style to every prior evaluation cell.
- The macro-average block computes `avg_p`, `avg_r`, `avg_f1` for Flair the same way as all four earlier systems.
- The closing `print` interprets the result in prose: Flair's character-level embeddings tend to give it cleaner span boundaries than BERT-NER's subword tokenizer; it shares the identical `DATE`/`CARDINAL` gap with BERT-NER, but its entity-type precision is typically stronger, especially on hyphenated names (like `Al-Rashidi`) and acronyms (like `WHO`, `CDC`) that a WordPiece tokenizer might otherwise fragment.

#### Full code block

```python
# Evaluate Flair NER across all gold-annotated texts
print("Flair NER evaluation (strict exact match)\n")
print(f"{'Text':<8} {'P':>6} {'R':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
print("-" * 45)

flair_scores = []
for idx, gold in gold_standard.items():
    pred   = run_flair_ner(texts[idx])
    scores = evaluate_ner(pred, gold)
    flair_scores.append(scores)
    print(f"texts[{idx}]  {scores['precision']:>6.2f} {scores['recall']:>6.2f} "
          f"{scores['f1']:>6.2f}  {scores['tp']:>4d} {scores['fp']:>4d} {scores['fn']:>4d}")

avg_p  = sum(s['precision'] for s in flair_scores) / len(flair_scores)
avg_r  = sum(s['recall']    for s in flair_scores) / len(flair_scores)
avg_f1 = sum(s['f1']        for s in flair_scores) / len(flair_scores)
print(f"\nMacro avg  {avg_p:>6.2f} {avg_r:>6.2f} {avg_f1:>6.2f}")

print(
    "\nInterpretation: Flair's character-level embeddings give it cleaner"
    " span boundaries than BERT-NER's subword tokenizer. The DATE/CARDINAL"
    " gap is identical to BERT-NER, but entity-type precision is typically"
    " stronger — especially on hyphenated names and acronyms."
)
```

### 10d. Final 5-model comparison, bar chart, and ranked summary

#### Line-by-line

- `import matplotlib.pyplot as plt` / `import numpy as np` — re-imported here for this cell's self-containment; this is harmless and cheap because Python caches already-imported modules in `sys.modules`, so a repeated `import` statement is effectively a fast no-op lookup rather than a second real import.
- `models_5 = ["spaCy", "BERT-NER", "NLTK", "Gazetteer", "Flair"]` — extends the Section 9c model list with Flair as the fifth entry.
- `score_lists_5 = [spacy_scores, hf_scores, nltk_scores, gaz_scores, flair_scores]` — the corresponding five score-dict lists, in the same order.
- `summary_5 = {...}` — the identical dict-comprehension pattern from 9c, now computing macro-averaged P/R/F1 for all five systems via `zip(models_5, score_lists_5)`.
- The `print` banner, header row, and per-metric row-building loop (`row = f"{display_lbl:<12}"`, then `row += ...` per model) follow the exact same structure as 9c, just with the wider 5-column table.
- **Bar chart:**
  - `x = np.arange(len(models_5))` — now `[0, 1, 2, 3, 4]`, one position per model.
  - `width = 0.22` — slightly narrower than the 4-model chart's `0.25`, needed because there's now a fifth cluster of bars competing for the same horizontal space between adjacent model positions.
  - The `fig, ax = plt.subplots(figsize=(11, 5))` call uses a wider figure (11 inches vs. 9) to accommodate the extra model without crowding the x-axis labels.
  - The three `ax.bar(...)` calls follow the identical offset pattern from 9c (`x - width`, `x`, `x + width`) with the same color scheme (`#4C72B0` for Precision, `#DD8452` for Recall, `#55A868` for F1) so the two charts in the notebook read as visually consistent, even though this one has an extra category.
  - The axis-labeling, tick, legend, grid, and layout calls (`set_xlabel`, `set_ylabel`, `set_title`, `set_xticks`, `set_xticklabels`, `set_ylim`, `legend`, `grid`, `tight_layout`, `show`) mirror 9c exactly, just parameterized on `models_5`/`summary_5`.
- **Ranked summary:**
  - `ranked = sorted(models_5, key=lambda m: summary_5[m]["F1"], reverse=True)` — sorts the model *names* (not the score dicts) by looking up each model's F1 score through the `summary_5` dict via a `lambda` key function, with `reverse=True` so the highest-F1 model comes first.
  - `for rank, m in enumerate(ranked, 1):` — `enumerate(..., 1)` produces 1-based rank numbers instead of the default 0-based index, so the first-place model prints as `1.` rather than `0.`.
  - `print(f"  {rank}. {m:<12}  P={summary_5[m]['P']:.2f}  R={summary_5[m]['R']:.2f}  F1={summary_5[m]['F1']:.2f}")` — prints one ranked line per model with all three metrics inline.
- The final `print("""...""")` is a triple-quoted multi-line string containing the lab's closing takeaways and design guidance: spaCy remains strongest overall due to broad label coverage maximizing recall; Flair offers the best span precision among the neural models (no subword fragmentation) but shares BERT-NER's `DATE`/`CARDINAL` recall ceiling; the gazetteer's precision reflects list completeness, with its regex component filling the date/cardinal gap, and its recall would rise directly with a larger hand-curated list; NLTK is competitive in precision on the types it covers but capped by the same missing-label problem; and BERT-NER scores lowest under strict matching due to both subword boundary misalignment and missing labels. The design guidance that follows translates this into concrete advice: use spaCy when `DATE`/`MONEY`/`CARDINAL` fields matter alongside named entities; use Flair when clean span boundaries matter more than broad label coverage; and consider ensembling Flair with the gazetteer for pipelines that need both high precision and full domain coverage.

#### Full code block

```python
import matplotlib.pyplot as plt
import numpy as np

# --- Final 5-model comparison ---
models_5      = ["spaCy", "BERT-NER", "NLTK", "Gazetteer", "Flair"]
score_lists_5 = [spacy_scores, hf_scores, nltk_scores, gaz_scores, flair_scores]

summary_5 = {
    model: {
        "P":  sum(s["precision"] for s in sl) / len(sl),
        "R":  sum(s["recall"]    for s in sl) / len(sl),
        "F1": sum(s["f1"]        for s in sl) / len(sl),
    }
    for model, sl in zip(models_5, score_lists_5)
}

print("=" * 78)
print("FINAL 5-MODEL COMPARISON vs. Gold Standard (macro avg, strict exact match)")
print("=" * 78)
print(f"{'Metric':<12} {'spaCy':>8} {'BERT-NER':>10} {'NLTK':>8} {'Gazetteer':>11} {'Flair':>7}")
print("-" * 60)
for display_lbl, key in [("Precision", "P"), ("Recall", "R"), ("F1", "F1")]:
    row = f"{display_lbl:<12}"
    for m in models_5:
        row += f" {summary_5[m][key]:>10.2f}"
    print(row)

# --- Bar chart ---
x     = np.arange(len(models_5))
width = 0.22

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width, [summary_5[m]['P']  for m in models_5], width, label='Precision', color='#4C72B0')
ax.bar(x,         [summary_5[m]['R']  for m in models_5], width, label='Recall',    color='#DD8452')
ax.bar(x + width, [summary_5[m]['F1'] for m in models_5], width, label='F1',        color='#55A868')

ax.set_xlabel('NER System')
ax.set_ylabel('Score (macro avg, 0–1)')
ax.set_title('5-Model NER Comparison — Precision / Recall / F1\nvs. Gold Standard (strict exact match)')
ax.set_xticks(x)
ax.set_xticklabels(models_5)
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# --- Ranked summary ---
ranked = sorted(models_5, key=lambda m: summary_5[m]["F1"], reverse=True)
print("\nModels ranked by F1 (high → low):")
for rank, m in enumerate(ranked, 1):
    print(f"  {rank}. {m:<12}  P={summary_5[m]['P']:.2f}  R={summary_5[m]['R']:.2f}  F1={summary_5[m]['F1']:.2f}")

print("""
Key takeaways:
  spaCy     — strongest overall: broad label coverage (DATE, CARDINAL) maximizes recall
  Flair     — best span precision of the neural models; no subword fragmentation;
              DATE/CARDINAL gap limits recall, same as BERT-NER
  Gazetteer — precision reflects list completeness; regex fills DATE/CARDINAL gap;
              expanding the gazetteer directly raises recall
  NLTK      — competitive precision on covered types; blind to DATE/CARDINAL caps F1
  BERT-NER  — lowest F1 under strict match: subword boundary misalignment and
              missing DATE/CARDINAL labels hurt both precision and recall

Design guidance:
  Use spaCy when you need DATE/MONEY/CARDINAL in addition to named entities.
  Use Flair when clean span boundaries matter (downstream slot-filling, relation
    extraction) and the label set (ORG/PER/LOC/MISC) is sufficient.
  Ensemble Flair + Gazetteer for high-precision domain-specific pipelines.
""")
```
