# Module 6 Integrated Lab — Code Walkthrough

This document explains every line of code in the lab notebook. Each section covers one pipeline segment: what each line does, why it exists, and what to watch for. The full code block for that segment appears at the end of each section.

---

## Setup: Install the spaCy model

Before running the notebook, you need to download spaCy's small English model from the command line:

```bash
python -m spacy download en_core_web_sm
```

This command asks spaCy to pull the `en_core_web_sm` package from its model registry and install it into your Python environment. The model is approximately 12 MB and contains a pipeline for tokenization, part-of-speech tagging, dependency parsing, and named entity recognition (NER). You only need to run this once per environment.

---

## Segment 1 — Load and Preprocess Abstracts

### Line-by-line explanation

```python
import pandas as pd
```
Imports the pandas library and gives it the conventional alias `pd`. Pandas provides the `DataFrame` — a table structure that makes it easy to display, filter, and manipulate rows of text data.

```python
import spacy
```
Imports the spaCy NLP library. spaCy provides industrial-strength NLP tools including tokenization, part-of-speech tagging, and the NER pipeline used in Segment 2.

```python
import numpy as np
```
Imports NumPy, the foundational numerical computing library. All embedding vectors are stored as NumPy arrays. The alias `np` is universal convention.

```python
import torch
```
Imports PyTorch, the deep learning framework that runs the DistilBERT model. We use it to run inference (forward pass) without computing gradients.

```python
from transformers import AutoTokenizer, AutoModel
```
Imports two classes from the Hugging Face `transformers` library.
- `AutoTokenizer`: automatically selects the correct tokenizer for a given model checkpoint name. It converts raw text into token IDs that the model understands.
- `AutoModel`: automatically selects the correct model architecture for a given checkpoint. For `distilbert-base-uncased` it loads a 6-layer transformer encoder.

```python
from sklearn.metrics.pairwise import cosine_similarity
```
Imports the `cosine_similarity` function from scikit-learn. This function takes two 2D arrays and returns the cosine similarity between each pair of rows. It is used to compare a query embedding against all corpus embeddings at once.

```python
abstracts = [ ... ]
```
A Python list of 8 strings, one per research paper abstract. Each string is a single paragraph that describes an NLP paper. The list is the "corpus" — the set of documents we will search over.

```python
titles = [ ... ]
```
A parallel list of 8 short title strings, one per abstract. The index of a title matches the index of its abstract (e.g., `titles[0]` is the title for `abstracts[0]`). These are used for display only.

```python
df = pd.DataFrame({'title': titles, 'abstract': abstracts})
```
Creates a pandas DataFrame with two columns — `title` and `abstract` — by passing a dictionary where each key becomes a column name. This is useful for displaying the data as a table in the notebook.

```python
print(f"Loaded {len(df)} abstracts")
```
Prints a confirmation showing how many rows are in the DataFrame. `len(df)` returns the number of rows. The f-string embeds the value inline.

```python
df.head()
```
Renders the first 5 rows of the DataFrame as a formatted table in the notebook output. This is a visual sanity check that the data loaded correctly.

### Full code block

```python
import pandas as pd
import spacy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

abstracts = [
    "We present a fine-tuned BERT model for text classification on the SST-2 benchmark. "
    "Our approach achieves state-of-the-art results by leveraging pre-trained contextual "
    "representations. Experiments at Stanford demonstrate that fine-tuning with task-specific "
    "heads outperforms traditional feature-based methods on sentiment classification tasks.",

    "This work proposes a novel attention mechanism for neural machine translation that reduces "
    "computational complexity while maintaining translation quality. We evaluate on the WMT "
    "benchmark and demonstrate improvements over standard transformer architectures. "
    "Experiments conducted at Google show consistent gains across language pairs.",

    "We propose a BiLSTM-CRF architecture for sequence labeling in named entity recognition tasks. "
    "Our model is evaluated on the CoNLL-2003 benchmark and achieves competitive F1 scores. "
    "Character-level features are combined with word-level embeddings to capture morphological "
    "information, improving recognition of rare and out-of-vocabulary entities.",

    "We analyze sentiment in social media text collected from Twitter using a hybrid approach "
    "combining VADER lexicon scoring with a neural classifier. Our pipeline handles informal "
    "language, slang, and emojis common in user-generated content. Results show improved "
    "accuracy over purely lexicon-based or purely neural methods on social media sentiment benchmarks.",

    "This paper evaluates word embedding models including word2vec and GloVe on a suite of "
    "intrinsic and extrinsic benchmarks. We include WordSim-353 and analogy tasks to measure "
    "semantic and syntactic properties. Our analysis reveals that embedding dimensionality and "
    "training corpus size significantly impact downstream NLP task performance.",

    "We investigate multilingual named entity recognition for low-resource languages using "
    "mBERT as a cross-lingual encoder. Fine-tuning on WikiAnn annotations enables zero-shot "
    "transfer to unseen languages. Our experiments show that typologically similar languages "
    "benefit most from cross-lingual transfer, while morphologically complex languages remain challenging.",

    "XLM-RoBERTa achieves strong zero-shot cross-lingual transfer on the XNLI benchmark for "
    "natural language inference. We analyze the factors contributing to cross-lingual generalization "
    "and find that multilingual pre-training on diverse corpora substantially outperforms "
    "bilingual models, even when the target language has limited training data.",

    "We present a retrieval-augmented generation (RAG) framework for open-domain question answering "
    "using Dense Passage Retrieval (DPR) over the Natural Questions dataset. Our system retrieves "
    "relevant passages from a large corpus and conditions a generative model on retrieved context. "
    "This approach significantly outperforms closed-book models that rely solely on parametric memory."
]

titles = [
    "BERT Classification",
    "Attention MT",
    "BiLSTM-CRF NER",
    "Social Media Sentiment",
    "Word Embedding Benchmarks",
    "Multilingual NER",
    "Cross-Lingual Transfer",
    "RAG Question Answering"
]

df = pd.DataFrame({'title': titles, 'abstract': abstracts})
print(f"Loaded {len(df)} abstracts")
df.head()
```

---

## Segment 2 — Run spaCy NER to Extract Entities

### Line-by-line explanation

```python
nlp = spacy.load('en_core_web_sm')
```
Loads the small English NLP pipeline that was downloaded in the setup step. The return value `nlp` is a callable object: calling `nlp(text)` runs the full pipeline (tokenizer → tagger → parser → NER) on the input string and returns a `Doc` object.

```python
def extract_entities(texts, nlp):
```
Defines a function that takes a list of strings (`texts`) and a loaded spaCy pipeline (`nlp`). Separating this logic into a function makes it reusable and testable.

```python
    all_entities = []
```
Initializes an empty list that will hold one dictionary per input text. After the loop, `all_entities[i]` will contain the entities found in `texts[i]`.

```python
    for text in texts:
```
Iterates over each string in the input list.

```python
        doc = nlp(text)
```
Runs the full spaCy pipeline on the current text. `doc` is a `Doc` object that exposes tokenized text, part-of-speech tags, dependency parse, and named entities.

```python
        entities = {}
```
Creates a fresh dictionary for this document. Keys will be NER label strings (e.g., `"ORG"`, `"DATE"`); values will be lists of entity text strings.

```python
        for ent in doc.ents:
```
Iterates over all named entity spans detected in the document. `doc.ents` is a tuple of `Span` objects; each `Span` has a `.text` attribute (the matched string) and a `.label_` attribute (the NER category).

```python
            label = ent.label_
```
Extracts the entity type as a string, e.g., `"ORG"`, `"GPE"`, `"CARDINAL"`. The trailing underscore is a spaCy convention: `.label` returns the integer ID; `.label_` returns the human-readable string.

```python
            if label not in entities:
                entities[label] = []
```
If this label has not been seen yet in this document, creates an empty list for it. This avoids a `KeyError` on the next line.

```python
            if ent.text not in entities[label]:
                entities[label].append(ent.text)
```
Appends the entity text to its label's list, but only if it is not already there. This deduplicates entities that appear multiple times in the same abstract.

```python
        all_entities.append(entities)
```
After processing all entities in the current document, appends the entity dictionary to the results list.

```python
    return all_entities
```
Returns the list of entity dictionaries — one per input text.

```python
entities = extract_entities(abstracts, nlp)
```
Calls the function on the 8 abstracts and stores the result.

```python
for i, (title, ents) in enumerate(zip(titles, entities)):
    print(f"\n{title}:")
    for label, values in ents.items():
        print(f"  {label}: {values}")
```
Loops over titles and their corresponding entity dictionaries simultaneously. `zip` pairs them by index. `enumerate` provides the loop counter `i` (not used here but useful for debugging). The inner loop prints each entity type and its associated values.

### What to look for in the output

- `Stanford` → correctly labeled `ORG`
- `BERT` → may be labeled `ORG`, `PRODUCT`, or missed entirely
- `CoNLL-2003` → may be labeled `DATE` (spaCy sees the year pattern) or missed
- `SST-2`, `WMT`, `WikiAnn` → likely missed or mislabeled

This reveals a genuine limitation: `en_core_web_sm` was trained on general web text, not NLP research papers. Domain-specific entities require fine-tuning or a specialized model.

### Full code block

```python
nlp = spacy.load('en_core_web_sm')

def extract_entities(texts, nlp):
    """Extract named entities from a list of texts using spaCy."""
    all_entities = []
    for text in texts:
        doc = nlp(text)
        entities = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            if ent.text not in entities[label]:
                entities[label].append(ent.text)
        all_entities.append(entities)
    return all_entities

entities = extract_entities(abstracts, nlp)

print("Extracted entities per abstract:\n")
for i, (title, ents) in enumerate(zip(titles, entities)):
    print(f"{title}:")
    if ents:
        for label, values in ents.items():
            print(f"  {label}: {values}")
    else:
        print("  (no entities detected)")
    print()
```

---

## Segment 3 — Compute DistilBERT Embeddings

### Line-by-line explanation

```python
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
```
Downloads (or loads from cache) the tokenizer for `distilbert-base-uncased`. This tokenizer uses WordPiece subword tokenization — it splits words into known sub-units so the model can handle rare or unseen words. `from_pretrained` checks a local cache first; if not found it downloads from the Hugging Face Hub.

```python
model = AutoModel.from_pretrained('distilbert-base-uncased')
```
Downloads (or loads from cache) the DistilBERT model weights. DistilBERT is a 6-layer, 768-hidden-dimension transformer encoder — a distilled (compressed) version of BERT. It is ~40% smaller and ~60% faster while retaining ~97% of BERT's performance on downstream tasks.

```python
model.eval()
```
Sets the model to evaluation mode. In training mode, layers like dropout and batch normalization behave differently (stochastic) than at inference time (deterministic). Calling `.eval()` freezes this behavior to the deterministic inference mode. Always call this before running inference.

```python
def compute_embedding(text, tokenizer, model):
```
Defines the embedding function. It takes a single string, the tokenizer, and the model.

```python
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
```
Tokenizes the input text.
- `return_tensors='pt'`: returns PyTorch tensors instead of plain Python lists.
- `truncation=True`: silently truncates any text longer than `max_length` tokens. DistilBERT has a maximum context window of 512 tokens.
- `max_length=512`: the maximum number of tokens.

The result `inputs` is a dictionary with at least two keys: `input_ids` (the token ID tensor, shape `(1, seq_len)`) and `attention_mask` (a binary mask tensor, shape `(1, seq_len)`, where `1` means "real token" and `0` means "padding").

```python
    with torch.no_grad():
        outputs = model(**inputs)
```
Runs the forward pass through DistilBERT. `torch.no_grad()` tells PyTorch not to build a computation graph, which saves memory and speeds up inference (we do not need gradients because we are not training). `**inputs` unpacks the dictionary as keyword arguments to the model.

`outputs.last_hidden_state` is a tensor of shape `(1, seq_len, 768)` — one 768-dimensional vector per token, for all tokens in the sequence.

```python
    hidden_states = outputs.last_hidden_state
```
Extracts the final layer's hidden states. Shape: `(1, seq_len, 768)`.

```python
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
```
Retrieves the attention mask and adds a trailing dimension so it can broadcast against `hidden_states`. Shape goes from `(1, seq_len)` to `(1, seq_len, 1)`. This allows element-wise multiplication with the `(1, seq_len, 768)` hidden states.

```python
    mean_pooled = (
        (hidden_states * attention_mask).sum(dim=1)
        / attention_mask.sum(dim=1)
    )
```
Computes a masked mean pool. Multiplying by `attention_mask` zeroes out any padding token vectors. Summing over `dim=1` (the sequence length dimension) collapses all token vectors into one. Dividing by the count of real tokens (`attention_mask.sum(dim=1)`) gives the true mean of non-padding token representations. The result has shape `(1, 768)`.

This is called **mean pooling** and it produces a single fixed-length vector that represents the entire text.

```python
    return mean_pooled.squeeze().numpy()
```
`squeeze()` removes the batch dimension, going from shape `(1, 768)` to `(768,)`. `.numpy()` converts the PyTorch tensor to a NumPy array so it can be used with scikit-learn and NumPy functions.

```python
corpus_embeddings = np.array([
    compute_embedding(a, tokenizer, model) for a in abstracts
])
```
Calls `compute_embedding` for each of the 8 abstracts and stacks the resulting `(768,)` arrays into a single `(8, 768)` NumPy array. This is the embedded corpus that will be searched.

```python
print(f"Corpus embeddings shape: {corpus_embeddings.shape}")
```
Confirms the shape is `(8, 768)` — 8 documents, each with a 768-dimensional embedding.

### Full code block

```python
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
model.eval()

def compute_embedding(text, tokenizer, model):
    """Compute a mean-pooled sentence embedding using DistilBERT."""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state             # shape: (1, seq_len, 768)
    attention_mask = inputs['attention_mask'].unsqueeze(-1)  # shape: (1, seq_len, 1)
    mean_pooled = (
        (hidden_states * attention_mask).sum(dim=1)
        / attention_mask.sum(dim=1)
    )                                                     # shape: (1, 768)
    return mean_pooled.squeeze().numpy()                  # shape: (768,)

print("Computing corpus embeddings...")
corpus_embeddings = np.array([
    compute_embedding(a, tokenizer, model) for a in abstracts
])
print(f"Corpus embeddings shape: {corpus_embeddings.shape}")
```

---

## Segment 4 — Implement Semantic Search on a Query

### Line-by-line explanation

```python
def semantic_search(query, corpus_embeddings, tokenizer, model, top_n=5):
```
Defines the search function. It takes a query string, the pre-computed corpus embeddings, the tokenizer and model (needed to embed the query), and `top_n` — how many results to return (default 5).

```python
    query_embedding = compute_embedding(query, tokenizer, model).reshape(1, -1)
```
Embeds the query using the same function as the corpus. `compute_embedding` returns a `(768,)` array. `.reshape(1, -1)` reshapes it to `(1, 768)` — a 2D array with one row — because `cosine_similarity` expects 2D inputs.

```python
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
```
Computes the cosine similarity between the query vector `(1, 768)` and the corpus matrix `(8, 768)`. The result is a `(1, 8)` array; `[0]` extracts the single row to get a flat `(8,)` array of similarity scores, one per document.

Cosine similarity ranges from `-1` (opposite direction) to `1` (identical direction). For text embeddings it typically stays between `0.7` and `1.0` for similar texts.

```python
    top_indices = np.argsort(similarities)[::-1][:top_n]
```
`np.argsort` returns the indices that would sort the array in ascending order. `[::-1]` reverses it to descending order (highest similarity first). `[:top_n]` slices the top N indices.

```python
    return [(idx, similarities[idx]) for idx in top_indices]
```
Returns a list of `(index, score)` tuples for the top-N results. The index can be used to look up the original abstract, title, or entities.

```python
query1 = "How do transformers improve text classification?"
results1 = semantic_search(query1, corpus_embeddings, tokenizer, model, top_n=5)
```
Defines a natural language query and runs semantic search. Expect "BERT Classification" (Abstract 1) and "Attention MT" (Abstract 2) to rank highly — both are about transformer architectures.

```python
for rank, (idx, score) in enumerate(results1, 1):
    print(f"  #{rank} (score: {score:.4f}) — {titles[idx]}")
```
Prints a numbered ranking list. `enumerate(results1, 1)` starts the counter at 1 instead of 0. The `:0.4f` format shows scores to 4 decimal places.

### Full code block

```python
def semantic_search(query, corpus_embeddings, tokenizer, model, top_n=5):
    """Embed a query and retrieve the top-N most similar corpus documents."""
    query_embedding = compute_embedding(query, tokenizer, model).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [(idx, similarities[idx]) for idx in top_indices]

query1 = "How do transformers improve text classification?"
results1 = semantic_search(query1, corpus_embeddings, tokenizer, model, top_n=5)

print(f"Query: {query1}\n")
for rank, (idx, score) in enumerate(results1, 1):
    print(f"  #{rank} (score: {score:.4f}) — {titles[idx]}")
```

---

## Segment 5 — Enrich Results with Entities

### Line-by-line explanation

```python
def display_results(query, results, abstracts, entities, titles):
```
Defines a display function that brings together all four data structures — query, ranked results, raw abstracts, extracted entities, and titles — into a single formatted output.

```python
    print(f"Query: {query}")
    print("=" * 60)
```
Prints the query and a separator line (60 equals signs) for visual clarity.

```python
    for rank, (idx, score) in enumerate(results, 1):
```
Iterates over the results list. Each element is a `(idx, score)` tuple. `enumerate(..., 1)` provides a 1-based rank counter.

```python
        print(f"\n  #{rank} — {titles[idx]} (similarity: {score:.4f})")
```
Prints the rank number, the paper title (looked up by index), and the similarity score.

```python
        print(f"  Preview: {abstracts[idx][:200]}...")
```
Prints the first 200 characters of the abstract followed by `...`. This gives the reader a sense of the paper's content without showing the entire text.

```python
        if entities[idx]:
            for label, values in entities[idx].items():
                print(f"  {label}: {', '.join(values)}")
        else:
            print("  No entities detected.")
```
If entities were found for this abstract, prints each entity type and its values joined by commas. If no entities were found (the dictionary is empty), prints a notice. This is where the two pipelines — NER and embeddings — combine to give the user both a relevance score and structured entity information.

```python
display_results(query1, results1[:3], abstracts, entities, titles)
```
Shows the top-3 enriched results for the first query. `results1[:3]` slices the list to the first three elements.

```python
query2 = "What methods work for named entity recognition?"
results2 = semantic_search(query2, corpus_embeddings, tokenizer, model, top_n=3)
display_results(query2, results2, abstracts, entities, titles)
```
Runs a second query specifically about NER. Expect "BiLSTM-CRF NER" (Abstract 3) and "Multilingual NER" (Abstract 6) to rank at the top, with "Cross-Lingual Transfer" (Abstract 7) possibly appearing because it shares the multilingual theme.

### Full code block

```python
def display_results(query, results, abstracts, entities, titles):
    """Display ranked search results with abstract preview and extracted entities."""
    print(f"Query: {query}")
    print("=" * 60)
    for rank, (idx, score) in enumerate(results, 1):
        print(f"\n  #{rank} — {titles[idx]} (similarity: {score:.4f})")
        print(f"  Preview: {abstracts[idx][:200]}...")
        if entities[idx]:
            for label, values in entities[idx].items():
                print(f"  {label}: {', '.join(values)}")
        else:
            print("  No entities detected.")
    print()

# Top-3 enriched results for the first query
display_results(query1, results1[:3], abstracts, entities, titles)

# Second query targeting NER papers
query2 = "What methods work for named entity recognition?"
results2 = semantic_search(query2, corpus_embeddings, tokenizer, model, top_n=3)
display_results(query2, results2, abstracts, entities, titles)
```

---

## Segment 6 — Compare to Keyword Search

### Line-by-line explanation

```python
def keyword_search(query, abstracts, top_n=5):
```
Defines a simple keyword search function. Unlike semantic search, this function does not need the model or tokenizer — it works purely on raw strings.

```python
    keywords = query.lower().split()
```
Lowercases the query and splits it on whitespace to get a list of keyword strings. For the query `"How do transformers improve text classification?"` this produces `['how', 'do', 'transformers', 'improve', 'text', 'classification?']`. Note that punctuation is not stripped — a more robust implementation would use regex.

```python
    scores = []
    for i, abstract in enumerate(abstracts):
```
Initializes a list to hold `(index, score)` pairs and loops over all abstracts with their indices.

```python
        text_lower = abstract.lower()
```
Lowercases the abstract text for case-insensitive matching.

```python
        count = sum(1 for kw in keywords if kw in text_lower)
```
Counts how many of the query keywords appear as substrings in the abstract. `sum(1 for ...)` is a generator expression that counts the number of `True` conditions. This is a simple bag-of-words overlap score.

```python
        scores.append((i, count / len(keywords)))
```
Appends a tuple of `(index, normalized_score)`. Dividing by `len(keywords)` normalizes the score to `[0, 1]` — where `1.0` means all query keywords appeared.

```python
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]
```
Sorts results by score descending and returns the top N.

```python
print(f"{'Rank':<6} {'Semantic Search':<35} {'Keyword Search':<35}")
print("-" * 76)
for rank, ((s_idx, s_score), (k_idx, k_score)) in enumerate(zip(sem_results, kw_results), 1):
    print(f"{rank:<6} {titles[s_idx]:<35} {titles[k_idx]:<35}")
```
Formats a side-by-side comparison table. `:<6` and `:<35` are left-aligned column width specifiers in f-strings. `zip(sem_results, kw_results)` pairs the two result lists element by element.

### What to look for

The interesting case is when the two rankings disagree. Semantic search may rank a paper highly because its meaning is close to the query, even if it does not contain the exact query words. For example, a paper that says "encoder-decoder architecture" may rank higher in semantic search than keyword search for the query "transformers," because both refer to the same architectural family.

### Full code block

```python
def keyword_search(query, abstracts, top_n=5):
    """Score each document by the fraction of query keywords it contains."""
    keywords = query.lower().split()
    scores = []
    for i, abstract in enumerate(abstracts):
        text_lower = abstract.lower()
        count = sum(1 for kw in keywords if kw in text_lower)
        scores.append((i, count / len(keywords)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

comparison_query = "How do transformers improve text classification?"

sem_results = semantic_search(comparison_query, corpus_embeddings, tokenizer, model, top_n=5)
kw_results  = keyword_search(comparison_query, abstracts, top_n=5)

print(f"Query: {comparison_query}\n")
print(f"{'Rank':<6} {'Semantic Search':<35} {'Keyword Search':<35}")
print("-" * 76)
for rank, ((s_idx, s_score), (k_idx, k_score)) in enumerate(zip(sem_results, kw_results), 1):
    print(f"{rank:<6} {titles[s_idx]:<35} {titles[k_idx]:<35}")
```

---

## Segment 7 — Production Improvements (Optional Extension)

Save embeddings to disk with `np.save` so they can be reloaded on future runs instead of recomputed. `np.load` restores the array at startup. `np.allclose` verifies the round-trip preserved all values within floating-point tolerance.

### Code

```python
# Save embeddings to disk for reuse
np.save('corpus_embeddings.npy', corpus_embeddings)
print("Embeddings saved to corpus_embeddings.npy")

# Load them back
loaded_embeddings = np.load('corpus_embeddings.npy')
print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
print("Shapes match:", np.allclose(corpus_embeddings, loaded_embeddings))
```

---

## Conceptual Summary

| Pipeline Stage | What it does | Library |
|---|---|---|
| Load data | Store raw text in a list + DataFrame | `pandas` |
| NER extraction | Identify named entities by type | `spacy` |
| Tokenization | Convert text to token IDs | `transformers.AutoTokenizer` |
| Embedding | Convert token IDs to a 768-dim vector | `transformers.AutoModel` + `torch` |
| Similarity | Compare query vector to corpus matrix | `sklearn.metrics.pairwise.cosine_similarity` |
| Ranking | Sort by similarity score | `numpy.argsort` |
| Enrichment | Attach entity metadata to ranked results | Python dict lookups |

The enrichment step is the integration point: it combines the Week A output (entities) with the Week B output (ranked results by embedding similarity) into a single user-facing response.
