# Walkthrough: Movie Plot Embedding Demo

This document explains each step and line of code in the notebook, using simple language.

---

## Introduction

This notebook demonstrates three ways to represent and compare short movie plot summaries: TF-IDF, GloVe, and DistilBERT. We use 8 curated movie summaries to show how each method captures similarity and meaning.

---

## 1. Setup: Movie Summaries and Titles

The notebook begins by loading GloVe word vectors — this happens before the movies are defined so that the GloVe dictionary is available for all later sections.

The four imports at the top each serve a distinct role: `numpy` is used throughout the notebook to create and manipulate numeric vectors; `os` provides `os.path.exists`, which lets us check for local files without raising exceptions; `urllib.request` handles the HTTP download in a single built-in call; and `zipfile` lets us read inside the zip archive without extracting every file to disk first.

Several constants are defined at the top so they are easy to change in one place:

- `GLOVE_FILE` — the local filename that will be read or created. Naming it here means you only have to update one line if you switch to a different pre-built subset.
- `GLOVE_URL` — the Stanford download URL for the full zip. Kept as a constant so `download_glove` does not need to hard-code it.
- `GLOVE_ZIP` — the name of the zip file saved locally. Separating it from `GLOVE_INNER` (below) reflects that the zip and the file inside it have different names.
- `GLOVE_INNER` — the filename inside the zip that contains 50-dimensional vectors. The full GloVe zip ships with multiple dimensionality options (50d, 100d, 200d, 300d); we pick the smallest so the notebook is fast.
- `MAX_WORDS` — how many lines (words) to extract from the full file; 50,000 is enough to cover all words in our movie summaries. The underscore in `50_000` is a Python readability convention — it has no effect on the value.

`download_glove` handles the full download-and-extract pipeline. `os.path.exists(GLOVE_ZIP)` is checked first so that a second run does not re-download the ~170 MB file if it is already on disk. `urllib.request.urlretrieve` streams the file from the URL and saves it directly to `GLOVE_ZIP` without loading it fully into memory. `zipfile.ZipFile(GLOVE_ZIP, 'r')` opens the archive in read-only mode, and `zf.open(GLOVE_INNER)` opens just the one internal file we need — no other files in the zip are touched. The `for i, raw_line in enumerate(src)` loop counts lines; once `i` reaches `MAX_WORDS` the loop breaks, so we never read the full 400k-line file. Each `raw_line` comes back as bytes, so `.decode('utf-8')` converts it to a regular Python string before writing it to `dst`.

`load_glove` first checks whether the local file already exists using `os.path.exists`. If it does not, it calls `download_glove` automatically rather than raising an error immediately. After the file is guaranteed to exist, the function opens it and iterates line by line. `line.split()` splits on any whitespace — the first element (`values[0]`) is the word itself, and the remaining elements (`values[1:]`) are the 50 numeric strings that form the vector. `np.array(values[1:], dtype='float32')` converts those strings to a compact 32-bit float array; `float32` halves memory use compared to the default `float64` while being precise enough for similarity comparisons. Each word-vector pair is stored in the `embeddings` dictionary so vectors can be retrieved by word in O(1) time.

If the auto-download itself fails (e.g. no internet connection), a `FileNotFoundError` is raised with instructions for a manual fix. The `from exc` clause at the end of the `raise` preserves the original exception as context so you can see the underlying network error alongside the human-readable message. `glove` is set to `None` in that case so the rest of the notebook can skip GloVe steps gracefully.

```python
import numpy as np
import os
import urllib.request
import zipfile

GLOVE_FILE = 'glove_50k_50d.txt'
GLOVE_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_ZIP = 'glove.6B.zip'
GLOVE_INNER = 'glove.6B.50d.txt'
MAX_WORDS = 50_000

def download_glove(target_path):
    """Download glove.6B.zip and save the top MAX_WORDS entries as target_path."""
    if not os.path.exists(GLOVE_ZIP):
        print(f'Downloading GloVe (~170 MB) from {GLOVE_URL} ...')
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
        print('Download complete.')
    print(f'Extracting top {MAX_WORDS:,} words from {GLOVE_INNER} ...')
    with zipfile.ZipFile(GLOVE_ZIP, 'r') as zf:
        with zf.open(GLOVE_INNER) as src, open(target_path, 'w', encoding='utf-8') as dst:
            for i, raw_line in enumerate(src):
                if i >= MAX_WORDS:
                    break
                dst.write(raw_line.decode('utf-8'))
    print(f"Saved '{target_path}'.")

def load_glove(filepath):
    if not os.path.exists(filepath):
        print(f"'{filepath}' not found — auto-downloading GloVe ...")
        try:
            download_glove(filepath)
        except Exception as exc:
            raise FileNotFoundError(
                f"Auto-download failed: {exc}\n"
                f"Manual fix: download {GLOVE_URL}, extract {GLOVE_INNER}, "
                f"rename/copy it to '{filepath}' next to this notebook."
            ) from exc
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

try:
    glove = load_glove(GLOVE_FILE)
    print(f'Loaded {len(glove):,} word vectors, dimension {len(next(iter(glove.values())))}')
except FileNotFoundError as e:
    print(e)
    glove = None
```

---

## 2. TF-IDF: Sparse Matrix Representation

The 8 movie plot summaries are defined as a plain Python list of strings. Each string is one document; the order here determines the row order in every matrix produced later, which is why the `titles` list is ordered to match exactly.

`from sklearn.feature_extraction.text import TfidfVectorizer` imports just the one class we need from scikit-learn rather than the whole library. `TfidfVectorizer(stop_words='english')` constructs the vectorizer; the `stop_words='english'` argument tells it to drop common words like "the", "a", and "is" from the vocabulary before computing scores, because those words appear in every document and carry no discriminating power.

`fit_transform(movies)` does two things in one call: `fit` scans all 8 summaries to build the vocabulary (every unique non-stop-word across all movies), and `transform` encodes each summary as a row of TF-IDF scores. The result is a **sparse matrix**, meaning only the non-zero cells are stored in memory — this is important because most words appear in only one or two summaries, so the matrix is mostly zeros.

`tfidf_matrix.shape` prints `(8, N)` where 8 is the number of movies and N is the vocabulary size. `tfidf_matrix.nnz` is a sparse-matrix property that counts only the stored (non-zero) cells. The sparsity formula `1 - nnz / (rows * cols)` computes what fraction of all possible cells are zero; a value close to 1.0 means the matrix is almost entirely empty.

```python
movies = [
    'A linguist is recruited by the military to communicate with aliens and must decode their language before global war breaks out.',
    'An ex-military team plans an elaborate heist to infiltrate a high-security vault, facing betrayals and explosive chases.',
    'An undercover agent races across continents to stop a global conspiracy, using disguises and daring escapes.',
    'A scientist faces a personal crisis after a failed experiment, questioning the ethics of discovery and the cost of ambition.',
    'A chef and a food critic fall in love amid kitchen chaos in a bustling restaurant, blending romance and comedy.',
    'A documentary explores efforts to protect ocean life, following activists and scientists fighting for conservation.',
    'A family drama about climate activists struggling to balance personal life and their mission to change environmental policy.',
    'In a dystopian future, artificial intelligence governs society, and a rebel group challenges the system to restore human rights.'
]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies)
print(f'Shape: {tfidf_matrix.shape}')
print(f'Non-zero entries: {tfidf_matrix.nnz}')
print(f'Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.1%}')
```

### Pairwise Similarity (TF-IDF)

The `titles` list is a short human-readable label for each movie; its order must match the `movies` list exactly because row/column index *i* in every similarity matrix refers to movie *i*.

`cosine_similarity(tfidf_matrix)` computes the dot product of every row pair divided by the product of their magnitudes, producing a value between 0 (no shared vocabulary) and 1 (identical vocabulary distribution). Because TF-IDF vectors are non-negative, the result is always in [0, 1]. The output is a full 8×8 NumPy array — every movie compared with every other movie, including itself along the diagonal (always 1.0).

`pd.DataFrame(tfidf_sim, index=titles, columns=titles)` wraps the raw array in a labelled table so rows and columns are identified by movie title rather than integer index. `.style.format('{:.3f}')` limits each cell to three decimal places. `.background_gradient(cmap='Blues')` shades each cell proportionally to its value using the Blues colour palette — darker blue means higher cosine similarity, making clusters of similar movies immediately visible.

```python
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
titles = [
    'Alien Linguist',
    'The Vault Job',
    'Agent Run',
    'Lab Crisis',
    'Kitchen Hearts',
    'Blue Planet Fight',
    'Climate Family',
    'AI Uprising'
]
tfidf_sim = cosine_similarity(tfidf_matrix)
sim_df = pd.DataFrame(tfidf_sim, index=titles, columns=titles)
display(sim_df.style.format('{:.3f}').background_gradient(cmap='Blues'))
```

Notice: action movies score highly with each other. The environmental documentary and the climate family drama share a theme, but TF-IDF rates them as dissimilar because they use different words. **TF-IDF only sees shared vocabulary.**

---

## 3. GloVe: Dense Word Embeddings

GloVe was already loaded in the first cell. Here we simply confirm it is available — if it failed to load earlier, we try once more and print a message either way.

```python
if glove is None:
    try:
        glove = load_glove('glove_50k_50d.txt')
        print(f"Loaded {len(glove)} word vectors, dimension {len(next(iter(glove.values())))}")
    except FileNotFoundError as e:
        print(e)
else:
    print(f"GloVe already loaded: {len(glove)} vectors")
```

### Word Similarity Examples (GloVe)

We define `cosine_sim` to measure how similar two word vectors are. `np.dot(a, b)` computes the sum of the element-wise products of the two vectors — larger when they point in the same direction. Dividing by `norm(a) * norm(b)` normalises by each vector's length so the result is always between -1 and 1 regardless of vector magnitude; for GloVe vectors the result is effectively between 0 and 1. A score near 1 means the words are closely related in meaning; near 0 means unrelated.

We then test four pairs to illustrate the range: 'ocean'/'sea' (near-synonyms, expected high), 'ocean'/'explosion' (unrelated, expected low), 'film'/'movie' (near-synonyms), and 'climate'/'environment' (thematically related). The check `if 'glove' in globals() and glove` skips this block gracefully when GloVe is unavailable — `in globals()` confirms the variable exists at all, and `and glove` confirms it is not `None`.

```python
from numpy.linalg import norm
if 'glove' in globals() and glove:
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    print(cosine_sim(glove['ocean'], glove['sea']))
    print(cosine_sim(glove['ocean'], glove['explosion']))
    print(cosine_sim(glove['film'], glove['movie']))
    print(cosine_sim(glove['climate'], glove['environment']))
else:
    print("GloVe embeddings not loaded. Please ensure the GloVe file is available and loaded.")
```

GloVe captures semantic similarity — 'ocean' and 'sea' score high, 'ocean' and 'explosion' score low. **But each word gets only one vector regardless of context.**

### Out-of-Vocabulary (OOV) Words

We check whether certain words exist in the GloVe dictionary. Common words like 'linguist' and 'alien' are usually present. Invented or very rare words like 'xenolinguist' are not.

```python
if glove:
    print('linguist' in glove)
    print('xenolinguist' in glove)
    print('alien' in glove)
else:
    print("GloVe embeddings not loaded. Skipping OOV check.")
```

---

## 4. GloVe Document Embeddings

We define `text_to_glove`, which turns a full summary into a single fixed-length vector.

`text.lower().split()` first converts the text to lowercase (so "Ocean" and "ocean" match the same dictionary key) and then splits on whitespace to produce a list of individual words. No punctuation stripping is done, which is why words like "future," may miss a GloVe match — a real pipeline would tokenise more carefully.

`[embeddings[w] for w in words if w in embeddings]` is a list comprehension that looks up each word in the GloVe dictionary and silently drops any word that is not found. This avoids a `KeyError` for out-of-vocabulary words.

`if not vectors` catches the edge case where every word in the text was out-of-vocabulary; in that case `np.zeros(50)` returns a 50-dimensional zero vector so downstream code always gets an array of the right shape.

`np.mean(vectors, axis=0)` stacks all the per-word vectors into a 2-D array and averages column-by-column, producing one 50-dimensional vector that represents the whole summary. `axis=0` means "collapse the rows (words) and average across them", leaving the 50 column dimensions intact.

`np.array([text_to_glove(m, glove) for m in movies])` applies this function to all 8 summaries and stacks the results into an (8, 50) matrix so pairwise similarity can be computed in one call.

```python
def text_to_glove(text, embeddings):
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    if not vectors:
        return np.zeros(50)
    return np.mean(vectors, axis=0)

if glove:
    glove_embeddings = np.array([text_to_glove(m, glove) for m in movies])
    print(f'Document embedding matrix: {glove_embeddings.shape}')
else:
    glove_embeddings = None
    print("GloVe not loaded. Skipping document embeddings.")
```

### Pairwise Similarity (GloVe)

We compute cosine similarity between the GloVe document embeddings. The result shows which movies are similar in meaning, even if they use different words. If GloVe was unavailable, `glove_sim` is set to `None` so the comparison step can handle it.

```python
if glove_embeddings is not None:
    glove_sim = cosine_similarity(glove_embeddings)
    glove_df = pd.DataFrame(glove_sim, index=titles, columns=titles)
    print(glove_df.round(3))
else:
    glove_sim = None
    print("GloVe embeddings not available. Skipping similarity matrix.")
```

Averaging word vectors loses word order and emphasis. But GloVe can detect some semantic similarity that TF-IDF misses entirely.

---

## 5. DistilBERT Embeddings

Before importing `torch`, the notebook removes any previously cached `torch` entries from `sys.modules`. `sys.modules` is Python's import cache — once a module is imported it stays there, so a second `import torch` simply retrieves the cached version without re-running the module's init code. The list comprehension `[k for k in sys.modules if k == "torch" or k.startswith("torch.")]` collects the keys for `torch` itself and all of its sub-modules (e.g. `torch.nn`, `torch.cuda`). Deleting them forces Python to reload the library from disk on the next `import torch`, avoiding subtle state issues that can occur when a Jupyter kernel has loaded an old or partially-initialised version in an earlier cell.

`AutoTokenizer.from_pretrained("distilbert-base-uncased")` downloads (or loads from cache) the vocabulary and configuration for DistilBERT's tokenizer. `AutoModel.from_pretrained("distilbert-base-uncased")` downloads the pre-trained model weights. The `"base-uncased"` suffix means this is the standard-size model trained on lowercased text — matching the lowercase normalisation we do in the GloVe section.

`model.eval()` switches the model from training mode to evaluation mode. In training mode PyTorch applies dropout (randomly zeroing out activations to prevent overfitting), which introduces randomness and makes results non-reproducible. In evaluation mode dropout is disabled, so the same input always produces the same output — which is what we want for embedding comparisons.

```python
import sys
for _m in [k for k in sys.modules if k == "torch" or k.startswith("torch.")]:
    del sys.modules[_m]

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()
```

### Tokenization Example

We tokenize the first summary and inspect the result. `tokenizer(text, return_tensors='pt', truncation=True, max_length=512)` does several things at once: it splits the text into sub-word tokens (for example "linguist" may become ["lin", "##guist"]), prepends a `[CLS]` token and appends a `[SEP]` token that BERT expects, converts each token to its integer ID in the model's vocabulary, and wraps everything in PyTorch tensors (`return_tensors='pt'`). `truncation=True` with `max_length=512` silently cuts any text longer than 512 tokens, which is BERT's hard limit.

`inputs["input_ids"].shape` shows `(1, N)` — a batch of 1 sequence with N tokens. `convert_ids_to_tokens(inputs["input_ids"][0])` maps the integer IDs back to human-readable tokens so you can see exactly how the tokenizer broke the text apart. The `[0]` index unwraps the batch dimension. The attention mask is a tensor of 1s for every real token and 0s for padding; when there is only one sequence (no padding needed) every value is 1.

```python
text = movies[0]
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
print(f'Input IDs shape: {inputs["input_ids"].shape}')
print(f'Tokens: {tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])}')
print(f'Attention mask: {inputs["attention_mask"]}')
```

### Extract Embedding (Mean Pooling)

We run the model on the tokenized input inside `torch.no_grad()`, which tells PyTorch not to build a computation graph for backpropagation. Since we are only doing inference (not training), this saves significant memory and time.

`outputs.last_hidden_state` is a tensor of shape `(1, N, 768)` — one 768-dimensional vector for every one of the N tokens in the sequence. Each vector reflects that token's meaning in the context of all surrounding tokens, which is BERT's key advantage over GloVe.

`inputs['attention_mask'].unsqueeze(-1)` reshapes the mask from `(1, N)` to `(1, N, 1)` so it can broadcast correctly against the `(1, N, 768)` hidden states. `hidden_states * attention_mask` zeroes out the vectors for any padding positions, ensuring they do not contribute to the average.

`masked_hidden.sum(dim=1)` collapses the token dimension by adding all the (now-masked) token vectors together, yielding shape `(1, 768)`. `attention_mask.sum(dim=1)` counts the number of real (non-padding) tokens so we can divide correctly. Dividing sum by count gives the mean. This is called **mean pooling** — it averages the contextualised representations of all real tokens into a single sentence vector, which tends to be more robust for similarity tasks than using only the `[CLS]` token.

`.squeeze()` removes the batch dimension (size 1), giving a 1-D array of shape `(768,)`. `.numpy()` converts the PyTorch tensor to a NumPy array so it works with scikit-learn's `cosine_similarity`.

```python
with torch.no_grad():
    outputs = model(**inputs)
hidden_states = outputs.last_hidden_state
attention_mask = inputs['attention_mask'].unsqueeze(-1)
masked_hidden = hidden_states * attention_mask
summed = masked_hidden.sum(dim=1)
counts = attention_mask.sum(dim=1)
mean_pooled = (summed / counts).squeeze().numpy()
print(f'Sentence embedding shape: {mean_pooled.shape}')
```

`extract_bert_embedding` packages the tokenise-→-forward-pass-→-mean-pool pipeline into a reusable function. The inline version `(hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)` fuses the masking, summing, and dividing into one expression rather than three separate variables, which is more concise but identical in behaviour to the step-by-step version above. The list comprehension `[extract_bert_embedding(m, tokenizer, model) for m in movies]` runs this for all 8 summaries, and `np.array(...)` stacks the results into an `(8, 768)` matrix.

```python
def extract_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    mean_pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    return mean_pooled.squeeze().numpy()

bert_embeddings = np.array([extract_bert_embedding(m, tokenizer, model) for m in movies])
print(f'BERT embedding matrix: {bert_embeddings.shape}')
```

### Pairwise Similarity (BERT)

We compute cosine similarity between the BERT embeddings. BERT captures meaning even when the words are completely different.

```python
bert_sim = cosine_similarity(bert_embeddings)
bert_df = pd.DataFrame(bert_sim, index=titles, columns=titles)
print(bert_df.round(3))
```

---

## 6. Compare Top-3 Similarities for Each Method

For three query movies (sci-fi, documentary, action heist), we show the top-3 most similar movies by each method. `query_indices = [0, 5, 1]` selects "Alien Linguist" (sci-fi), "Blue Planet Fight" (documentary), and "The Vault Job" (action heist) by their position in the `movies` list.

The `methods` list is built dynamically — GloVe is only appended `if glove_sim is not None` — so the comparison table is produced correctly whether or not the GloVe file was available.

Inside the loop, `scores = sim_matrix[qi].copy()` extracts the row for the query movie as a 1-D array of similarity scores against all 8 movies. `.copy()` is important because the next line modifies the array; without it we would mutate the original similarity matrix. `scores[qi] = -1` sets the query movie's own score to -1 so it cannot appear in the top-3 results (a movie is always most similar to itself, which would make the output trivial).

`np.argsort(scores)` returns the indices that would sort the array from smallest to largest. `[-3:]` takes the last three — the highest-scoring indices. `[::-1]` reverses them so the best match is first. The result is three indices pointing to the top-3 most similar movies for this query and method. `.ljust(25)` left-justifies each entry in a 25-character field so the columns line up evenly regardless of title length.

```python
query_indices = [0, 5, 1]  # sci-fi, documentary, action heist
methods = [('TF-IDF', tfidf_sim)]
if glove_sim is not None:
    methods.append(('GloVe', glove_sim))
methods.append(('BERT', bert_sim))

for qi in query_indices:
    print(f'\nQuery: {titles[qi]}')
    print(f'{"Method":<12} {"#1":<25} {"#2":<25} {"#3":<25}')
    for name, sim_matrix in methods:
        scores = sim_matrix[qi].copy()
        scores[qi] = -1  # exclude self
        top3 = np.argsort(scores)[-3:][::-1]
        row = f'{name:<12}'
        for idx in top3:
            row += f' {titles[idx]} ({scores[idx]:.3f})'.ljust(25)
        print(row)
```

### Discussion

- **TF-IDF** works well when movies share the same vocabulary, but misses thematic similarity when different words are used.
- **GloVe** captures some semantic similarity by understanding word meaning, but averaging word vectors loses word order and context.
- **BERT** is best at capturing meaning even when the words differ, because it reads the full sentence in context.

BERT is slower but more powerful. For large datasets, pre-compute and cache embeddings.
