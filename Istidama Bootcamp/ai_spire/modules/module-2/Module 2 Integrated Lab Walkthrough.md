# Module 2 Integrated Lab — Code Walkthrough

This document walks every line of [train.py](train.py). Each section starts with a line-by-line explanation, then shows the full code block at the end so you can copy-paste it cleanly.

The script trains a small feed-forward network to predict monthly electricity consumption (kWh) for commercial buildings from five building features — the differentiated support-instructor demo for M2-I2-pytorch. Read alongside `train.py`.

---

## 0. Header — module docstring, imports, and section banner

### Why this section exists
Before any code runs, the reader needs to know what this file does, what domain it's demonstrating, and what libraries it depends on. The docstring also records the exact architecture/optimizer/loss/epoch settings so a reader doesn't have to scroll through the whole file to find them.

### Line-by-line

- `"""train.py — Electricity Consumption Prediction Pipeline ... """` — the module-level docstring. It states the file's purpose (demo script for M2-I2-pytorch, support-instructor use only), the domain (monthly electricity consumption for commercial buildings), and a compact summary of the architecture (`Linear(5 → 32) → ReLU → Linear(32 → 1)`), optimizer (`Adam(lr=0.01)`), loss (`MSELoss`), and training length (100 epochs, printed every 10). Anyone opening the file gets the full picture before reading a single line of code — this mirrors how the lab's own `train.py` should be documented.
- `import os` — the standard library's operating-system interface. Used later for exactly one thing: `os.path.exists(filepath)` in `load_data`, to check the dataset file is present before pandas tries to read it.
- `import pandas as pd` — the DataFrame library used to read the CSV, select feature/target columns, and compute per-column mean/std for standardization. `pd` is the near-universal alias, so any reader familiar with pandas recognizes it instantly.
- `import torch` — the core PyTorch package. Supplies `torch.Tensor`, `torch.tensor(...)`, `torch.optim`, and `torch.no_grad()`, all used later in the file.
- `import torch.nn as nn` — PyTorch's neural-network building-block module, aliased `nn` by convention. Supplies `nn.Module` (the base class every PyTorch model subclasses), `nn.Linear`, `nn.ReLU`, and `nn.MSELoss`.
- `# ───── 1. Model definition ─────` — the first of five ASCII banner comments that divide the file into numbered sections. These aren't executed code — they're a lightweight table of contents so a reader scrolling through a 250-line script can jump straight to "training loop" or "main" without reading everything above it. This walkthrough follows the same section numbers train.py uses internally.

### Full code block

```python
"""
train.py — Electricity Consumption Prediction Pipeline
Demo script for M2-I2-pytorch (Support Instructor use only).
Domain: Monthly electricity consumption for commercial buildings.

Architecture:
    EnergyModel: Linear(5 → 32) → ReLU → Linear(32 → 1)
Optimizer: Adam(lr=0.01)
Loss:     MSELoss
Epochs:   100, print every 10
"""

import os
import pandas as pd
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model definition
# ─────────────────────────────────────────────────────────────────────────────
```

---

## 1. Model definition — `EnergyModel`

### Why this section exists
This is the actual neural network. It has to be a `nn.Module` subclass so PyTorch can track its parameters, move it to a device, and hand it to an optimizer. Keeping the architecture tiny (one hidden layer) keeps the demo fast and the loss curve easy to read on a projector.

### Line-by-line

- `class EnergyModel(nn.Module):` — declares a new model class that inherits from `nn.Module`. Subclassing `nn.Module` is what makes `model.parameters()`, `model.to(device)`, `model.eval()`, `model.train()`, and `print(model)` all work automatically — none of that machinery exists if you just write a plain Python class.
- The class docstring — documents the architecture (`Input (5 features) → Linear(5, 32) → ReLU → Linear(32, 1)`) and the two constructor arguments, so `help(EnergyModel)` or hovering in an editor shows the shape without opening `forward`.
- `def __init__(self, input_size: int = 5, hidden_size: int = 32):` — the constructor. Both dimensions are parameterized with defaults matching the dataset (5 input features) and a reasonable hidden width (32 units), so the class stays reusable for other tabular regression problems without editing the class body.
- `super().__init__()` — calls the parent `nn.Module` constructor **before** any layers are assigned. This line is marked `REQUIRED` in the inline comment for good reason: `nn.Module.__setattr__` is overridden to register any `nn.Module`-typed attribute (like `nn.Linear`) into an internal `_modules` dict. That registration machinery is set up inside `nn.Module.__init__`. Skip `super().__init__()` and `self.fc1 = nn.Linear(...)` silently becomes a plain attribute — `model.parameters()` returns an empty generator, and `optimizer.step()` has nothing to update. The model would appear to run but never learn.
- `self.fc1 = nn.Linear(input_size, hidden_size)` — the first fully-connected (dense) layer, mapping the 5 raw features to a 32-dimensional hidden representation. `nn.Linear` internally holds a weight matrix of shape `(hidden_size, input_size)` and a bias vector of shape `(hidden_size,)`, both `float32` by default and both registered as trainable parameters the moment they're assigned to `self`.
- `self.relu = nn.ReLU()` — a non-linear activation layer (`max(0, x)` element-wise). Without a non-linearity between the two `Linear` layers, stacking `Linear → Linear` is mathematically equivalent to a single `Linear` layer (matrix multiplication composes linearly) — the "hidden layer" would add capacity in name only. ReLU is stored as a layer object (rather than called as a bare function) purely for consistency with the other layers and so it shows up in `print(model)`.
- `self.fc2 = nn.Linear(hidden_size, 1)` — the output layer, collapsing the 32-dimensional hidden vector down to a single scalar: the predicted monthly kWh. Output size is `1` because this is a regression task with one continuous target, not a classifier with multiple class logits.
- `def forward(self, x: torch.Tensor) -> torch.Tensor:` — defines the forward pass: how input tensors flow through the layers to produce output. PyTorch calls this method automatically whenever you write `model(x)` (via `nn.Module.__call__`, which also handles hooks) — you should call `model(x)`, not `model.forward(x)`, in normal use so those hooks still fire.
- `"""Run a forward pass through the network."""` — a one-line docstring; the method's short and self-explanatory, so a longer docstring would just restate the code.
- `out = self.fc1(x)` — applies the first linear transformation. `x` has shape `(N, 5)` (N rows/buildings, 5 features each); `out` has shape `(N, 32)`.
- `out = self.relu(out)` — applies ReLU element-wise; shape is unchanged, `(N, 32)`, but negative activations are zeroed out.
- `out = self.fc2(out)` — applies the second linear transformation, collapsing `(N, 32)` down to `(N, 1)` — one predicted kWh value per row.
- `return out` — returns the `(N, 1)` tensor of predictions to the caller. Note there is no final activation function (like `sigmoid` or `softmax`) — this is deliberate for regression: the raw linear output is the prediction itself, unconstrained in range, which is what `nn.MSELoss` expects to compare against the raw target values.
- `# ───── 2. Data loading & preparation ─────` — the next banner comment in the file, marking the boundary between the model definition above and the three data-preparation functions below (`load_data`, `standardize`, `to_tensors`). Like the section 1 banner, it has no runtime effect — it's a table-of-contents marker for anyone scrolling the raw script.

### Full code block

```python
class EnergyModel(nn.Module):
    """Feed-forward regression network for monthly kWh prediction.

    Architecture:
        Input (5 features) → Linear(5, 32) → ReLU → Linear(32, 1)

    Args:
        input_size: Number of input features (default 5).
        hidden_size: Width of the single hidden layer (default 32).
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 32):
        # super().__init__() is REQUIRED — registers layers as model parameters.
        # Without it, model.parameters() returns nothing and the optimizer
        # has nothing to update.
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading & preparation
# ─────────────────────────────────────────────────────────────────────────────
```

---

## 2. Data loading & preparation

### 2a. `load_data`

#### Why this section exists
Every training run needs the raw CSV turned into separated feature/target DataFrames before anything else can happen. Failing loudly and early if the file is missing saves a confusing stack trace three functions later inside `pandas`.

#### Line-by-line

- `def load_data(filepath: str = "demo_data/electricity.csv"):` — the default path matches the repo's `demo_data/` layout, so calling `load_data()` with no arguments "just works" when run from the project root, while still letting a caller point at a different CSV for testing.
- The docstring documents the args and return value (`Tuple of (X DataFrame, y Series)`) — useful for anyone importing this function from `tests/test_train.py` without re-reading the body.
- `if not os.path.exists(filepath):` — an explicit existence check performed **before** calling `pd.read_csv`. This is a deliberate "fail fast with a helpful message" pattern.
- `raise FileNotFoundError(f"Dataset not found at '{filepath}'. Make sure demo_data/electricity.csv is present.")` — raises a specific, descriptive exception instead of letting `pd.read_csv` raise its own less-friendly `FileNotFoundError` (whose message is just the OS-level errno text). The custom message tells a learner exactly what's missing and where it should be, which matters a lot when this script is run by someone unfamiliar with the repo layout.
- `df = pd.read_csv(filepath)` — reads the entire CSV into a DataFrame in one call. For a 150-row file this is trivially fast; pandas infers column dtypes automatically (all numeric columns here come in as `int64`/`float64`).
- `feature_cols = ["area_sqm", "num_floors", "occupants", "insulation_rating", "has_ac"]` — names the five input columns explicitly as a list, rather than inferring them (e.g. "all columns except the last"). Being explicit means if the CSV schema changes — a column gets added or reordered — this line still selects exactly the intended features instead of silently picking up something new.
- `X = df[feature_cols]` — selects those five columns as a new DataFrame, shape `(150, 5)`. This is the model's input.
- `y = df[["monthly_kwh"]]  # double brackets → shape (N, 1)` — selects the target column. The inline comment calls out a common pandas gotcha: `df["monthly_kwh"]` (single brackets) returns a 1-D `Series` of shape `(N,)`, while `df[["monthly_kwh"]]` (double brackets, a list containing one column name) returns a 2-D `DataFrame` of shape `(N, 1)`. That distinction matters downstream — `to_tensors` needs a 2-D target so it lines up with the model's `(N, 1)` output shape for `MSELoss`; a mismatched `(N,)` vs `(N, 1)` shape is a classic silent bug in PyTorch (broadcasting quietly computes the wrong loss instead of erroring).
- `print(f"[load_data] {len(df)} rows | features: {feature_cols}")` — a visible confirmation line, prefixed with the function name in brackets (a lightweight logging convention used consistently across this file) so console output is traceable to its source even without a full logging framework.
- `return X, y` — returns the raw (not yet standardized) feature and target DataFrames to the caller.

#### Full code block

```python
def load_data(filepath: str = "demo_data/electricity.csv"):
    """Load the electricity dataset and return feature/target DataFrames.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Make sure demo_data/electricity.csv is present."
        )
    df = pd.read_csv(filepath)
    feature_cols = ["area_sqm", "num_floors", "occupants", "insulation_rating", "has_ac"]
    X = df[feature_cols]
    y = df[["monthly_kwh"]]  # double brackets → shape (N, 1)
    print(f"[load_data] {len(df)} rows | features: {feature_cols}")
    return X, y
```

### 2b. `standardize`

#### Why this section exists
The raw features live on wildly different scales — `area_sqm` runs into the thousands while `insulation_rating` runs 1–5. Feeding that straight into Adam produces unstable, inefficient training. This function is the fix, and the docstring explains why in concrete terms rather than just asserting "normalize your data."

#### Line-by-line

- `def standardize(X: pd.DataFrame):` — takes the raw feature DataFrame and returns a rescaled version plus the statistics used to rescale it.
- The docstring spells out the failure mode this function prevents: without standardization, `area_sqm` (hundreds to thousands) numerically overwhelms `insulation_rating` (1–5); gradients for the large-scale feature dominate the weight update; and the optimizer moves inefficiently, so loss stagnates or diverges. This is the single most load-bearing comment in the file — it's the answer to "why does this extra step exist at all?"
- `mean = X.mean()` — computes the per-column arithmetic mean across all 150 rows. Returns a `pandas.Series` indexed by column name, one scalar mean per feature.
- `std = X.std()` — computes the per-column standard deviation (pandas defaults to the sample standard deviation, dividing by `N-1`). Also a `Series` indexed by column name.
- `X_std = (X - mean) / std` — applies z-score standardization: subtract each column's mean, then divide by its standard deviation. Because `mean` and `std` are `Series` aligned by column label, this single vectorized expression broadcasts correctly across every column simultaneously — no explicit loop needed. After this line, every column in `X_std` has mean ≈ 0 and standard deviation ≈ 1, putting all five features on a comparable scale.
- `return X_std, mean, std` — returns not just the standardized features but also `mean` and `std` themselves. The docstring notes why: those two values are needed to apply the *exact same* transformation to any new data later (e.g. at inference time on unseen buildings) — recomputing mean/std from a different dataset would standardize inconsistently and corrupt predictions.

#### Full code block

```python
def standardize(X: pd.DataFrame):
    """Standardise features to zero mean and unit variance.

    This is the single most important preprocessing step. Without it:
    - area_sqm (hundreds) overwhelms insulation_rating (1–5)
    - Gradients for large-scale features dominate the update
    - The optimizer moves inefficiently — loss stagnates or diverges

    Args:
        X: Raw feature DataFrame.

    Returns:
        Tuple of (standardised DataFrame, mean Series, std Series).
        Mean and std are returned so they can be applied to new data.
    """
    mean = X.mean()
    std = X.std()
    X_std = (X - mean) / std
    return X_std, mean, std
```

### 2c. `to_tensors`

#### Why this section exists
PyTorch models operate on `torch.Tensor` objects, not pandas DataFrames. This function is the bridge between the two, and it exists as its own step because getting the dtype wrong here is a common, confusing first PyTorch error.

#### Line-by-line

- `def to_tensors(X: pd.DataFrame, y: pd.DataFrame):` — takes the standardized feature DataFrame and the target DataFrame, returns their tensor equivalents.
- The docstring calls out exactly why `dtype=torch.float32` is required: pandas defaults numeric columns to `float64`, but `nn.Linear` weights are `float32` by default, and multiplying a `float32` weight matrix against a `float64` input tensor raises `RuntimeError: expected scalar type Float but found Double`. Naming the exact error message in the docstring means a learner who hits it can search this file and find the explanation immediately.
- `X_tensor = torch.tensor(X.values, dtype=torch.float32)` — `X.values` pulls the underlying NumPy array out of the DataFrame (shape `(150, 5)`, dtype `float64`); `torch.tensor(...)` copies that data into a new PyTorch tensor, and `dtype=torch.float32` explicitly downcasts it to match what `nn.Linear` expects. Using `torch.tensor(...)` (which copies) rather than `torch.from_numpy(...)` (which shares memory) is the safer default here since the NumPy array is a short-lived intermediate anyway.
- `y_tensor = torch.tensor(y.values, dtype=torch.float32)  # shape (N, 1)` — same conversion for the target. The inline comment reiterates the shape `(N, 1)` because that 2-D shape (not a flat `(N,)` vector) is exactly what makes it compatible, element-for-element, with the model's `(N, 1)` output inside `nn.MSELoss`.
- `return X_tensor, y_tensor` — hands both tensors back to the caller, ready to be fed straight into `train(...)`.
- `# ───── 3. Training loop ─────` — the banner marking the boundary between the data-preparation functions above and the `train` function below. Same purely-cosmetic role as the other section banners.

#### Full code block

```python
def to_tensors(X: pd.DataFrame, y: pd.DataFrame):
    """Convert DataFrames to float32 PyTorch tensors.

    dtype=torch.float32 is required:
    - pandas defaults to float64
    - nn.Linear weights are float32 by default
    - Mismatched dtypes cause: RuntimeError: expected scalar type Float but found Double

    Args:
        X: Feature DataFrame (standardised).
        y: Target DataFrame, shape (N, 1).

    Returns:
        Tuple of (X_tensor, y_tensor).
    """
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)  # shape (N, 1)
    return X_tensor, y_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training loop
# ─────────────────────────────────────────────────────────────────────────────
```

---

## 3. Training loop — `train`

### Why this section exists
This is the heart of the script: the five-step PyTorch training loop learners are meant to internalize as a pattern they'll reuse in every future model. The docstring frames it explicitly as something to memorize, and the inline comments number each step so the code and the explanation stay in lockstep.

### Line-by-line

- `def train(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor, epochs: int = 100, lr: float = 0.01, print_every: int = 10) -> list:` — takes the model and full training tensors plus three hyperparameters, all with sensible defaults matching the values documented in the module docstring (100 epochs, `lr=0.01`, print every 10). Returning a typed `-> list` signals to callers that they get structured history back, not just side-effect printing.
- The docstring lays out "the four-step training loop" (numbered 1–5 in the actual list — forward, loss, zero_grad, backward, step) as something to memorize because it never changes across PyTorch projects: `predictions = model(X)` → `loss = criterion(predictions, y)` → `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`. It also explains **why** `zero_grad()` must come before `backward()`: PyTorch accumulates gradients into `.grad` by default on every `backward()` call, rather than overwriting them. Skipping `zero_grad()` means each epoch's gradient gets added on top of the last one, causing the optimizer to take increasingly wrong, oversized steps — a classic silent-divergence bug for PyTorch beginners.
- `criterion = nn.MSELoss()` — instantiates the loss function: Mean Squared Error, `mean((prediction - target)^2)`. MSE is the standard choice for regression because it's differentiable everywhere and penalizes large errors more than small ones (squaring), which pushes the model to avoid big misses.
- `optimizer = torch.optim.Adam(model.parameters(), lr=lr)` — instantiates the Adam optimizer, an adaptive-learning-rate variant of gradient descent. `model.parameters()` is a generator yielding every registered trainable tensor in the model — the `fc1` and `fc2` weights and biases. This is the exact line that would return nothing if `super().__init__()` had been skipped in `EnergyModel.__init__`. `lr=lr` passes through the function's `lr` parameter (default `0.01`), the step size Adam uses when updating each parameter.
- `history = []` — an empty list that will accumulate `(epoch, loss)` tuples, but only on epochs matching `print_every` — this becomes the return value used by `main()` to verify the loss actually decreased.
- `print("\n--- Training ---")` and `print(f"{'Epoch':>8}  {'Loss':>15}")` and `print("-" * 28)` — three lines that print a small header table before training starts: a section label, then column headers ("Epoch", "Loss") right-aligned to 8 and 15 characters respectively so they line up with the numeric rows printed later, then a 28-character rule to visually separate the header from the data.
- `for epoch in range(epochs + 1):` — loops from `0` through `epochs` inclusive (`epochs + 1` because `range` is exclusive of its stop value). Starting at epoch `0` means the very first printed row shows the loss **before** any weight update has happened — a useful baseline for the "initial loss vs. final loss" comparison done later in `main()`.
- `# Step 1 — forward pass` / `predictions = model(X_tensor)` — runs the full batch of 150 rows through the network in one call (this script uses full-batch gradient descent, not mini-batches — simple and fine at this data size). Calling `model(X_tensor)` invokes `EnergyModel.forward` under the hood via `nn.Module.__call__`.
- `# Step 2 — compute loss` / `loss = criterion(predictions, y_tensor)` — compares the `(150, 1)` predictions against the `(150, 1)` true targets and reduces them to a single scalar tensor representing the mean squared error for this epoch.
- `# Step 3 — zero gradients (must happen before backward)` / `optimizer.zero_grad()` — clears out the `.grad` attribute on every parameter tensor, resetting it to zero (or `None`, depending on PyTorch version/settings) before the new gradients are computed. The inline comment reiterates the ordering constraint spelled out in the docstring: this must run before `backward()`, not after.
- `# Step 4 — backward pass (compute gradients)` / `loss.backward()` — triggers PyTorch's autograd engine to walk the computation graph backward from `loss`, computing `∂loss/∂param` for every parameter and storing the result in each parameter's `.grad` attribute. No weights change yet — this step only computes gradients.
- `# Step 5 — update weights` / `optimizer.step()` — Adam reads each parameter's `.grad` and updates the parameter in place, using its adaptive per-parameter learning rate logic (maintaining running estimates of gradient mean and variance under the hood). This is the only line in the loop that actually changes the model's weights.
- `if epoch % print_every == 0:` — every 10th epoch (and epoch 0), execute the block below. This keeps console output readable — printing all 100 epochs would flood the screen with numbers nobody reads line-by-line.
- `loss_val = loss.item()` — `loss` is still a 0-dimensional PyTorch tensor at this point; `.item()` extracts its value as a plain Python float. This is necessary both for clean printing (a tensor would print as `tensor(9200.123, grad_fn=...)`, cluttered with autograd metadata) and because `history` should hold plain floats, not tensors that keep a reference to the whole computation graph alive.
- `print(f"{epoch:>8}  {loss_val:>15.4f}")` — prints one row of the table: the epoch number right-aligned to 8 characters, and the loss right-aligned to 15 characters with 4 decimal places, matching the header row's column widths from the initial `print` calls.
- `history.append((epoch, loss_val))` — records this epoch/loss pair as a 2-tuple. Only sampled epochs (every `print_every`) are stored, not all 101 — this keeps `history` small while still giving `main()` enough data points (epoch 0 vs. the final epoch) to confirm the loss trend.
- `print("-" * 28)` — closes the table with a matching bottom rule.
- `return history` — hands the list of `(epoch, loss)` tuples back to the caller.
- `# ───── 4. Save predictions ─────` — the banner marking the boundary between the training loop above and `save_predictions` below.

### Full code block

```python
def train(model: nn.Module,
          X_tensor: torch.Tensor,
          y_tensor: torch.Tensor,
          epochs: int = 100,
          lr: float = 0.01,
          print_every: int = 10) -> list:
    """Train the model using MSELoss and Adam.

    The four-step training loop — memorise this, it never changes:
        1. forward:    predictions = model(X)
        2. loss:       loss = criterion(predictions, y)
        3. zero_grad:  optimizer.zero_grad()  ← must come before backward
        4. backward:   loss.backward()
        5. step:       optimizer.step()

    Why zero_grad? PyTorch *accumulates* gradients by default.
    Without zeroing, each backward() adds to the previous gradient —
    causing divergence or incorrect updates.

    Args:
        model:       The neural network to train.
        X_tensor:    Feature tensor, shape (N, 5).
        y_tensor:    Target tensor, shape (N, 1).
        epochs:      Number of training epochs.
        lr:          Learning rate for Adam.
        print_every: Print loss every N epochs.

    Returns:
        List of (epoch, loss) tuples for every printed epoch.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    print("\n--- Training ---")
    print(f"{'Epoch':>8}  {'Loss':>15}")
    print("-" * 28)

    for epoch in range(epochs + 1):
        # Step 1 — forward pass
        predictions = model(X_tensor)

        # Step 2 — compute loss
        loss = criterion(predictions, y_tensor)

        # Step 3 — zero gradients (must happen before backward)
        optimizer.zero_grad()

        # Step 4 — backward pass (compute gradients)
        loss.backward()

        # Step 5 — update weights
        optimizer.step()

        if epoch % print_every == 0:
            loss_val = loss.item()
            print(f"{epoch:>8}  {loss_val:>15.4f}")
            history.append((epoch, loss_val))

    print("-" * 28)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Save predictions
# ─────────────────────────────────────────────────────────────────────────────
```

---

## 4. Save predictions — `save_predictions`

### Why this section exists
Training is only useful if the results are captured somewhere. This function runs the trained model in inference mode over the training data and writes a human-readable actual-vs-predicted CSV — the artifact learners inspect to judge whether the model actually learned anything sensible.

### Line-by-line

- `def save_predictions(model: nn.Module, X_tensor: torch.Tensor, y_original: pd.DataFrame, output_path: str = "demo_predictions.csv") -> None:` — takes the trained model, the (already standardized) feature tensor used for inference, the **original, non-standardized** target DataFrame for comparison, and an output path defaulting to `demo_predictions.csv` in the working directory.
- The docstring flags that `y_original` must be the non-standardized target — important because the predictions coming out of the model are in the same units as whatever `y_tensor` was trained on (raw kWh, since only `X` was standardized in this script, not `y`), and the CSV needs to show real, human-readable kWh values, not scaled ones.
- `model.eval()` — switches the model into evaluation mode. `EnergyModel` here has no `Dropout` or `BatchNorm` layers, so `.eval()` has no visible numerical effect on this specific architecture — but calling it unconditionally before inference is a correctness habit worth building, because forgetting it is a classic bug the moment a model *does* contain those layers (dropout would keep randomly zeroing activations at inference time, giving non-deterministic predictions).
- `with torch.no_grad():` — a context manager that disables autograd's gradient tracking for everything inside the block. Without it, every tensor operation during inference would still build up a computation graph in memory (needed only for `.backward()`, which we're not calling here) — wasting memory and compute for no benefit. This is the standard idiom for any inference-only code path in PyTorch.
- `preds = model(X_tensor).numpy().flatten()` — runs the forward pass to get predictions of shape `(150, 1)`, converts the tensor to a NumPy array with `.numpy()` (only legal because we're inside `no_grad()` and the tensor doesn't require gradients — calling `.numpy()` on a tensor that still requires grad raises an error), then `.flatten()` collapses `(150, 1)` down to a flat `(150,)` array so it lines up cleanly as a single DataFrame column.
- `df_out = pd.DataFrame({...})` — builds the output table with two columns.
  - `"actual_kwh": y_original["monthly_kwh"].values` — pulls the raw ground-truth values out of the original (pre-standardization) target DataFrame as a NumPy array, so the CSV shows real kWh figures a human can sanity-check against the input CSV.
  - `"predicted_kwh": preds.round(2)` — rounds the model's raw float predictions to 2 decimal places, which is plenty of precision for a kWh figure and keeps the CSV readable rather than showing 8 significant digits of float noise.
- `df_out.to_csv(output_path, index=False)` — writes the DataFrame to disk as CSV. `index=False` omits pandas' auto-generated row-number index column from the output — without it, the CSV would gain an unwanted leading unnamed column of `0, 1, 2, ...`.
- `print(f"\n[save_predictions] Saved {len(df_out)} rows → {output_path}")` — confirms completion and row count, following the same `[function_name]` logging convention used in `load_data`.
- `# ───── 5. Main ─────` — the final banner, marking the boundary between `save_predictions` above and the `main()` orchestrator / entrypoint guard below.

### Full code block

```python
def save_predictions(model: nn.Module,
                     X_tensor: torch.Tensor,
                     y_original: pd.DataFrame,
                     output_path: str = "demo_predictions.csv") -> None:
    """Run inference and save actual vs. predicted values to CSV.

    Args:
        model:       Trained model.
        X_tensor:    Feature tensor used for inference.
        y_original:  Original (non-standardised) target DataFrame.
        output_path: Where to write the CSV.
    """
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()

    df_out = pd.DataFrame({
        "actual_kwh": y_original["monthly_kwh"].values,
        "predicted_kwh": preds.round(2),
    })
    df_out.to_csv(output_path, index=False)
    print(f"\n[save_predictions] Saved {len(df_out)} rows → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────
```

---

## 5. Main — `main` and the entrypoint guard

### Why this section exists
`main()` wires every previous function together into one runnable pipeline, in the correct order, and adds a final sanity check that training actually worked. The `if __name__ == "__main__":` guard is what makes `python train.py` runnable directly while still allowing every function above to be imported and unit-tested individually (as `tests/test_train.py` does).

### Line-by-line

- `def main():` — no arguments; this function orchestrates the whole pipeline end-to-end using the module's documented defaults, matching how a learner would run the script from the command line.
- `print("=" * 50)` / `print(" Electricity Consumption Predictor — Demo")` / `print("=" * 50)` — prints a banner (50 equals signs, a title, 50 more equals signs) so the start of a training run is unmistakable in the console, especially useful when this script's output is compared side-by-side against a learner's own script during grading or demoing.
- `X_raw, y = load_data()` — calls `load_data` with its default path (`demo_data/electricity.csv`), unpacking the raw (non-standardized) feature DataFrame and target DataFrame.
- `X_std, mean, std = standardize(X_raw)` — standardizes the raw features, capturing the mean/std alongside the scaled DataFrame. The inline comment `# Standardise — critical for convergence` reiterates the point made at length in `standardize`'s own docstring.
- `print(f"\n[standardize] Feature means:\n{mean.round(2).to_string()}")` — prints the per-feature means (rounded to 2 decimals) computed during standardization, formatted with `.to_string()` so the full `Series` renders as a readable multi-line block rather than pandas' potentially truncated default `repr`. This gives a visible sanity check that the five features have plausible average values (e.g. `area_sqm` in the thousands, `has_ac` near 0.5).
- `X_tensor, y_tensor = to_tensors(X_std, y)` — converts the **standardized** features (not the raw ones) and the raw target DataFrame into `float32` tensors ready for the model. Note `y` here is the original, non-standardized target — only the features are standardized in this script.
- `print(f"\n[to_tensors] X shape: {X_tensor.shape}  y shape: {y_tensor.shape}")` — prints the resulting tensor shapes (`(150, 5)` and `(150, 1)`) as a final shape-check before training — a quick way to catch a broadcasting bug before it silently corrupts a 100-epoch run.
- `model = EnergyModel(input_size=5, hidden_size=32)` — instantiates the network with the dimensions matching the dataset (5 features) and the architecture documented in the module docstring (32 hidden units).
- `print(f"\n[model] Architecture:\n{model}")` — `nn.Module` defines a `__repr__` that pretty-prints every registered submodule and its shape (e.g. `EnergyModel((fc1): Linear(in_features=5, out_features=32, bias=True) ...)`). Printing `model` directly leverages that built-in representation instead of manually describing the architecture in a print statement — it can never drift out of sync with the actual layers.
- `history = train(model, X_tensor, y_tensor, epochs=100, lr=0.01, print_every=10)` — runs the full training loop with the defaults spelled out explicitly here (rather than relying on `train`'s own defaults) so the exact hyperparameters used for this run are visible at the call site, not hidden in a function signature elsewhere in the file.
- `initial_loss = history[0][1]` — `history[0]` is the `(epoch, loss)` tuple for epoch 0 (the pre-training baseline); `[1]` extracts the loss value from that tuple.
- `final_loss = history[-1][1]` — `history[-1]` is the last recorded tuple (epoch 100, the final loss after all training); `[1]` extracts its loss value.
- `print(f"\n[train] Initial loss: {initial_loss:.4f}  →  Final loss: {final_loss:.4f}")` — prints both numbers side by side with an arrow, giving an at-a-glance summary of how much the loss improved over the run.
- `assert final_loss < initial_loss, "Loss did not decrease — check standardization and learning rate."` — a hard correctness check: if training somehow made things worse (or didn't improve at all), the script crashes immediately with a specific, actionable message pointing at the two most likely causes (standardization or the learning rate) rather than silently producing a garbage `demo_predictions.csv`. This is the same "fail loudly rather than silently" philosophy used in `load_data`'s file-existence check.
- `save_predictions(model, X_tensor, y, output_path="demo_predictions.csv")` — runs inference with the trained model and writes the actual-vs-predicted CSV. Note it passes `y` (the original, non-standardized target DataFrame) as `y_original`, not `y_tensor` — `save_predictions` needs the human-readable values for the `actual_kwh` column.
- `print("\nDone. Check demo_predictions.csv.")` — a final completion message pointing the user at the output artifact.
- `if __name__ == "__main__":` — the standard Python entrypoint guard. `__name__` equals `"__main__"` only when the file is executed directly (`python train.py`), and equals the module name (`"train"`) when the file is imported elsewhere (e.g. by `tests/test_train.py` importing `EnergyModel` or `standardize`). This guard is what lets the test suite import individual functions from this file without triggering a full training run as a side effect of the import.
- `main()` — the single line inside the guard; calling the orchestration function kicks off the entire pipeline when the script is run directly.

### Full code block

```python
def main():
    print("=" * 50)
    print(" Electricity Consumption Predictor — Demo")
    print("=" * 50)

    # Load
    X_raw, y = load_data()

    # Standardise — critical for convergence
    X_std, mean, std = standardize(X_raw)
    print(f"\n[standardize] Feature means:\n{mean.round(2).to_string()}")

    # Convert to tensors
    X_tensor, y_tensor = to_tensors(X_std, y)
    print(f"\n[to_tensors] X shape: {X_tensor.shape}  y shape: {y_tensor.shape}")

    # Build model
    model = EnergyModel(input_size=5, hidden_size=32)
    print(f"\n[model] Architecture:\n{model}")

    # Train
    history = train(model, X_tensor, y_tensor, epochs=100, lr=0.01, print_every=10)

    # Verify loss decreased
    initial_loss = history[0][1]
    final_loss = history[-1][1]
    print(f"\n[train] Initial loss: {initial_loss:.4f}  →  Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease — check standardization and learning rate."

    # Save predictions
    save_predictions(model, X_tensor, y, output_path="demo_predictions.csv")
    print("\nDone. Check demo_predictions.csv.")


if __name__ == "__main__":
    main()
```
