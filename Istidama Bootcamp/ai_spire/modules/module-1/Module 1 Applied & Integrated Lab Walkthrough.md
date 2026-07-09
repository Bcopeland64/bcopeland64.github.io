# Python Fundamentals to Intermediate — Code Walkthrough

This document walks every cell of [python_fundamentals_to_intermediate.ipynb](python_fundamentals_to_intermediate.ipynb). Each section explains the code line-by-line — assuming no prior Python experience — and then shows the full code block from that cell (or group of cells) at the end so you can copy-paste it cleanly.

The notebook itself is bilingual: every explanatory markdown cell appears once in English and once in Arabic, back to back. This walkthrough only narrates the English content (the Arabic cells are a translation of the same material, not new material), but it covers **every single code cell**, in the order they appear in the notebook, including the 40 stub cells in the Group Coding Challenges section at the end.

---

## 0. Orientation — what the notebook covers

### Why this section exists
The opening cells (0–3) are pure markdown — a title, a table of contents, and a "how to use this notebook" guide, each duplicated in English and Arabic. There is no code to run yet, but it's worth knowing the roadmap: 17 topics running from variables through generators, followed by a "common mistakes" section, a "common errors" reference section, and finally a set of group coding challenges. This walkthrough follows that same order.

There is no code in this section — move straight to Section 1.

---

## 1. Variables & Data Types

### Why this section exists
Python is **dynamically typed**: a variable is just a name bound to an object, and the object — not the name — carries the type. This section establishes that mental model and introduces `type()` vs `isinstance()`, which you'll use constantly to debug "why is this the wrong type" errors later.

### Line-by-line

- `name = "Ada Lovelace"`, `age = 36`, `pi = 3.14159`, `is_engineer = True`, `nothing = None` — five assignments, one for each of Python's core built-in types (`str`, `int`, `float`, `bool`, `NoneType`). Python infers the type from the value on the right-hand side automatically; there is no `let`, `var`, or type declaration like in JavaScript, Java, or C.
- `print(type(name))` and the four prints that follow it — `type(x)` returns the exact class object of `x` (e.g. `<class 'str'>`). It is useful for quick inspection while debugging but is considered too strict for production type-checks because it doesn't account for inheritance.
- `print(isinstance(age, int))` — `isinstance()` is the preferred way to check a type because it returns `True` for the given type **or any subclass of it**. This matters immediately in Python because `bool` is technically a subclass of `int` (`True == 1`, `False == 0`), so `isinstance(True, int)` is `True`.
- `print(isinstance(pi, (int, float)))` — passing a **tuple** of types to `isinstance()` checks "is this any one of these types?" in a single call. This is the idiomatic way to check "is this a number" without writing `type(x) == int or type(x) == float`.

### Full code block

```python
# --- Variable Assignment ---
name = "Ada Lovelace"       # str
age = 36                    # int
pi = 3.14159                # float
is_engineer = True          # bool
nothing = None               # NoneType

# Inspect types
print(type(name))           # <class 'str'>
print(type(age))            # <class 'int'>
print(type(pi))              # <class 'float'>
print(type(is_engineer))    # <class 'bool'>
print(type(nothing))        # <class 'NoneType'>

# isinstance() is preferred over type() for type checking
print(isinstance(age, int))         # True
print(isinstance(pi, (int, float))) # True — checks against a tuple of types
```

### Line-by-line (Multiple Assignment)

- `x, y, z = 1, 2, 3` — **tuple unpacking assignment**: Python evaluates the right-hand side as a tuple `(1, 2, 3)` and unpacks it into three names in one statement. The number of names on the left must match the number of values on the right, or Python raises a `ValueError`.
- `print(x, y, z)` — `print()` accepts any number of positional arguments and joins them with a single space by default.
- `a = b = c = 0` — **chained assignment**: all three names are bound to the *same* `0` object. This is safe here because integers are immutable; chaining assignment with a mutable object (like a list) would make all three names point to the *same* mutable object, which is a common source of bugs (revisited in the "Common Mistakes" section).
- `x, y = y, x` — the classic Python **swap idiom**. The right-hand side `y, x` is packed into a temporary tuple *before* any assignment happens, so there's no need for a manual temp variable like `temp = x; x = y; y = temp` as you'd write in C or Java.
- `print(f"After swap: x={x}, y={y}")` — an f-string (formatted string literal): the `f` prefix before the opening quote tells Python to evaluate any `{expression}` inside the string and substitute its value. `{x}` here is simply the variable `x`.
- `value = 10` then `print(type(value))` — confirms `value` is currently an `int`.
- `value = "ten"` then `print(type(value))` — **re-binds** the same name `value` to a completely different object of a completely different type. This is legal in Python precisely because the name and the type live with the object, not the variable slot — this is what "dynamically typed" means in practice.

### Full code block

```python
# --- Multiple Assignment (Pythonic and clean) ---
x, y, z = 1, 2, 3
print(x, y, z)

# Assign the same value to multiple variables
a = b = c = 0
print(a, b, c)

# Swapping variables — no temp variable needed in Python!
x, y = y, x
print(f"After swap: x={x}, y={y}")

# Dynamic typing in action
value = 10
print(type(value))   # int
value = "ten"        # Now it's a string — Python allows this
print(type(value))   # str
```

---

## 2. Strings & String Operations

### Why this section exists
Strings are the data type you will touch most often — file paths, API payloads, prompts, log messages. This section covers how to create them, the three formatting styles (and why f-strings win), the most-used string methods, and slicing.

### Line-by-line (Creating Strings)

- `single = 'Hello'` and `double = "World"` — Python treats single and double quotes identically; there is no functional difference, only style preference (pick one and be consistent, or use the other kind when your string itself contains a quote character).
- `multiline = """This is\na multiline\nstring."""` — triple quotes (`"""..."""` or `'''...'''`) let a string literal span multiple physical lines in your source code; every line break you type becomes a literal `\n` inside the resulting string.
- `print(single, double)` and `print(multiline)` — printing multiple comma-separated arguments joins them with spaces; printing the multiline string reproduces the line breaks exactly as typed.

### Full code block

```python
# --- Creating Strings ---
single = 'Hello'
double = "World"
multiline = """This is
a multiline
string."""

print(single, double)
print(multiline)
```

### Line-by-line (String Formatting)

- `name = "Alan Turing"` and `age = 41` — two variables to format into output strings three different ways.
- `print(f"Name: {name}, Age: {age}")` — the f-string style (Python 3.6+, the modern default). Anything inside `{}` is a live Python expression, evaluated when the string is built.
- `print(f"In 10 years: {age + 10}")` — proves that f-strings aren't just variable substitution — `age + 10` is a full arithmetic expression evaluated inline.
- `print("Name: {}, Age: {}".format(name, age))` — the older `.format()` method: empty `{}` placeholders are filled positionally by the arguments passed to `.format()`, in order.
- `print("Name: %s, Age: %d" % (name, age))` — the legacy C-style `%`-formatting: `%s` means "substitute as a string", `%d` means "substitute as an integer". This style predates Python 3 and should be avoided in new code — it's shown here only so you recognize it in older codebases.

### Full code block

```python
# --- String Formatting ---
name = "Alan Turing"
age = 41

# f-strings (recommended — fast and readable)
print(f"Name: {name}, Age: {age}")
print(f"In 10 years: {age + 10}")   # Expressions work inside {}

# .format() method (older style)
print("Name: {}, Age: {}".format(name, age))

# %-formatting (legacy, avoid in new code)
print("Name: %s, Age: %d" % (name, age))
```

### Line-by-line (String Methods)

- `text = "  Python is Awesome!  "` — a string with leading/trailing whitespace, used to demonstrate cleanup methods.
- `text.strip()` — returns a **new** string with leading and trailing whitespace removed (strings are immutable, so `text` itself never changes — every method here returns a new string object).
- `text.lower()` / `text.upper()` — return new strings with every character converted to lower/upper case.
- `text.strip().replace("Awesome", "Powerful")` — method **chaining**: `.strip()` runs first and its returned string immediately has `.replace()` called on it. `.replace(old, new)` returns a copy with every occurrence of `old` swapped for `new`.
- `text.strip().split(" ")` — `.split(sep)` breaks a string into a **list** of substrings wherever `sep` occurs; called with no argument it also collapses runs of any whitespace, but here an explicit `" "` is passed.
- `sentence = "the quick brown fox"` — a fresh string for the next batch of methods.
- `sentence.title()` — capitalizes the first letter of every word ("The Quick Brown Fox").
- `sentence.capitalize()` — capitalizes only the very first character of the whole string, lowercasing the rest.
- `sentence.count("o")` — returns an `int`: how many non-overlapping times the substring `"o"` appears.
- `sentence.find("quick")` — returns the **index** of the first occurrence of the substring, or `-1` if it's not found (this is different from `.index()`, which raises a `ValueError` instead of returning `-1`).
- `sentence.startswith("the")` / `sentence.endswith("fox")` — return booleans; useful for filtering strings by prefix/suffix (e.g. checking a filename extension).
- `words = ["AI", "engineers", "use", "Python"]` then `" ".join(words)` — `str.join(iterable)` is called *on the separator string* and concatenates every element of the iterable, inserting the separator between each pair. This is the idiomatic (and fast) way to build a string from a list — the opposite operation of `.split()`.

### Full code block

```python
# --- String Methods ---
text = "  Python is Awesome!  "

print(text.strip())           # Remove leading/trailing whitespace
print(text.lower())           # Lowercase
print(text.upper())           # Uppercase
print(text.strip().replace("Awesome", "Powerful"))  # Replace substring
print(text.strip().split(" "))  # Split into list by space

sentence = "the quick brown fox"
print(sentence.title())       # Title Case
print(sentence.capitalize())  # First letter capitalized
print(sentence.count("o"))    # Count occurrences
print(sentence.find("quick")) # Index of first match (-1 if not found)
print(sentence.startswith("the"))  # Boolean check
print(sentence.endswith("fox"))    # Boolean check

# Join a list into a string
words = ["AI", "engineers", "use", "Python"]
print(" ".join(words))
```

### Line-by-line (Indexing & Slicing)

- `lang = "Python"` — the six-character string used throughout.
- `lang[0]` — indexing accesses a single character by its zero-based position (`0` is `'P'`).
- `lang[-1]` — negative indices count from the end; `-1` is always the last character (`'n'`), avoiding the need to compute `len(lang) - 1`.
- `lang[1:4]` — **slicing** with `[start:stop]` returns the substring from index `1` up to but **not including** index `4`, i.e. characters at positions 1, 2, 3 → `'yth'`.
- `lang[:3]` — omitting `start` defaults to `0` ("from the beginning").
- `lang[3:]` — omitting `stop` defaults to the end of the string ("to the end").
- `lang[::-1]` — the full `[start:stop:step]` form with a negative `step` of `-1` walks the string backwards, producing the reversed string. This is the standard Python idiom for reversing any sequence.
- `len(lang)` — returns the number of characters in the string (`6`).

### Full code block

```python
# --- String Indexing & Slicing ---
# Strings are sequences — you can index and slice them
lang = "Python"

print(lang[0])      # First character:  'P'
print(lang[-1])     # Last character:   'n'
print(lang[1:4])    # Slice [start:stop] → 'yth'
print(lang[:3])     # From start:       'Pyt'
print(lang[3:])     # To end:           'hon'
print(lang[::-1])   # Reversed:         'nohtyP'
print(len(lang))    # Length:           6
```

---

## 3. Numbers & Arithmetic

### Why this section exists
Numeric types look simple but hide two traps every beginner hits: `/` always returns a `float` in Python 3 (unlike Python 2 or many other languages), and floating-point numbers can't represent most decimals exactly. Getting comfortable with the operator table and the `math` module now avoids confusing bugs later.

### Line-by-line (Arithmetic Operators)

- `a, b = 17, 5` — tuple-unpacking assignment (seen in Section 1) sets up two operands.
- `a + b`, `a - b`, `a * b` — standard addition, subtraction, multiplication; behave as expected for `int` operands and return an `int`.
- `a / b` — **true division**. In Python 3 this *always* returns a `float`, even when both operands are `int` and divide evenly (`10 / 2` is `5.0`, not `5`). This is one of the most important behavior changes from Python 2.
- `a // b` — **floor division**. Divides and then rounds down (toward negative infinity), returning an `int` when both operands are `int`. Use this when you specifically want whole-number division.
- `a % b` — the **modulus** operator, returning the remainder after floor division. Extremely common for "is this number divisible by N" checks (`n % 2 == 0` for even) and for wrapping indices.
- `a ** b` — exponentiation. `17 ** 5` means 17 to the 5th power; this operator is generally faster than calling the `pow()` function for simple integer powers.
- Every line above is wrapped in an f-string like `f"Addition:       {a} + {b} = {a + b}"` — the expression `{a + b}` is evaluated live inside the string, so the printed value always reflects the actual computed result rather than a hardcoded number.

### Full code block

```python
# --- Arithmetic Operators ---
a, b = 17, 5

print(f"Addition:       {a} + {b} = {a + b}")
print(f"Subtraction:    {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division:       {a} / {b} = {a / b}")    # Always returns float
print(f"Floor Division: {a} // {b} = {a // b}")  # Returns int (truncates)
print(f"Modulus:        {a} % {b} = {a % b}")    # Remainder
print(f"Exponentiation: {a} ** {b} = {a ** b}")  # Power
```

### Line-by-line (Built-in Math Functions)

- `import math` — loads the standard library's `math` module. Because it's part of the standard library, no installation is needed — it ships with every Python interpreter.
- `abs(-42)` — a built-in (no import needed) that returns the absolute value, always non-negative.
- `round(3.14159, 2)` — rounds to the given number of decimal places (here, 2). Note: Python uses "banker's rounding" (round-half-to-even) for `.5` cases, which can surprise beginners coming from other languages.
- `max(3, 1, 4, 1, 5)` / `min(3, 1, 4, 1, 5)` — return the largest/smallest of any number of arguments (they also accept a single iterable, e.g. `max([3, 1, 4])`).
- `sum([1, 2, 3, 4, 5])` — adds every element of an iterable; requires a list/tuple/etc., not individual arguments like `max`/`min`.
- `pow(2, 10)` — the function form of `**`; equivalent to `2 ** 10`. `pow()` also supports a third argument for modular exponentiation (`pow(base, exp, mod)`), which `**` does not.
- `math.sqrt(144)` — square root, always returns a `float`. Raises `ValueError` for negative inputs (unlike `**0.5`, which returns a complex number for negatives in some contexts — `math.sqrt` is stricter).
- `math.floor(3.9)` / `math.ceil(3.1)` — round down / up to the nearest integer, returning an `int`.
- `math.pi` / `math.e` — module-level constants for π and Euler's number, accurate to float precision.
- `math.log(math.e)` — natural logarithm (base *e*); passing `math.e` itself demonstrates the identity `ln(e) == 1.0`.
- `math.log10(1000)` — logarithm base 10, a separate convenience function rather than passing a `base` argument to `math.log`.

### Full code block

```python
# --- Useful Built-in Math Functions ---
import math

print(abs(-42))             # Absolute value: 42
print(round(3.14159, 2))    # Round to 2 decimals: 3.14
print(max(3, 1, 4, 1, 5))  # Maximum: 5
print(min(3, 1, 4, 1, 5))  # Minimum: 1
print(sum([1, 2, 3, 4, 5])) # Sum: 15
print(pow(2, 10))           # 2^10 = 1024

print(math.sqrt(144))       # Square root: 12.0
print(math.floor(3.9))      # Floor: 3
print(math.ceil(3.1))       # Ceiling: 4
print(math.pi)              # π
print(math.e)               # Euler's number
print(math.log(math.e))     # Natural log of e = 1.0
print(math.log10(1000))     # Log base 10: 3.0
```

---

## 4. Collections: Lists, Tuples, Sets & Dictionaries

### Why this section exists
Choosing the *right* collection type is one of the highest-leverage decisions in everyday Python: it affects correctness (can this data have duplicates? does order matter?), readability, and performance (an `in` check on a `set` is O(1); on a `list` it's O(n)).

### Line-by-line (Lists)

- `frameworks = ["TensorFlow", "PyTorch", "JAX", "Keras", "PyTorch"]` — a list literal in square brackets. Lists are **ordered** (items keep the position you put them in) and allow **duplicates** (`"PyTorch"` appears twice on purpose).
- `frameworks[0]` / `frameworks[-1]` — indexing works exactly like string indexing: `0` is the first element, `-1` is the last.
- `frameworks[1:3]` — slicing a list returns a **new list** containing elements at indices 1 and 2 (stop index excluded, same rule as strings).
- `frameworks.append("scikit-learn")` — adds one element to the **end** of the list, mutating it in place. This is an O(1) amortized operation.
- `frameworks.insert(1, "ONNX")` — inserts an element at a specific index, shifting every later element one position to the right. This is O(n) because of the shift, unlike `.append()`.
- `frameworks.remove("Keras")` — removes the **first** matching value (not by index). Raises `ValueError` if the value isn't present.
- `popped = frameworks.pop()` — removes and **returns** the last element (or the element at an optional index argument). Unlike `.remove()`, `.pop()` gives you the removed value back.
- `frameworks.sort()` — sorts the list **in place** and returns `None`. A very common beginner bug is writing `frameworks = frameworks.sort()`, which sets `frameworks` to `None`.
- `numbers = [3, 1, 4, 1, 5, 9]` then `sorted(numbers, reverse=True)` — `sorted()` is a built-in **function** (not a method) that returns a brand-new sorted list and leaves the original list untouched; `reverse=True` sorts descending. `print(numbers)` afterward confirms the original list was never mutated.

### Full code block

```python
# --- LISTS ---
# Lists are ordered, mutable, and allow duplicates
frameworks = ["TensorFlow", "PyTorch", "JAX", "Keras", "PyTorch"]

print(frameworks[0])          # Access by index
print(frameworks[-1])         # Last element
print(frameworks[1:3])        # Slicing

frameworks.append("scikit-learn")   # Add to end
frameworks.insert(1, "ONNX")        # Insert at position
frameworks.remove("Keras")          # Remove first match
popped = frameworks.pop()           # Remove & return last item
print(f"Popped: {popped}")
print(f"Length: {len(frameworks)}")

frameworks.sort()                   # Sort in-place
print(frameworks)

# Sorted returns a new list (original unchanged)
numbers = [3, 1, 4, 1, 5, 9]
print(sorted(numbers, reverse=True))
print(numbers)  # Original intact
```

### Line-by-line (Tuples)

- `rgb_red = (255, 0, 0)` and `coordinates = (40.7128, -74.0060)` — tuple literals in parentheses. Tuples behave like lists for reading (indexing, slicing, `len()`) but are **immutable** — once created, elements cannot be added, removed, or reassigned.
- `rgb_red[0]` and `len(coordinates)` — indexing and `len()` work identically to lists because tuples are also sequences.
- `lat, lon = coordinates` — **tuple unpacking**: the two values inside the tuple are unpacked directly into two named variables in one line, which is far more readable than `lat = coordinates[0]; lon = coordinates[1]`.
- `single_tuple = (42,)` — the **trailing comma is mandatory** to make a single-element tuple; without it, Python treats the parentheses as ordinary grouping parentheses, not a tuple constructor.
- `not_a_tuple = (42)` — this is just the integer `42` wrapped in (redundant) parentheses — proven by `print(type(not_a_tuple))` showing `<class 'int'>`, not `<class 'tuple'>`.

### Full code block

```python
# --- TUPLES ---
# Tuples are immutable — use them for fixed collections (e.g., coordinates, RGB values)
rgb_red = (255, 0, 0)
coordinates = (40.7128, -74.0060)  # New York City lat/lon

print(rgb_red[0])        # Indexing works
print(len(coordinates))  # Length works

# Tuple unpacking
lat, lon = coordinates
print(f"Latitude: {lat}, Longitude: {lon}")

# Single-element tuple MUST have a trailing comma
single_tuple = (42,)   # This is a tuple
not_a_tuple  = (42)    # This is just an int!
print(type(single_tuple))  # <class 'tuple'>
print(type(not_a_tuple))   # <class 'int'>
```

### Line-by-line (Sets)

- `tags_a = {"python", "ml", "ai", "python"}` — a set literal in curly braces. Sets automatically drop duplicates on construction, so this set ends up with only 3 elements even though 4 strings were listed; sets are also **unordered**, so printing may show them in any order.
- `tags_a | tags_b` — the **union** operator: all elements present in either set.
- `tags_a & tags_b` — the **intersection** operator: only elements present in both sets.
- `tags_a - tags_b` — the **difference** operator: elements in `tags_a` that are *not* in `tags_b`.
- `tags_a ^ tags_b` — the **symmetric difference** operator: elements in exactly one of the two sets (present in either, but not both).
- `tags_a.add("nlp")` — adds a single element in place; adding an already-present element silently does nothing (no error, no duplicate).
- `tags_a.discard("ai")` — removes an element if present; unlike the similar `.remove()` method, `.discard()` does **not** raise an error if the element is missing, making it safer for "remove if it happens to exist" logic.
- `"python" in tags_a` — membership testing. Because sets are backed by a hash table, this check runs in O(1) time regardless of set size — dramatically faster than the equivalent `in` check on a list, which has to scan every element (O(n)).

### Full code block

```python
# --- SETS ---
# Sets are unordered collections of unique elements
tags_a = {"python", "ml", "ai", "python"}   # Duplicates auto-removed
tags_b = {"python", "data", "cloud"}

print(tags_a)                     # {'ml', 'ai', 'python'}
print(tags_a | tags_b)            # Union
print(tags_a & tags_b)            # Intersection
print(tags_a - tags_b)            # Difference (in a but not b)
print(tags_a ^ tags_b)            # Symmetric difference

tags_a.add("nlp")                 # Add element
tags_a.discard("ai")              # Remove (no error if missing)
print("python" in tags_a)         # Membership test — O(1) for sets!
```

### Line-by-line (Dictionaries)

- `model_info = {"name": "GPT-4", "parameters": "1.8T", "context_length": 128000, "multimodal": True}` — a dict literal: `{key: value, ...}` pairs in curly braces. Dictionaries preserve insertion order (guaranteed since Python 3.7) and each key must be unique and hashable (so strings, numbers, and tuples work as keys; lists do not).
- `model_info["name"]` — **direct access** by key using square brackets; raises `KeyError` if the key doesn't exist.
- `model_info.get("version", "unknown")` — **safe access**: `.get(key, default)` returns the default value instead of raising an error when the key is missing. This is the idiomatic way to read a dict when a key might not be present.
- `model_info["version"] = "4.0"` — assigns to a bracket-indexed key: if the key already exists its value is overwritten; if it doesn't exist, a new key-value pair is created. There is no separate "insert" method for a single key — assignment does both jobs.
- `del model_info["parameters"]` — the `del` statement removes a key (and its value) from the dictionary entirely.
- `for key, value in model_info.items():` — `.items()` returns a view of `(key, value)` tuple pairs; the `for` loop unpacks each pair directly into `key` and `value` on every iteration.
- `list(model_info.keys())` / `list(model_info.values())` — `.keys()` and `.values()` return dict *view* objects (not lists); wrapping them in `list()` materializes a concrete list you can index or print cleanly.
- `extra = {"creator": "OpenAI"}` — a second small dict to demonstrate merging.
- `merged = model_info | extra` — the **dict union operator** (`|`, Python 3.9+): returns a brand-new dict containing all keys from both, with `extra`'s values winning on any key collision. Neither original dict is modified.
- `merged_compat = {**model_info, **extra}` — the **dict-unpacking** merge pattern, which works on *every* Python 3 version (not just 3.9+): `**dict` inside a `{}` literal spreads all of that dict's key-value pairs into the new dict being built, with later spreads overriding earlier ones on key collisions.

### Full code block

```python
# --- DICTIONARIES ---
# Key-value stores — foundational to JSON, APIs, configs, and much more
model_info = {
    "name": "GPT-4",
    "parameters": "1.8T",
    "context_length": 128000,
    "multimodal": True
}

# Access
print(model_info["name"])                         # Direct access
print(model_info.get("version", "unknown"))        # Safe access with default

# Modify
model_info["version"] = "4.0"                     # Add or update
del model_info["parameters"]                      # Delete a key

# Iterate
for key, value in model_info.items():
    print(f"  {key}: {value}")

print(list(model_info.keys()))    # All keys
print(list(model_info.values()))  # All values

# Merge dictionaries
extra = {"creator": "OpenAI"}

# Python 3.9+ — dict union operator (clean and readable)
merged = model_info | extra
print("3.9+ merge:", merged)

# Python 3.8 and below — use dict unpacking (works on ALL versions)
merged_compat = {**model_info, **extra}
print("All versions:", merged_compat)
```

---

## 5. Control Flow: if / elif / else

### Why this section exists
Control flow is how a program branches based on data. This section covers the `if`/`elif`/`else` chain, Python's rule that indentation *is* the block syntax (not optional style), the ternary one-liner, and — critically — the difference between `==` (value equality) and `is` (identity), which trips up almost every beginner at least once.

### Line-by-line (Basic if/elif/else)

- `score = 82` — the input value to classify.
- `if score >= 90: grade = "A"` — the first branch checked; conditions are evaluated top to bottom.
- `elif score >= 80: grade = "B"` — only reached if the `if` above was `False`; since `82 >= 80` is `True`, this branch runs and every subsequent `elif`/`else` is skipped, even though `82 >= 70` and `82 >= 60` are also technically true.
- `elif score >= 70: ...` / `elif score >= 60: ...` — additional branches for lower grade bands, never reached in this run.
- `else: grade = "F"` — the catch-all branch that runs only if none of the `if`/`elif` conditions matched.
- `print(f"Score: {score} → Grade: {grade}")` — reports the result; `grade` is whichever branch actually executed.

### Full code block

```python
# --- Basic if/elif/else ---
score = 82

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Score: {score} → Grade: {grade}")
```

### Line-by-line (Ternary, Logical Operators, Identity, Truthiness)

- `status = "hot" if temperature > 30 else "comfortable"` — the **ternary expression**: `value_if_true if condition else value_if_false`, evaluated as a single expression and assigned directly. It's a compact substitute for a 4-line `if/else` block, appropriate only for simple cases.
- `x > 5 and x < 20` — `and` requires **both** sides to be `True`; it short-circuits, meaning if the left side is `False`, the right side is never even evaluated (relevant when the right side has side effects or could error).
- `x < 5 or x > 8` — `or` requires **at least one** side to be `True`; short-circuits the opposite way, skipping the right side once the left side is already `True`.
- `not x == 10` — `not` negates a boolean. Note the operator precedence: this is `not (x == 10)`, not `(not x) == 10`.
- `5 < x < 20` — a **chained comparison**, syntactic sugar equivalent to `5 < x and x < 20`, but evaluated more efficiently and read more naturally left-to-right — a feature many other languages lack.
- `a = [1, 2, 3]`, `b = [1, 2, 3]`, `c = a` — sets up three variables: `a` and `b` are two *separate* list objects that happen to contain equal values; `c` is bound to the *same* object as `a`.
- `a == b` — `True`, because `==` compares **value equality** (do these contain the same elements?).
- `a is b` — `False`, because `is` compares **identity** (are these literally the same object in memory?) — two separately-created lists are never the same object even with identical contents.
- `a is c` — `True`, because `c = a` bound `c` to the exact same object `a` already pointed to; no new list was created.
- `for val in [None, 0, "", [], {}, "hello", 42, [1]]: print(f"  bool({repr(val)}) = {bool(val)}")` — iterates over a mix of values and calls `bool()` on each to show Python's **truthiness** rules explicitly: `None`, `0`, empty string, and empty collections are all "falsy"; everything else (a non-empty string, a non-zero number, a non-empty list) is "truthy". `repr(val)` is used instead of `val` directly so that the empty string prints visibly as `''` rather than as nothing.

### Full code block

```python
# --- Ternary (One-line if/else) ---
temperature = 35
status = "hot" if temperature > 30 else "comfortable"
print(status)

# --- Comparison & Logical Operators ---
x = 10

print(x > 5 and x < 20)   # True — both conditions
print(x < 5 or x > 8)     # True — at least one
print(not x == 10)         # False — negation

# Chained comparisons (very Pythonic!)
print(5 < x < 20)         # True — equivalent to x > 5 and x < 20

# Identity vs Equality
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)   # True  — same VALUE
print(a is b)   # False — different OBJECTS in memory
print(a is c)   # True  — same object (c points to a)

# Truthiness — values Python considers True or False
# Falsy: None, 0, 0.0, "", [], {}, set()
for val in [None, 0, "", [], {}, "hello", 42, [1]]:
    print(f"  bool({repr(val)}) = {bool(val)}")
```

---

## 6. Loops: for and while

### Why this section exists
`for` loops in Python iterate over *content*, not index counters — this is a genuinely different mental model than C-style `for (int i=0; ...)` loops. This section builds up from basic iteration to `range()`, `enumerate()`, `zip()`, `while` loops, and the `break`/`continue`/`else` control-flow trio.

### Line-by-line (for loops)

- `models = ["BERT", "GPT", "T5", "LLaMA"]` then `for model in models:` — the idiomatic Python loop: `model` is bound to each element of the list in turn, with no manual index bookkeeping required.
- `print(f"  Model: {model}")` — runs once per iteration, printing the current element.
- `for i in range(5):` — `range(5)` lazily produces the integers `0, 1, 2, 3, 4` (5 values, starting at 0, stopping *before* 5). `print(i, end=" ")` uses `end=" "` to replace `print`'s default trailing newline with a space, so all five numbers print on one line.
- `for i in range(2, 10, 2):` — the three-argument form `range(start, stop, step)`: starts at 2, stops before 10, and increments by 2 each time, producing `2, 4, 6, 8`.
- `for idx, model in enumerate(models, start=1):` — `enumerate()` wraps any iterable and yields `(index, value)` pairs; `start=1` shifts the index to begin counting at 1 instead of the default 0 — convenient for human-readable numbered output ("1. BERT" rather than "0. BERT").

### Full code block

```python
# --- for loops ---

# Iterate over a list
models = ["BERT", "GPT", "T5", "LLaMA"]
for model in models:
    print(f"  Model: {model}")

print()

# range() — generates a sequence of numbers
for i in range(5):          # 0, 1, 2, 3, 4
    print(i, end=" ")
print()

for i in range(2, 10, 2):  # start=2, stop=10, step=2
    print(i, end=" ")
print()

# enumerate() — get index AND value simultaneously
for idx, model in enumerate(models, start=1):
    print(f"  {idx}. {model}")
```

### Line-by-line (zip, while, break/continue/else)

- `names = ["Alice", "Bob", "Carol"]` and `scores = [95, 87, 91]` then `for name, score in zip(names, scores):` — `zip()` pairs up elements from two (or more) iterables **positionally**: `names[0]` with `scores[0]`, `names[1]` with `scores[1]`, and so on. It stops as soon as the shortest input iterable is exhausted.
- `counter = 0` then `while counter < 5:` — a `while` loop repeats its body as long as the condition stays `True`. Unlike `for`, there's no automatic iteration variable — you're responsible for both checking and updating `counter`.
- `counter += 1` — the augmented-assignment shorthand for `counter = counter + 1`; without this line inside the loop body, the condition would never become `False` and the loop would run forever.
- `for num in range(10):` — iterates 0 through 9.
- `if num % 2 == 0: continue` — `continue` immediately jumps to the next iteration, skipping the rest of the current loop body. Here it skips even numbers.
- `if num > 7: break` — `break` exits the loop entirely (not just the current iteration), so no more numbers are checked once one exceeds 7.
- `print(num, end=" ")` — only reached for odd numbers ≤ 7, because of the `continue`/`break` guards above it, producing `1 3 5 7`.
- `target = 99` then `for num in [10, 20, 30]: if num == target: ... break` `else: print(...)` — demonstrates the **`for...else`** construct, a Python-specific feature: the `else` block runs only if the loop completes **without** hitting a `break`. Since `99` is never found in the list, the loop runs to completion and the `else` clause fires, printing "99 not found in the list". This pattern is useful for "search and report if not found" logic without a separate boolean flag variable.

### Full code block

```python
# --- zip() — iterate over multiple iterables in parallel ---
names = ["Alice", "Bob", "Carol"]
scores = [95, 87, 91]

for name, score in zip(names, scores):
    print(f"  {name}: {score}")

print()

# --- while loops ---
counter = 0
while counter < 5:
    print(f"  counter = {counter}")
    counter += 1

print()

# --- Loop control: break, continue, else ---
# break: exit the loop early
# continue: skip to the next iteration
# else: runs if the loop completed WITHOUT hitting a break

for num in range(10):
    if num % 2 == 0:
        continue          # Skip even numbers
    if num > 7:
        break             # Stop when num exceeds 7
    print(num, end=" ")  # 1 3 5 7
print()

# for...else pattern
target = 99
for num in [10, 20, 30]:
    if num == target:
        print(f"Found {target}")
        break
else:
    print(f"{target} not found in the list")
```

---

## 7. Functions

### Why this section exists
Functions are the primary unit of code reuse and, in Python, are **first-class objects** — they can be passed around, stored in variables, and returned from other functions. This section covers `def`, docstrings, default/keyword arguments, `*args`/`**kwargs`, and type hints — everything needed to write clean, reusable functions.

### Line-by-line (Basic Function Definition)

- `def greet(name):` — defines a function named `greet` taking one required positional parameter `name`.
- `"""Return a greeting string. This is a docstring — always write them!"""` — a **docstring**: a string literal placed immediately after the `def` line, which Python automatically stores as the function's `__doc__` attribute. It's not a comment — it's introspectable at runtime.
- `return f"Hello, {name}!"` — `return` ends the function and sends the given value back to the caller; without an explicit `return`, a Python function returns `None` by default.
- `print(greet("Turing"))` — calls the function with a positional argument and prints the returned string.
- `print(greet.__doc__)` — accesses the docstring directly as an attribute on the function object itself, demonstrating that a Python function is an object with its own attributes, not just a block of code.

### Full code block

```python
# --- Basic Function Definition ---
def greet(name):
    """Return a greeting string. This is a docstring — always write them!"""
    return f"Hello, {name}!"

print(greet("Turing"))
print(greet.__doc__)  # Access the docstring
```

### Line-by-line (Default & Keyword Arguments)

- `def train_model(model_name, epochs=10, learning_rate=0.001, verbose=True):` — `model_name` is a required positional parameter; `epochs`, `learning_rate`, and `verbose` all have **default values**, making them optional at the call site. Default values are evaluated **once**, when the `def` statement runs (a fact that becomes important — and dangerous with mutable defaults — in the Common Mistakes section).
- `if verbose: print(...)` — uses the `verbose` flag to conditionally print, demonstrating a common pattern for toggling logging/diagnostic output.
- `return {"model": model_name, "epochs": epochs, "lr": learning_rate}` — returns a dict summarizing what was "trained", built from a dict literal.
- `train_model("ResNet50")` — a purely **positional** call; every optional parameter falls back to its default.
- `train_model("BERT", learning_rate=0.0001, epochs=3)` — a call mixing one positional argument with two **keyword** arguments. Keyword arguments can be supplied in any order because they're matched by name, not position.
- `train_model("GPT-2", epochs=50, verbose=False)` — overrides two of the three defaults while leaving `learning_rate` at its default `0.001`.

### Full code block

```python
# --- Default Arguments & Keyword Arguments ---
def train_model(model_name, epochs=10, learning_rate=0.001, verbose=True):
    """Simulate a training run with configurable hyperparameters."""
    if verbose:
        print(f"Training {model_name} for {epochs} epochs at lr={learning_rate}")
    return {"model": model_name, "epochs": epochs, "lr": learning_rate}

# Positional call
train_model("ResNet50")

# Keyword arguments (order doesn't matter)
train_model("BERT", learning_rate=0.0001, epochs=3)

# Override defaults
train_model("GPT-2", epochs=50, verbose=False)
```

### Line-by-line (*args and **kwargs)

- `def sum_all(*args):` — `*args` collects **any number** of extra positional arguments into a **tuple** named `args` inside the function. The name `args` is convention, not syntax — the `*` is what matters.
- `return sum(args)` — `sum()` (a built-in) adds up every element of the `args` tuple.
- `sum_all(1, 2, 3)` and `sum_all(10, 20, 30, 40)` — demonstrate that the function accepts a variable number of arguments without needing to be rewritten or overloaded.
- `def log_config(**kwargs):` — `**kwargs` collects any number of extra **keyword** arguments into a **dict** named `kwargs`, mapping each argument name (as a string) to its value.
- `for key, value in kwargs.items(): print(...)` — iterates the collected keyword arguments the same way you'd iterate any dict.
- `log_config(batch_size=32, optimizer="Adam", dropout=0.3)` — the caller passes arbitrary named settings; the function had no idea in advance what keys would be provided, which is exactly the point of `**kwargs` — building flexible configuration-style APIs.

### Full code block

```python
# --- *args and **kwargs ---
# *args  → variable number of positional arguments (comes in as a tuple)
# **kwargs → variable number of keyword arguments  (comes in as a dict)

def sum_all(*args):
    """Accept any number of numbers and return their sum."""
    return sum(args)

print(sum_all(1, 2, 3))          # 6
print(sum_all(10, 20, 30, 40))   # 100

def log_config(**kwargs):
    """Print all keyword arguments as configuration."""
    print("Configuration:")
    for key, value in kwargs.items():
        print(f"  {key} = {value}")

log_config(batch_size=32, optimizer="Adam", dropout=0.3)
```

### Line-by-line (Type Hints & Functions as Objects)

- `def calculate_accuracy(correct: int, total: int) -> float:` — **type hints**: `correct: int` and `total: int` document the expected parameter types, and `-> float` documents the return type. None of this is enforced at runtime by plain Python — it's purely documentation and tooling metadata — but it enables IDE autocompletion and static checkers like `mypy` to catch mismatches before the code ever runs.
- The docstring uses **Google-style** formatting with `Args:` and `Returns:` sections — a convention for documenting parameters and return values in a way that documentation generators can parse automatically.
- `if total == 0: raise ValueError("total must be greater than 0")` — **input validation**: rather than letting a `ZeroDivisionError` happen implicitly on the next line, the function proactively raises a more descriptive, intentional error.
- `return correct / total` — true division (Section 3), always returning a `float` here as promised by the `-> float` annotation.
- `accuracy = calculate_accuracy(90, 100)` then `print(f"Accuracy: {accuracy:.2%}")` — the `:.2%` format spec inside the f-string multiplies the float by 100, rounds to 2 decimal places, and appends a `%` sign — turning `0.9` into `"90.00%"` in one step.
- `my_func = calculate_accuracy` — assigns the function itself (not its result — note there are no parentheses) to a new name. Because functions are first-class objects in Python, `my_func` now refers to the exact same function object as `calculate_accuracy`.
- `my_func(45, 50)` — calling through the new name works identically, proving the function object — not just its original name — was what got assigned.

### Full code block

```python
# --- Type Hints (Python 3.5+, strongly recommended for AI code) ---
def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate classification accuracy.

    Args:
        correct: Number of correct predictions.
        total: Total number of predictions.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    if total == 0:
        raise ValueError("total must be greater than 0")
    return correct / total

accuracy = calculate_accuracy(90, 100)
print(f"Accuracy: {accuracy:.2%}")  # 90.00%

# Functions as first-class objects
my_func = calculate_accuracy           # Assign function to variable
print(my_func(45, 50))                 # Call via new name
```

---

## 8. Lambda Functions

### Why this section exists
A `lambda` is a single-expression anonymous function, most useful as a throwaway `key=` argument to `sorted()`/`min()`/`max()` or as the callback passed to `map()`/`filter()`. This section shows the syntax and its two classic companions, `map()` and `filter()`.

### Line-by-line

- `square = lambda x: x ** 2` — the syntax is `lambda parameters: expression`. There is no `return` keyword — the expression's value is returned automatically. Calling `square(5)` evaluates to `25`.
- `add = lambda x, y: x + y` — a lambda can take multiple parameters, comma-separated, just like a regular function.
- `models = [...]` — a list of dicts, each with a `"name"` and `"params_M"` key, used as data for the sorting demo.
- `sorted_models = sorted(models, key=lambda m: m["params_M"])` — `sorted()`'s `key=` parameter takes a function that's called once per element to produce the value actually used for comparison; here, each dict `m` is reduced to its `"params_M"` number for sorting purposes, while the full dict is what ends up in the output list. This is the single most common real-world use of `lambda`.
- `for m in sorted_models: print(f"  {m['name']}: {m['params_M']}M parameters")` — iterates the sorted list, printing each model's name and size.
- `temperatures_c = [0, 20, 37, 100]` then `list(map(lambda c: c * 9/5 + 32, temperatures_c))` — `map(func, iterable)` applies `func` to every element and returns a lazy iterator; wrapping it in `list()` forces it to produce a concrete list of Celsius-to-Fahrenheit conversions immediately.
- `numbers = [1, 2, ..., 10]` then `list(filter(lambda n: n % 2 == 0, numbers))` — `filter(func, iterable)` keeps only the elements for which `func` returns a truthy value; here it keeps only even numbers. Like `map()`, `filter()` returns a lazy iterator that must be wrapped in `list()` to see the results directly.

### Full code block

```python
# --- Lambda Syntax: lambda arguments: expression ---
square = lambda x: x ** 2
print(square(5))         # 25

add = lambda x, y: x + y
print(add(3, 4))         # 7

# --- Practical usage with sorted() ---
models = [
    {"name": "BERT",   "params_M": 340},
    {"name": "GPT-2",  "params_M": 117},
    {"name": "T5-base","params_M": 220},
]

# Sort by number of parameters
sorted_models = sorted(models, key=lambda m: m["params_M"])
for m in sorted_models:
    print(f"  {m['name']}: {m['params_M']}M parameters")

# --- map() — apply function to every element ---
temperatures_c = [0, 20, 37, 100]
temperatures_f = list(map(lambda c: c * 9/5 + 32, temperatures_c))
print(temperatures_f)

# --- filter() — keep elements where function returns True ---
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda n: n % 2 == 0, numbers))
print(evens)
```

---

## 9. List Comprehensions (& Dict/Set Comprehensions)

### Why this section exists
Comprehensions are a concise, often-faster alternative to writing a `for` loop plus `.append()` calls, and they're used constantly in AI/ML code for reshaping and filtering data. This section builds from a basic list comprehension up through dict and set comprehensions and nested (flattening) comprehensions.

### Line-by-line (Basic List Comprehension)

- `squares_loop = []` then `for n in range(10): squares_loop.append(n ** 2)` — the traditional, explicit way to build a list: start empty, loop, append each computed value.
- `squares = [n ** 2 for n in range(10)]` — the equivalent **list comprehension**: `[expression for item in iterable]`. It produces the identical list to the four-line loop above, in one line, and is typically 20–30% faster because the loop body compiles to a single optimized operation instead of repeated `.append()` method lookups.
- `even_squares = [n ** 2 for n in range(10) if n % 2 == 0]` — adds a **filter clause**: `[expression for item in iterable if condition]`. The `if` is evaluated for every item; only items where it's `True` contribute to the resulting list.
- `words = [" hello ", " WORLD ", "  Python  "]` then `cleaned = [w.strip().lower() for w in words]` — shows that the "expression" part of a comprehension can be any Python expression, including chained method calls (`.strip().lower()`), not just arithmetic.

### Full code block

```python
# --- Basic List Comprehension ---
# Syntax: [expression for item in iterable if condition]

# Traditional loop
squares_loop = []
for n in range(10):
    squares_loop.append(n ** 2)

# List comprehension — same result, one line
squares = [n ** 2 for n in range(10)]
print(squares)

# With condition (filter)
even_squares = [n ** 2 for n in range(10) if n % 2 == 0]
print(even_squares)

# String processing
words = [" hello ", " WORLD ", "  Python  "]
cleaned = [w.strip().lower() for w in words]
print(cleaned)
```

### Line-by-line (Dict, Set, and Nested Comprehensions)

- `words = ["apple", "banana", "cherry", "date"]` then `word_lengths = {word: len(word) for word in words}` — a **dict comprehension**: `{key_expr: value_expr for item in iterable}`. Here each word becomes a key mapped to its length.
- `original = {"a": 1, "b": 2, "c": 3}` then `inverted = {v: k for k, v in original.items()}` — iterating `.items()` inside a dict comprehension and swapping `k`/`v` in the output expression **inverts** the dictionary (keys become values and vice versa). This only works safely if the original values are unique and hashable — duplicate values would silently overwrite each other.
- `data = [1, 2, 2, 3, 3, 3, 4]` then `unique_squares = {x ** 2 for x in data}` — a **set comprehension**: same syntax as a list comprehension but with `{}` instead of `[]` and no colon. Because the result is a set, duplicate squared values (there are none here, but duplicate inputs are automatically deduplicated) collapse into one.
- `matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]` then `flat = [elem for row in matrix for elem in row]` — a **nested comprehension** used for flattening: read the `for` clauses left to right exactly as you would nested `for` loops (`for row in matrix:` outer, `for elem in row:` inner), with `elem` as the final expression collected into the flat output list.

### Full code block

```python
# --- Dict Comprehension ---
# Syntax: {key_expr: value_expr for item in iterable}

words = ["apple", "banana", "cherry", "date"]
word_lengths = {word: len(word) for word in words}
print(word_lengths)

# Invert a dictionary (swap keys and values)
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)

# --- Set Comprehension ---
data = [1, 2, 2, 3, 3, 3, 4]
unique_squares = {x ** 2 for x in data}
print(unique_squares)

# --- Nested List Comprehension (matrix flattening) ---
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [elem for row in matrix for elem in row]
print(flat)
```

---

## 10. File I/O

### Why this section exists
Real AI pipelines constantly read training data, write checkpoints, and load configs from disk. This section establishes the `with` statement as the non-negotiable way to work with files (it guarantees the file gets closed even if an error happens), covers the difference between reading strategies, and then applies the same pattern to JSON specifically.

### Line-by-line (Writing, Reading, Appending)

- `import os` — loads the standard library module used later for `os.path.abspath()`.
- `filepath = "sample_data.txt"` — a relative path; the file will be created in the notebook's current working directory.
- `lines = ["name,score,grade\n", "Alice,95,A\n", "Bob,82,B\n", "Carol,91,A\n"]` — a list of strings, each one representing a full CSV line, each ending with an explicit `\n` newline character (which `.writelines()` does **not** add automatically).
- `with open(filepath, "w") as f:` — opens the file in `"w"` (write) mode, which **creates** the file if it doesn't exist and **truncates** (erases) it if it does. The `with` statement is a **context manager**: no matter how the block exits — normally or via an exception — Python guarantees `f.close()` is called, flushing any buffered data to disk. This is why you should always prefer `with open(...)` over a bare `open()`/`close()` pair.
- `f.writelines(lines)` — writes every string in the `lines` list to the file back-to-back, with no separator added between them (hence the manual `\n` in each string above).
- `os.path.abspath(filepath)` — converts the relative path into a full absolute path, useful for confirming exactly where the file landed on disk.
- `with open(filepath, "r") as f: content = f.read()` — `"r"` (read, the default mode) opens an existing file; `.read()` slurps the **entire file** into one string. This is convenient for small files but risky for large ones since the whole thing must fit in memory at once.
- `with open(filepath, "r") as f: for line in f:` — iterating directly over an open file object yields one line at a time (each including its trailing `\n`), without ever holding the whole file in memory — the preferred approach for large files. `line.strip()` removes that trailing newline (and any other surrounding whitespace) before printing.
- `with open(filepath, "a") as f: f.write("David,78,C\n")` — `"a"` (append) mode opens the file for writing **without** truncating existing content; new writes are added after everything already there. `.write()` (singular) writes exactly the one string passed to it, unlike `.writelines()`.
- The final `with open(filepath, "r") as f: print(f.read())` block re-reads the whole file to visually confirm the appended line is now present.

### Full code block

```python
import os

# --- Writing to a file ---
filepath = "sample_data.txt"

lines = [
    "name,score,grade\n",
    "Alice,95,A\n",
    "Bob,82,B\n",
    "Carol,91,A\n",
]

with open(filepath, "w") as f:
    f.writelines(lines)

print(f"File written: {os.path.abspath(filepath)}")

# --- Reading the entire file ---
with open(filepath, "r") as f:
    content = f.read()
print("Full content:\n", content)

# --- Reading line by line (memory-efficient for large files) ---
with open(filepath, "r") as f:
    for line in f:
        print(line.strip())

# --- Appending to a file ---
with open(filepath, "a") as f:
    f.write("David,78,C\n")

# Verify append worked
with open(filepath, "r") as f:
    print("After append:")
    print(f.read())
```

### Line-by-line (Working with JSON)

- `import json` — the standard library's JSON encoder/decoder.
- `config = {"model": "transformer", "layers": 12, "hidden_size": 768, "dropout": 0.1, "use_gpu": True}` — an ordinary Python dict representing a model configuration, using types (`str`, `int`, `float`, `bool`) that all map cleanly to JSON.
- `with open("config.json", "w") as f: json.dump(config, f, indent=4)` — `json.dump(obj, file)` serializes `obj` directly to an already-open file handle; `indent=4` pretty-prints the output with 4-space indentation instead of writing it as one dense line.
- `with open("config.json", "r") as f: loaded_config = json.load(f)` — `json.load(file)` is the inverse: reads and parses JSON directly from an open file handle back into native Python objects (JSON objects become dicts, arrays become lists, etc.).
- `print(loaded_config)` and `print(f"Hidden size: {loaded_config['hidden_size']}")` — confirms the round trip preserved the data, including nested numeric types.
- `json_str = json.dumps(config)` — `json.dumps()` (with an "s", for "string") converts a Python object directly to a JSON **string** in memory, without touching a file — useful when building an HTTP request body, for example.
- `back_to_dict = json.loads(json_str)` — `json.loads()` is the string-based counterpart to `json.load()`, parsing a JSON string back into a Python dict.
- `print(type(back_to_dict))` — confirms the round trip through string form produced a `dict`, not left it as a string.

### Full code block

```python
import json

# --- Working with JSON ---
# JSON is everywhere in APIs, configs, and ML metadata

config = {
    "model": "transformer",
    "layers": 12,
    "hidden_size": 768,
    "dropout": 0.1,
    "use_gpu": True
}

# Write JSON to file
with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

# Read JSON from file
with open("config.json", "r") as f:
    loaded_config = json.load(f)

print(loaded_config)
print(f"Hidden size: {loaded_config['hidden_size']}")

# JSON string conversions (useful for API responses)
json_str = json.dumps(config)         # dict → JSON string
back_to_dict = json.loads(json_str)   # JSON string → dict
print(type(back_to_dict))
```

---

## 11. Exception Handling

### Why this section exists
Errors will happen; the mark of professional code is expecting and handling them gracefully instead of crashing. This section covers the full `try`/`except`/`else`/`finally` structure and then shows how to define your own exception classes so error conditions in your code are explicit and searchable.

### Line-by-line (Basic try/except)

- `def safe_divide(a, b):` — wraps a division in a function so the error-handling logic is reusable and testable.
- `try: result = a / b` — the `try` block contains the code that might raise an exception; Python starts executing it and jumps to a matching `except` the instant an exception is raised.
- `except ZeroDivisionError: print(...); return None` — catches specifically a division-by-zero error (raised when `b` is `0`), prints a friendly message, and returns `None` instead of letting the program crash.
- `except TypeError as e: print(f"Error: Invalid types — {e}"); return None` — catches the case where `a` or `b` isn't a number at all (e.g. a string); `as e` binds the exception object to the name `e` so its message can be included in the printed output.
- `else: print(...); return result` — the `else` block runs **only if no exception was raised** in the `try` block. Putting the success path here (rather than at the end of `try`) makes it explicit that this code assumes the division actually succeeded.
- `finally: print("  (division attempted)")` — the `finally` block runs **unconditionally** — whether the division succeeded, failed with `ZeroDivisionError`, failed with `TypeError`, or even if an *uncaught* exception type occurred. This is the correct place for cleanup code (closing files, releasing locks) that must always execute.
- `safe_divide(10, 2)`, `safe_divide(10, 0)`, `safe_divide("ten", 2)` — three calls exercising the success path, the `ZeroDivisionError` path, and the `TypeError` path respectively, so you can see all four blocks (`try`, both `except`s, `else`, `finally`) fire under different conditions.

### Full code block

```python
# --- Basic try/except ---
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError as e:
        print(f"Error: Invalid types — {e}")
        return None
    else:
        # Runs ONLY if no exception was raised
        print(f"{a} / {b} = {result}")
        return result
    finally:
        # ALWAYS runs — use for cleanup (closing connections, etc.)
        print("  (division attempted)")

safe_divide(10, 2)
safe_divide(10, 0)
safe_divide("ten", 2)
```

### Line-by-line (Custom Exceptions)

- `class ModelNotTrainedError(Exception): """..."""; pass` — defines a brand-new exception type by inheriting from the built-in `Exception` class. `pass` means the class body adds no extra behavior beyond what `Exception` already provides — its whole purpose is to exist as a distinctly-named, catchable error type.
- `class InvalidHyperparameterError(ValueError):` — inherits from `ValueError` specifically (rather than the generic `Exception`), which means code that catches `ValueError` broadly will also catch this more specific exception, thanks to Python's exception inheritance rules.
- `def __init__(self, param_name, value, valid_range): self.param_name = param_name; super().__init__(...)` — a custom constructor that stores `param_name` as an attribute on the exception instance (so calling code can inspect *which* parameter was invalid programmatically) and calls `super().__init__(...)` to set the human-readable error message that will be shown if the exception is printed or left uncaught.
- `def set_learning_rate(lr: float): if not (0.0 < lr < 1.0): raise InvalidHyperparameterError(...)` — validates the input using a chained comparison (Section 5) and `raise`s the custom exception with details about what went wrong when the value is out of the valid `(0.0, 1.0)` range.
- `try: set_learning_rate(0.001); set_learning_rate(5.0) except InvalidHyperparameterError as e: print(f"Caught: {e}")` — the first call succeeds silently (prints a confirmation), the second call raises the custom exception, which is caught and its message printed. Note execution never reaches the print statement that would have followed a successful second call, because the exception jumps straight to `except`.

### Full code block

```python
# --- Custom Exceptions ---
# Define your own exceptions to make error handling expressive and specific

class ModelNotTrainedError(Exception):
    """Raised when attempting to use a model that hasn't been trained."""
    pass

class InvalidHyperparameterError(ValueError):
    """Raised when a hyperparameter value is out of valid range."""
    def __init__(self, param_name, value, valid_range):
        self.param_name = param_name
        super().__init__(f"Invalid value {value} for '{param_name}'. Expected: {valid_range}")

def set_learning_rate(lr: float):
    if not (0.0 < lr < 1.0):
        raise InvalidHyperparameterError("learning_rate", lr, "(0.0, 1.0)")
    print(f"Learning rate set to {lr}")

try:
    set_learning_rate(0.001)   # Valid
    set_learning_rate(5.0)     # Invalid — will raise
except InvalidHyperparameterError as e:
    print(f"Caught: {e}")
```

---

## 12. Object-Oriented Programming (OOP)

### Why this section exists
OOP bundles data and the behavior that operates on it into a single unit. This section builds a `NeuralNetwork` class from scratch covering constructors, instance vs. class variables, `@property`, `@classmethod`, `@staticmethod`, and dunder methods (`__repr__`/`__str__`), then extends it with `Inheritance` via a `ConvolutionalNetwork` subclass.

### Line-by-line (Defining a Class)

- `class NeuralNetwork:` — defines a new class (a blueprint for objects) named `NeuralNetwork`.
- `framework = "PyTorch"` — a **class variable**, defined directly inside the class body but outside any method. It's shared by *every* instance of the class unless a specific instance overrides it.
- `def __init__(self, name: str, layers: int, learning_rate: float = 0.001):` — the **constructor**, automatically called every time you create a new instance (`NeuralNetwork(...)`). `self` is the instance being constructed — every instance method takes `self` as its first parameter by convention, giving the method access to that specific object's data. `learning_rate` has a default value, making it optional at construction time.
- `self.name = name`, `self.layers = layers`, `self.learning_rate = learning_rate` — **instance variables**: each is stored directly on `self`, meaning every `NeuralNetwork` object gets its own independent copy (unlike the shared `framework` class variable).
- `self._is_trained = False` and `self._epoch = 0` — additional instance state. The leading underscore is a **naming convention** (not enforced by the language) signaling "this is internal — please don't access it directly from outside the class."
- `def train(self, epochs: int):` — an **instance method**: it operates on `self`, mutating `self._epoch` and `self._is_trained` and printing a status message.
- `def predict(self, input_data): if not self._is_trained: raise RuntimeError(...)` — **guards** against calling `predict()` before `train()`, raising a built-in `RuntimeError` with a descriptive message if the precondition isn't met.
- `@property def is_trained(self): return self._is_trained` — the `@property` decorator turns a method into something accessed **like an attribute** (`nn.is_trained`, no parentheses) rather than called like a method (`nn.is_trained()`). It exposes `_is_trained` for reading without allowing direct external mutation, since no corresponding setter was defined.
- `@classmethod def create_simple(cls, name: str): return cls(name, layers=3, learning_rate=0.01)` — `@classmethod` methods receive the **class itself** (`cls`) as the first argument instead of an instance. This is commonly used to build **alternative constructors** — here, a shortcut for creating a `NeuralNetwork` with sensible defaults for a "simple" model.
- `@staticmethod def supported_optimizers(): return [...]` — `@staticmethod` methods receive **neither** `self` nor `cls` — they're really just plain functions that happen to live inside the class's namespace for organizational purposes, useful for utility logic conceptually related to the class but not dependent on any instance or class state.
- `def __repr__(self): return f"NeuralNetwork(name={self.name!r}, ...)"` — the "official", developer-facing string representation, shown by the debugger/REPL and by `repr()`. The `!r` conversion flag calls `repr()` on `self.name` specifically, so a string value is shown with its quotes (`'TransformerNet'`) for unambiguous debugging.
- `def __str__(self): ...` — the "friendly", human-facing representation used automatically by `print()` and `str()`; falls back to `__repr__` if not defined.
- `nn = NeuralNetwork("TransformerNet", layers=12, learning_rate=0.0001)` — creates an instance, implicitly calling `__init__`.
- `print(repr(nn))` / `print(str(nn))` — explicitly invoke each dunder method to show the difference between the two representations.
- `nn.train(10)` then `print(nn.predict("image_data.jpg"))` — trains the instance (satisfying the guard in `predict`) then successfully calls `predict`.
- `simple_nn = NeuralNetwork.create_simple("QuickNet")` — calls the classmethod on the class itself (not an instance), which internally calls `cls(...)` to build and return a new, fully-formed instance.
- `NeuralNetwork.supported_optimizers()` — calls the static method directly on the class, with no instance required at all.

### Full code block

```python
# --- Defining a Class ---
class NeuralNetwork:
    """A simplified representation of a neural network."""

    # Class variable — shared across all instances
    framework = "PyTorch"

    def __init__(self, name: str, layers: int, learning_rate: float = 0.001):
        """Constructor — called when an instance is created."""
        self.name = name              # Instance variable
        self.layers = layers
        self.learning_rate = learning_rate
        self._is_trained = False      # Convention: _ prefix = "private"
        self._epoch = 0

    def train(self, epochs: int):
        """Simulate training the model."""
        self._epoch += epochs
        self._is_trained = True
        print(f"[{self.name}] Trained for {epochs} epochs (total: {self._epoch})")

    def predict(self, input_data):
        """Make a prediction."""
        if not self._is_trained:
            raise RuntimeError(f"Model '{self.name}' must be trained before predicting!")
        return f"Prediction from {self.name} based on: {input_data}"

    @property
    def is_trained(self):
        """Property: access _is_trained as a read-only attribute."""
        return self._is_trained

    @classmethod
    def create_simple(cls, name: str):
        """Class method: alternative constructor."""
        return cls(name, layers=3, learning_rate=0.01)

    @staticmethod
    def supported_optimizers():
        """Static method: doesn't access class or instance."""
        return ["Adam", "SGD", "RMSProp", "AdaGrad"]

    def __repr__(self):
        """Official string representation — used in debugging."""
        return f"NeuralNetwork(name={self.name!r}, layers={self.layers}, lr={self.learning_rate})"

    def __str__(self):
        """User-friendly string representation."""
        status = "trained" if self._is_trained else "untrained"
        return f"{self.name} ({self.layers} layers, {status})"


# --- Using the class ---
nn = NeuralNetwork("TransformerNet", layers=12, learning_rate=0.0001)
print(repr(nn))
print(str(nn))
print(f"Is trained: {nn.is_trained}")

nn.train(10)
print(nn.predict("image_data.jpg"))

# Class method
simple_nn = NeuralNetwork.create_simple("QuickNet")
print(repr(simple_nn))

# Static method
print(NeuralNetwork.supported_optimizers())
```

### Line-by-line (Inheritance)

- `class ConvolutionalNetwork(NeuralNetwork):` — declares `ConvolutionalNetwork` as a **subclass** of `NeuralNetwork`, written by putting the parent class name in parentheses after the child's class name. The subclass automatically gets every attribute and method `NeuralNetwork` defines, unless it overrides them.
- `def __init__(self, name: str, layers: int, kernel_size: int = 3): super().__init__(name, layers)` — `super().__init__(...)` explicitly calls the **parent class's constructor** to set up `self.name`, `self.layers`, etc., exactly as `NeuralNetwork.__init__` would. Without this call, none of that inherited setup would happen.
- `self.kernel_size = kernel_size` — adds a new instance variable specific to CNNs, on top of everything inherited from `NeuralNetwork`.
- `def predict(self, input_data):` — **overrides** the parent's `predict` method with CNN-specific behavior (a different error message and a return string that mentions the kernel size). When you call `cnn.predict(...)`, Python uses this version, not the parent's.
- `def __repr__(self):` — also overridden, so `repr(cnn)` shows `ConvolutionalNetwork(...)` with its own fields rather than the parent's `NeuralNetwork(...)` format.
- `cnn = ConvolutionalNetwork("VGG16", layers=16, kernel_size=3)` — constructs an instance of the subclass.
- `cnn.train(20)` — calls `.train()`, which `ConvolutionalNetwork` never redefined, so Python looks it up on the parent class `NeuralNetwork` and runs that version unchanged — this is **inheritance** in action: code reuse without copy-pasting.
- `print(cnn.predict("cat.jpg"))` — calls the **overridden** version defined directly on `ConvolutionalNetwork`.
- `isinstance(cnn, NeuralNetwork)` — `True`, because a `ConvolutionalNetwork` **is a** `NeuralNetwork` (it inherits from it) — this is the polymorphism payoff: code written to accept a `NeuralNetwork` will happily also accept a `ConvolutionalNetwork`.
- `isinstance(cnn, ConvolutionalNetwork)` — also `True`, trivially, since `cnn` is a direct instance of that exact class.

### Full code block

```python
# --- Inheritance ---
class ConvolutionalNetwork(NeuralNetwork):
    """A CNN that extends NeuralNetwork."""

    def __init__(self, name: str, layers: int, kernel_size: int = 3):
        super().__init__(name, layers)   # Call parent __init__
        self.kernel_size = kernel_size

    def predict(self, input_data):       # Override parent method
        if not self._is_trained:
            raise RuntimeError("CNN must be trained first!")
        return f"CNN prediction (kernel={self.kernel_size}x{self.kernel_size}) on: {input_data}"

    def __repr__(self):
        return f"ConvolutionalNetwork(name={self.name!r}, layers={self.layers}, kernel={self.kernel_size})"


cnn = ConvolutionalNetwork("VGG16", layers=16, kernel_size=3)
cnn.train(20)          # Inherited from NeuralNetwork
print(cnn.predict("cat.jpg"))
print(isinstance(cnn, NeuralNetwork))           # True — it IS a NeuralNetwork
print(isinstance(cnn, ConvolutionalNetwork))    # True
```

---

## 13. Modules & Imports

### Why this section exists
Python's standard library and PyPI ecosystem are enormous; knowing the different `import` styles, when to alias, and a handful of standard-library modules you'll reach for constantly (`Counter`, `defaultdict`, `Path`, `datetime`) saves a lot of reinventing the wheel.

### Line-by-line

- `import math` then `math.sqrt(256)` — imports the whole module; every name inside it must be accessed with the `math.` prefix.
- `from math import pi, factorial` then `print(pi)` / `print(factorial(5))` — imports specific names directly into the current namespace, so no prefix is needed when using them. This is convenient but means the reader has to remember (or look up) that `pi` and `factorial` came from `math`.
- `import os`, `import sys` — two more whole-module imports from the standard library, used later.
- `import datetime as dt` — imports the module under an **alias**. Aliasing is standard practice for long or frequently-typed module names (the AI/ML ecosystem convention `import numpy as np` follows the same pattern).
- `from collections import defaultdict, Counter, OrderedDict` — imports three specific classes from the `collections` module directly by name.
- `from typing import Optional, Any` — imports type-hint helpers; the comment in the cell notes that in Python 3.9+ you can use built-in generics (`list`, `dict`) directly and in 3.10+ you can write `X | None` instead of `Optional[X]`, but `Optional` is still imported here for clarity/back-compatibility.
- `from pathlib import Path` — imports the modern, object-oriented path-handling class (preferred over the older `os.path` string-based functions).
- `now = dt.datetime.now()` — gets the current local date and time as a `datetime` object (accessed through the `dt` alias set up above).
- `now.strftime('%Y-%m-%d %H:%M:%S')` — formats the datetime object into a human-readable string using `strftime` format codes (`%Y` = 4-digit year, `%m` = month, `%d` = day, `%H:%M:%S` = 24-hour time).
- `p = Path(".") / "config.json"` — `pathlib.Path` overloads the `/` operator to join path segments in a readable, cross-platform way — no need to worry about `/` vs `\` between operating systems.
- `p.exists()` — a method on the `Path` object that checks whether the file actually exists on disk, returning a boolean.
- `tokens = ["the", "cat", "sat", "on", "the", "mat", "the"]` then `counts = Counter(tokens)` — `Counter` is a dict subclass specialized for counting; constructing it directly from an iterable automatically tallies how many times each element appears.
- `counts.most_common(2)` — returns the top-2 most frequent items as a list of `(item, count)` tuples, sorted by descending count — far simpler than manually sorting a plain dict by value.
- `word_positions = defaultdict(list)` — `defaultdict(list)` is a dict subclass where accessing a **missing key** automatically creates it with a fresh empty `list()` (the "factory" passed to the constructor) instead of raising `KeyError`.
- `for i, word in enumerate(tokens): word_positions[word].append(i)` — because of the `defaultdict` behavior, `word_positions[word]` never needs an explicit "does this key exist yet?" check before appending — the first access for any new word silently creates its empty list first.
- `dict(word_positions)` — converts the `defaultdict` to a plain `dict` purely for cleaner printing (a `defaultdict`'s `repr()` shows its factory function, which is noisier output).

### Full code block

```python
# --- Import styles ---

# 1. Import the whole module
import math
print(math.sqrt(256))   # Must prefix with module name

# 2. Import specific names from a module
from math import pi, factorial
print(pi)
print(factorial(5))     # No prefix needed

# 3. Import with alias (standard conventions for AI/ML libraries)
import os
import sys
import datetime as dt
from collections import defaultdict, Counter, OrderedDict
# Python 3.9+: use built-in types directly: list, dict, tuple
# Python 3.10+: use X | Y instead of Union[X, Y]
# Optional still imported for clarity; Optional[X] == X | None
from typing import Optional, Any  # Only import what built-ins cannot replace
from pathlib import Path

# datetime module
now = dt.datetime.now()
print(f"Now: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# Path is the modern, recommended way to handle file paths
p = Path(".") / "config.json"
print(f"Path: {p}, Exists: {p.exists()}")

# collections.Counter — instantly count occurrences
tokens = ["the", "cat", "sat", "on", "the", "mat", "the"]
counts = Counter(tokens)
print(counts)
print(counts.most_common(2))  # Top 2 most frequent

# collections.defaultdict — dict that provides default value for missing keys
word_positions = defaultdict(list)
for i, word in enumerate(tokens):
    word_positions[word].append(i)
print(dict(word_positions))
```

---

## 14. Decorators

### Why this section exists
Decorators wrap a function with extra behavior without modifying its source — the mechanism behind `@property`, Flask/FastAPI routes, and `@functools.lru_cache`. This section builds a `@timer` decorator from scratch, then a parameterized `@retry(max_attempts=..., delay=...)` decorator factory.

### Line-by-line (@timer)

- `import time` and `import functools` — `time` provides `time.perf_counter()` for high-resolution timing; `functools` provides the `@functools.wraps` helper used inside the decorator.
- `def timer(func):` — the decorator itself is just a regular function that accepts the function-to-be-wrapped, `func`, as its only argument.
- `@functools.wraps(func) def wrapper(*args, **kwargs):` — `wrapper` is the replacement function that will actually run in place of the original. `@functools.wraps(func)` copies `func`'s `__name__`, `__doc__`, and other metadata onto `wrapper`, so tools like `help()` still show information about the *original* function rather than the generic-looking `wrapper`. `*args, **kwargs` lets `wrapper` accept absolutely any combination of arguments and forward them on, so `timer` can decorate *any* function regardless of its signature.
- `start = time.perf_counter()` ... `result = func(*args, **kwargs)` ... `end = time.perf_counter()` — records a timestamp, calls the **original** function with whatever arguments were passed to the wrapper, then records a second timestamp. `func(*args, **kwargs)` unpacks the collected arguments back out to call `func` exactly as the caller intended.
- `print(f"[timer] {func.__name__!r} took {end - start:.6f}s")` — reports the elapsed time; `func.__name__` still correctly shows the original function's name (e.g. `'compute_sum'`) because it's read directly off `func`, not `wrapper`.
- `return result` then `return wrapper` — `wrapper` returns whatever the original function returned, preserving normal call semantics; `timer` itself returns the newly built `wrapper` function object, which is what actually replaces the original name.
- `@timer def compute_sum(n: int) -> int: return sum(range(n))` — the `@timer` syntax immediately above a `def` is exactly equivalent to writing `compute_sum = timer(compute_sum)` right after the original definition — from now on, calling `compute_sum(...)` actually calls `wrapper(...)`, which calls the real logic internally and adds timing around it.
- `@timer def slow_function(): time.sleep(0.1); return "done"` — a second decorated function, deliberately slow, to show the timer producing a visibly larger duration.
- `print(compute_sum(1_000_000))` and `print(slow_function())` — both calls transparently go through the timing wrapper; the caller doesn't need to know or care that decoration happened.

### Full code block

```python
import time
import functools

# --- Building a Decorator ---
def timer(func):
    """Decorator that measures and prints the execution time of a function."""
    @functools.wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[timer] {func.__name__!r} took {end - start:.6f}s")
        return result
    return wrapper


@timer
def compute_sum(n: int) -> int:
    """Sum all numbers from 0 to n."""
    return sum(range(n))

@timer
def slow_function():
    """A deliberately slow function."""
    time.sleep(0.1)
    return "done"

print(compute_sum(1_000_000))
print(slow_function())
```

### Line-by-line (Parameterized @retry)

- `def retry(max_attempts=3, delay=0.1):` — this is a **decorator factory**, not a decorator itself — it's a function that takes *configuration* (`max_attempts`, `delay`) and returns a decorator built around that configuration. This extra layer is required because `@retry(max_attempts=4, delay=0.05)` needs to evaluate `retry(max_attempts=4, delay=0.05)` first (producing a decorator), and only then apply that decorator to the function underneath.
- `def decorator(func):` — the actual decorator, nested one level inside `retry`. It closes over `max_attempts` and `delay` from the enclosing `retry(...)` call — this is a **closure**: `decorator` (and `wrapper` inside it) can still see and use `max_attempts`/`delay` even after `retry()` itself has returned.
- `@functools.wraps(func) def wrapper(*args, **kwargs):` — same metadata-preserving pattern as the `timer` decorator.
- `for attempt in range(1, max_attempts + 1): try: return func(*args, **kwargs) except Exception as e: ...` — attempts to call the wrapped function; if it raises *any* exception, the `except` block prints which attempt failed and, if attempts remain, sleeps `delay` seconds before looping again to retry. If the call succeeds, `return` exits the function (and the loop) immediately with the real result.
- `if attempt < max_attempts: time.sleep(delay)` — only pauses between retries, not after the very last attempt (no point waiting if there's nothing left to retry).
- `raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts")` — reached only if every attempt inside the loop raised an exception and the loop finished naturally; converts the repeated failures into a single, clear `RuntimeError`.
- `import random` then `@retry(max_attempts=4, delay=0.05) def flaky_api_call():` — applies the configured decorator; the call `retry(max_attempts=4, delay=0.05)` runs first and returns `decorator`, which is then applied to `flaky_api_call`.
- `if random.random() < 0.7: raise ConnectionError("API timeout")` — simulates a network call that fails 70% of the time by comparing a random float in `[0.0, 1.0)` against `0.7`.
- `return "API response: {\"status\": \"ok\"}"` — the success path, only reached on the 30% chance the random check doesn't trigger.
- `try: result = flaky_api_call(); print("Success:", result) except RuntimeError as e: print("Final failure:", e)` — catches the `RuntimeError` that `wrapper` raises only if all 4 attempts failed; because of the retry logic, most runs will eventually succeed within 4 tries, but occasionally all 4 will fail and this `except` fires.

### Full code block

```python
# --- Decorator with arguments ---
def retry(max_attempts=3, delay=0.1):
    """Decorator factory: retries a function on failure up to max_attempts times."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"  Attempt {attempt} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator


import random

@retry(max_attempts=4, delay=0.05)
def flaky_api_call():
    """Simulates an unreliable API call."""
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("API timeout")
    return "API response: {\"status\": \"ok\"}"

try:
    result = flaky_api_call()
    print("Success:", result)
except RuntimeError as e:
    print("Final failure:", e)
```

---

## 15. Generators & Iterators

### Why this section exists
Generators produce values on demand instead of building an entire list in memory upfront — essential when a dataset is too large to load all at once (a batch of 1M images, for example). This section compares generator memory footprint to a list, builds a generator function, and implements a custom iterator class from the underlying protocol.

### Line-by-line (Generator vs List: memory comparison)

- `import sys` — used for `sys.getsizeof()`, which reports the memory footprint (in bytes) of a Python object.
- `n = 1_000_000` — the underscore in `1_000_000` is just a readability separator for large integer literals; it has no effect on the value.
- `list_version = [x ** 2 for x in range(n)]` — a list comprehension that **immediately** computes and stores all 1,000,000 squared values in memory as one big list object.
- `gen_version = (x ** 2 for x in range(n))` — swapping the square brackets `[]` for parentheses `()` turns the exact same expression into a **generator expression** instead — it does not compute anything yet; it just creates a lightweight object that *knows how* to produce the values lazily, one at a time, when asked.
- `sys.getsizeof(list_version)` vs `sys.getsizeof(gen_version)` — printed with the `,` format spec for thousands separators. The list's reported size scales with all 1,000,000 stored values; the generator's size stays tiny and constant no matter how many values it will eventually produce, because it hasn't produced any of them yet.

### Full code block

```python
import sys

# --- Generator vs List: memory comparison ---
n = 1_000_000

list_version = [x ** 2 for x in range(n)]      # Builds entire list in memory
gen_version  = (x ** 2 for x in range(n))       # Generator expression — lazy

print(f"List size:      {sys.getsizeof(list_version):,} bytes")
print(f"Generator size: {sys.getsizeof(gen_version):,} bytes")
```

### Line-by-line (Generator Function & Batch Generator)

- `def fibonacci():` — any function containing a `yield` statement anywhere in its body automatically becomes a **generator function**; calling it does not run any of the code inside — it immediately returns a generator object.
- `a, b = 0, 1` then `while True: yield a; a, b = b, a + b` — an intentionally infinite loop. `yield a` **pauses** execution at that exact point, hands `a`'s current value back to whoever called `next()`, and **freezes** all local state (`a`, `b`, and the loop position) until `next()` is called again, at which point execution resumes right after the `yield` — here, at the tuple-unpacking swap `a, b = b, a + b` that advances the sequence.
- `fib = fibonacci()` — creates the generator object; still nothing has executed inside `fibonacci` yet.
- `first_10 = [next(fib) for _ in range(10)]` — a list comprehension that calls `next(fib)` ten times, pulling ten successive Fibonacci numbers out of the generator one at a time; `_` is the conventional throwaway variable name used when the loop variable itself (here, the index from `range(10)`) is never actually used.
- `def batch_generator(data: list, batch_size: int):` — a practical generator pattern directly relevant to ML training loops.
- `for i in range(0, len(data), batch_size): yield data[i : i + batch_size]` — walks through `data` in steps of `batch_size`, yielding a slice (a sub-list) of up to `batch_size` elements each time, rather than materializing all batches in memory at once.
- `dataset = list(range(20))` then `for batch_num, batch in enumerate(batch_generator(dataset, batch_size=6), start=1):` — `enumerate(..., start=1)` numbers the batches starting from 1 as the generator lazily produces each one; with 20 items and a batch size of 6, this produces 4 batches (three of size 6, one final partial batch of size 2).

### Full code block

```python
# --- Generator Function ---
def fibonacci():
    """Infinite Fibonacci sequence generator."""
    a, b = 0, 1
    while True:
        yield a          # Pause here, return a, resume on next()
        a, b = b, a + b


fib = fibonacci()
first_10 = [next(fib) for _ in range(10)]
print("First 10 Fibonacci numbers:", first_10)


# --- Practical: Batch Data Generator for ML ---
def batch_generator(data: list, batch_size: int):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


dataset = list(range(20))   # Simulated dataset of 20 samples

for batch_num, batch in enumerate(batch_generator(dataset, batch_size=6), start=1):
    print(f"Batch {batch_num}: {batch}")
```

### Line-by-line (Custom Iterator class)

- `class Countdown:` — demonstrates the iterator protocol manually, the same underlying mechanism that makes `for` loops work on any custom object.
- `def __init__(self, start): self.current = start` — stores the starting value as instance state.
- `def __iter__(self): return self` — implementing `__iter__` is what makes an object an **iterable** (something a `for` loop can start iterating over). Returning `self` means the object is *also* its own iterator — a common (though not universal) pattern for simple custom iterators.
- `def __next__(self): if self.current < 0: raise StopIteration` — implementing `__next__` is what makes an object an **iterator**. Each call either returns the next value or, once exhausted, raises the built-in `StopIteration` exception — this is the exact signal a `for` loop watches for internally to know when to stop looping (it catches `StopIteration` silently; you never see it as a crash in normal `for`-loop usage).
- `value = self.current; self.current -= 1; return value` — captures the current value before decrementing, so the countdown includes the starting number itself, then returns the captured value.
- `for n in Countdown(5): print(n, end=" ")` — the `for` loop transparently calls `iter(Countdown(5))` (which returns the object itself, per `__iter__`) and then repeatedly calls `next()` on it (which is `__next__`) until `StopIteration` is raised, printing `5 4 3 2 1 0` before stopping.
- `print("🚀")` — prints a rocket emoji after the loop completes, purely as a playful "liftoff" flourish once the countdown reaches zero.

### Full code block

```python
# --- Custom Iterator class ---
class Countdown:
    """An iterator that counts down from n to 0."""

    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self    # The iterator object itself

    def __next__(self):
        if self.current < 0:
            raise StopIteration   # Signal that iteration is complete
        value = self.current
        self.current -= 1
        return value

for n in Countdown(5):
    print(n, end=" ")
print("🚀")
```

---

## 16. Common Python Mistakes & How to Fix Them

### Why this section exists
These seven mistakes are the ones seen most often across every experience level, and most of them **don't crash** — they silently produce wrong results, which is far more dangerous than an obvious error message. Each mistake below is shown broken, then fixed, in the same cell.

### 16.1 — Mutable Default Argument

#### Why this section exists
Default argument values are evaluated **once**, at function-definition time — not fresh on every call. A mutable default (list, dict, set) is therefore silently shared and accumulates state across unrelated calls, which is one of Python's most infamous gotchas.

#### Line-by-line

- `def add_item_wrong(item, items=[]):` — the empty list `[]` is created exactly once, when this `def` statement runs, and that *same* list object is reused as the default for every future call that doesn't supply its own `items`.
- `items.append(item); return items` — mutates whichever list object `items` currently refers to. On the first call with no `items` argument, that's the shared default list.
- `add_item_wrong("apple")` returns `['apple']`, but `add_item_wrong("banana")` returns `['apple', 'banana']` — the second call's "default" list is the *same object* left over from the first call, because it was never recreated.
- `def add_item_correct(item, items=None): if items is None: items = []` — the fix: default to the immutable sentinel `None` (safe to reuse, since it can't be mutated) and only create a **fresh** empty list *inside the function body* — meaning a brand-new list is created on every call that doesn't pass its own.
- `add_item_correct("apple")` and `add_item_correct("banana")` now each correctly return a single-item list, because each call gets its own fresh `items` list.

#### Full code block

```python
# ❌ WRONG — the list persists between calls!
def add_item_wrong(item, items=[]):
    items.append(item)
    return items

print(add_item_wrong("apple"))   # ['apple']
print(add_item_wrong("banana"))  # ['apple', 'banana']  ← Bug! Expected ['banana']

print("---")

# ✅ FIX — use None as default, create the list inside the function
def add_item_correct(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item_correct("apple"))   # ['apple']
print(add_item_correct("banana"))  # ['banana']  ← Correct!
```

### 16.2 — Modifying a List While Iterating Over It

#### Why this section exists
A `for` loop tracks its position in a list by internal index. Removing elements from the list *while* iterating shifts every later element down one position, causing the loop to silently skip items — with no error raised at all.

#### Line-by-line

- `numbers = [1, 2, 3, 4, 5, 6]` then `for num in numbers: if num % 2 == 0: numbers.remove(num)` — as soon as `2` is removed, everything after it shifts left by one: `3` moves into the position `2` used to occupy, but the loop's internal index has already moved past that position, so `3` gets silently skipped over on the next iteration, and the pattern compounds from there.
- `print("Wrong result:", numbers)` — shows `[1, 3, 5]`, which happens to look plausible, masking the fact that the *mechanism* that produced it was buggy (some evens survive, or some odds get accidentally caught, depending on the exact list — this is why the bug is dangerous: it can look right by luck).
- `for num in numbers[:]:` — **Fix 1**: `numbers[:]` (a full slice) creates a **shallow copy** of the list before iteration begins. The loop iterates safely over the frozen copy while `.remove()` still mutates the real `numbers` list — no interference between "what's being iterated" and "what's being changed".
- `numbers = [n for n in numbers if n % 2 != 0]` — **Fix 2**, the cleanest: a list comprehension builds an entirely **new** list containing only the elements that pass the filter, with no in-place removal — and therefore no iteration-while-mutating hazard at all. This is the generally preferred fix.

#### Full code block

```python
# ❌ WRONG — removing elements during iteration causes skipped items
numbers = [1, 2, 3, 4, 5, 6]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)
print("Wrong result:", numbers)   # [1, 3, 5] — but 2 was removed, then 4 was skipped!

# ✅ FIX 1 — iterate over a copy
numbers = [1, 2, 3, 4, 5, 6]
for num in numbers[:]:           # numbers[:] creates a shallow copy
    if num % 2 == 0:
        numbers.remove(num)
print("Fix 1:", numbers)         # [1, 3, 5]

# ✅ FIX 2 — list comprehension (cleanest and most Pythonic)
numbers = [1, 2, 3, 4, 5, 6]
numbers = [n for n in numbers if n % 2 != 0]
print("Fix 2:", numbers)         # [1, 3, 5]
```

### 16.3 — Confusing `is` with `==`

#### Why this section exists
`is` checks object identity, `==` checks value equality. Small integers and short strings are sometimes cached and reused internally by CPython, which makes `is` *appear* to work by coincidence for tiny examples — right up until it silently fails on larger, real values.

#### Line-by-line

- `x = 1000` and `y = 1000` then `print(x is y)` — may print `False`, because 1000 falls outside CPython's small-integer cache range (roughly -5 to 256), so `x` and `y` are usually two distinct integer objects that happen to hold the same value.
- `name = "hello"` then `print(name is "hello")` — string literal caching ("interning") behavior is implementation-specific and not guaranteed, so this comparison can give inconsistent results across Python versions/implementations — which is exactly why `is` should never be used for value comparison.
- `print(x == y)` and `print(name == "hello")` — both reliably `True`, because `==` compares the actual values, not object identity.
- `result = None` then `print(result is None)` and `print(result is not None)` — this is the **one legitimate, idiomatic use of `is`**: checking against the singleton `None` object. Because there is only ever one `None` object in a running Python process, `is None` is both correct and slightly faster than `== None`.

#### Full code block

```python
# ❌ WRONG — using 'is' to compare values
x = 1000
y = 1000
print(x is y)   # May be False! (different objects for large ints)

name = "hello"
print(name is "hello")   # May give unexpected results

# ✅ FIX — use == for value comparison, 'is' only for None/True/False
print(x == y)             # True — correct value comparison
print(name == "hello")    # True

# 'is' should ONLY be used for singleton checks
result = None
print(result is None)     # ✅ Correct use of 'is'
print(result is not None) # ✅ Correct
```

### 16.4 — Not Using `enumerate()` When You Need the Index

#### Why this section exists
This is a style mistake rather than a correctness bug, but manual index tracking is more error-prone (off-by-one bugs, forgetting to increment) and less readable than the built-in tool designed for exactly this purpose.

#### Line-by-line

- `items = ["alpha", "beta", "gamma"]` — the sample data.
- `i = 0; for item in items: print(f"{i}: {item}"); i += 1` — **Wrong pattern 1**: manually maintaining a counter variable alongside the loop — extra state to manage and an extra line that's easy to forget or place incorrectly.
- `for i in range(len(items)): print(f"{i}: {items[i]}")` — **Wrong pattern 2**: looping over indices via `range(len(...))` and then re-indexing back into the list on every iteration. It works, but it's considered a Python "code smell" — a sign the loop isn't using the language's idioms.
- `for i, item in enumerate(items): print(f"{i}: {item}")` — the **fix**: `enumerate()` (introduced in Section 6) yields both the index and the value directly, in one clean, idiomatic line, with no manual counter and no re-indexing.

#### Full code block

```python
items = ["alpha", "beta", "gamma"]

# ❌ WRONG — manual index tracking (C-style, un-Pythonic)
i = 0
for item in items:
    print(f"{i}: {item}")
    i += 1

# ❌ ALSO WRONG — range(len()) is a code smell
for i in range(len(items)):
    print(f"{i}: {items[i]}")

# ✅ FIX — use enumerate()
for i, item in enumerate(items):
    print(f"{i}: {item}")
```

### 16.5 — Shallow vs Deep Copy Confusion

#### Why this section exists
Python's default assignment (`=`) never copies anything — it just makes a second name point at the same object. Even `.copy()` only copies **one level deep**, which is a trap for nested structures like a list of lists.

#### Line-by-line

- `import copy` — the standard library module providing `copy.deepcopy()`.
- `original = [[1, 2, 3], [4, 5, 6]]` then `alias = original; alias[0][0] = 999` — `alias = original` does **not** create a copy at all; `alias` and `original` are simply two names for the exact same list object, so mutating through either name affects "both" (because there's really only one object).
- `print("Original after alias modification:", original)` — shows `original` also changed, confirming `alias` was never independent.
- `shallow = original.copy()` then `shallow[0][0] = 999` — `.copy()` (or the equivalent `original[:]`) creates a **new outer list**, but the *inner* lists inside it are not copied — the new outer list just holds references to the exact same inner list objects as `original`. So replacing an entire top-level element of `shallow` wouldn't affect `original`, but mutating *into* a shared inner list (`shallow[0][0] = 999`) does.
- `print("Original after shallow modification:", original)` — shows the inner list was still shared, so `original` is affected despite using `.copy()`.
- `deep = copy.deepcopy(original)` then `deep[0][0] = 999` — `copy.deepcopy()` recursively copies **every level** of nested structure, producing a completely independent object graph with no shared references anywhere.
- `print("Original after deepcopy modification:", original)` — confirms `original` is now completely unaffected by changes to `deep`.

#### Full code block

```python
import copy

original = [[1, 2, 3], [4, 5, 6]]

# ❌ WRONG — assignment creates a reference, not a copy
alias = original
alias[0][0] = 999
print("Original after alias modification:", original)  # Also changed!

original = [[1, 2, 3], [4, 5, 6]]  # Reset

# ❌ ALSO WRONG for nested structures — shallow copy only copies the top level
shallow = original.copy()
shallow[0][0] = 999
print("Original after shallow modification:", original)  # Inner list still shared!

original = [[1, 2, 3], [4, 5, 6]]  # Reset

# ✅ FIX — deep copy creates completely independent copy
deep = copy.deepcopy(original)
deep[0][0] = 999
print("Original after deepcopy modification:", original)  # Unchanged ✅
```

### 16.6 — Catching Exceptions Too Broadly

#### Why this section exists
A bare `except:` (or an overly broad `except Exception:`) hides bugs by catching things you never intended to catch — including `KeyboardInterrupt` (Ctrl+C) in the bare case — making programs impossible to stop cleanly and masking real problems.

#### Line-by-line

- `try: result = 10 / 0 except:` — a **bare except** with no exception type specified. This catches *literally everything*, including `KeyboardInterrupt` and `SystemExit`, which are normally supposed to be able to interrupt a running program. Never write this.
- `try: result = 10 / 0 except Exception as e:` — better (at least it doesn't swallow `KeyboardInterrupt`/`SystemExit`, since those inherit from `BaseException`, not `Exception`), but still catches *any* kind of error that could occur in the `try` block, potentially hiding bugs that have nothing to do with the division you actually meant to guard against.
- `try: result = 10 / 0 except ZeroDivisionError: print(...); result = 0` — the **fix**: catch only the *specific* exception type you actually expect and know how to handle. If some other, unrelated exception occurs, it will propagate normally and be visible — rather than being silently absorbed by an overly broad `except`.
- `def load_config(path): try: with open(path) as f: return json.load(f) except FileNotFoundError: ... except json.JSONDecodeError as e: ...` — a realistic example with **multiple specific `except` clauses**, each handling a different, anticipated failure mode (missing file vs. malformed JSON) with a tailored message, before falling back to `return {}` as a safe default.

#### Full code block

```python
# ❌ WRONG — bare except catches EVERYTHING, including KeyboardInterrupt, SystemExit
try:
    result = 10 / 0
except:                          # Never do this!
    print("Something went wrong")

# ❌ ALSO WRONG — catching Exception is usually too broad
try:
    result = 10 / 0
except Exception as e:
    print(f"Error: {e}")        # Might silence real bugs

# ✅ FIX — catch only what you expect
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
    result = 0

# ✅ Multiple specific exceptions
def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config: {e}")
    return {}  # Return safe default
```

### 16.7 — String Concatenation in a Loop

#### Why this section exists
Strings are immutable, so `result += word` inside a loop doesn't extend the existing string — it builds an entirely new string object every single iteration and copies everything seen so far into it, making the whole loop O(n²) instead of O(n) for large inputs.

#### Line-by-line

- `words = ["Python", "is", "fast", "and", "readable"]` — the sample list.
- `result = ""` then `for word in words: result += word + " "` — each `+=` creates a brand-new string containing everything accumulated so far plus the new word; for `n` words, this does roughly `1 + 2 + 3 + ... + n` character copies in total — quadratic growth that becomes catastrophically slow for large lists (thousands or millions of items).
- `print(result.strip())` — removes the trailing extra space left by the loop's `+ " "` before printing.
- `result = " ".join(words)` — the **fix**: `str.join()` (introduced in Section 2) builds the final string in a single O(n) pass internally, because the join algorithm knows the total length up front and allocates memory once, rather than repeatedly reallocating and copying.
- `parts = [f"item_{i}" for i in range(5)]` then `", ".join(parts)` — a second example showing the same join pattern used to assemble formatted output from a list comprehension, reinforcing that "build a list, then join once" is the idiomatic replacement for "concatenate in a loop."

#### Full code block

```python
words = ["Python", "is", "fast", "and", "readable"]

# ❌ WRONG — strings are immutable, so += creates a new string each iteration
# This is O(n²) — catastrophically slow for large data
result = ""
for word in words:
    result += word + " "
print(result.strip())

# ✅ FIX — collect parts in a list, then join once
# str.join() is O(n) — much faster
result = " ".join(words)
print(result)

# ✅ Also fine for building formatted output
parts = [f"item_{i}" for i in range(5)]
print(", ".join(parts))
```

---

## 17. Common Python Error Types Explained

### Why this section exists
Recognizing an error by name — and knowing exactly what causes it — is the difference between a 30-second fix and an hour of confused debugging. This section walks through the most common built-in exception types you'll encounter, one at a time, each demonstrated live and then caught.

### 17.1 — SyntaxError

#### Why this section exists
`SyntaxError` is unique among these errors: it's caught **before your code ever runs**, because Python can't even finish parsing it into something executable. There is no `try/except` that can catch a `SyntaxError` in the same cell it occurs in — the cell simply fails to run at all.

#### Line-by-line

- The comment block explains, but does not execute, the common causes: a missing colon after `if`/`for`/`def`, mismatched parentheses/brackets/quotes, or an invalid expression.
- `# if True:` / `#     print("missing colon")` — the example of broken syntax (`if True` without the trailing colon) is deliberately left **commented out**, because if it were live code, it would prevent the entire notebook cell — and therefore the rest of the demo — from running at all.
- The "HOW TO FIX" comments summarize the recommended workflow: check the reported line number, then check the line **above** it too (Python's parser sometimes reports the error one line later than where the actual mistake is), and use an IDE with syntax highlighting to catch these before running the code at all.
- `print("SyntaxError demo — code above is commented out to allow notebook execution")` — the only line that actually executes in this cell, confirming the cell ran successfully despite discussing an error that would have prevented exactly that.

#### Full code block

```python
# A SyntaxError means Python couldn't even parse your code.
# These are caught BEFORE your code runs.

# Common causes:
#   - Missing colon after if/for/def
#   - Mismatched parentheses/brackets/quotes
#   - Invalid expression

# Uncomment to see the error:
# if True
#     print("missing colon")

# HOW TO FIX:
# 1. Look at the line number in the traceback
# 2. Check the line ABOVE too — sometimes Python reports one line late
# 3. Use an IDE with syntax highlighting

print("SyntaxError demo — code above is commented out to allow notebook execution")
```

### 17.2 — NameError

#### Why this section exists
`NameError` fires when Python can't find a name (variable, function) anywhere in the current scope chain — almost always a typo, a variable used before it was assigned, or a variable defined in a different (inaccessible) scope.

#### Line-by-line

- `try: print(undefined_variable) except NameError as e:` — `undefined_variable` was never assigned anywhere, so trying to read it raises `NameError`; the `except` catches it and prints the exception's message (something like `"name 'undefined_variable' is not defined"`).
- `try: total = totl + 1 except NameError as e:` — a more realistic scenario: `totl` is a **typo** for `total`, so this also raises `NameError`, demonstrating the most common real-world cause of this error.
- The printed fix advice reinforces the two things to check: spelling, and whether the variable is actually defined *before* the line that uses it (definition order matters top-to-bottom in Python).

#### Full code block

```python
# NameError: name 'x' is not defined
# Cause: typo in variable name, using before assignment, or wrong scope

try:
    print(undefined_variable)
except NameError as e:
    print(f"NameError caught: {e}")

# Common scenario: forgetting to define a variable
try:
    total = totl + 1   # Typo: 'totl' instead of 'total'
except NameError as e:
    print(f"NameError caught: {e}")
    print("Fix: Check for typos and ensure the variable is defined before use")
```

### 17.3 — TypeError

#### Why this section exists
`TypeError` means an operation or function was given a value of a type it fundamentally cannot work with — e.g. trying to add a string and an integer, where Python has no built-in rule for what `"5" + 5` should mean.

#### Line-by-line

- `try: result = "5" + 5 except TypeError as e:` — the `+` operator is overloaded differently for `str` (concatenation) and `int` (addition), and Python refuses to implicitly guess which one you meant when mixing the two types, raising `TypeError` instead.
- `print(f"  int + int = {int('5') + 5}")` and `print(f"  str + str = {'5' + str(5)}")` — shows both valid fixes: explicitly convert the string to an `int` first (giving `10`), or explicitly convert the integer to a `str` first (giving `"55"`) — the two are **not** the same result, which is exactly why Python won't guess for you.
- `try: def add(a, b): return a + b; add(1, 2, 3) except TypeError as e:` — a second, different cause of the same exception type: calling a function with the **wrong number of arguments**. `add` was defined to accept exactly two parameters; passing three raises `TypeError` with a message about the argument count mismatch.

#### Full code block

```python
# TypeError: operation or function applied to wrong type

try:
    result = "5" + 5    # Can't add str and int
except TypeError as e:
    print(f"TypeError: {e}")
    print("Fix: int('5') + 5  or  '5' + str(5)")
    print(f"  int + int = {int('5') + 5}")
    print(f"  str + str = {'5' + str(5)}")

try:
    def add(a, b): return a + b
    add(1, 2, 3)   # Too many arguments
except TypeError as e:
    print(f"TypeError: {e}")
```

### 17.4 — ValueError

#### Why this section exists
`ValueError` is subtly different from `TypeError`: the **type** is correct (a `str` was passed where a `str` was expected), but the specific **value** doesn't make sense for what the operation is trying to do.

#### Line-by-line

- `try: num = int("hello") except ValueError as e:` — `int()` accepts a `str` argument in general, so this isn't a type mismatch; but `"hello"` isn't a string that represents any valid integer, so it's a value problem, hence `ValueError` rather than `TypeError`.
- `def safe_int(s): try: return int(s) except ValueError: return None` — a small reusable helper that converts a conversion failure into a safe `None` return value instead of letting the exception propagate — a common defensive-programming pattern.
- `print(safe_int("42"))` and `print(safe_int("hello"))` — show the helper succeeding (`42`) and failing gracefully (`None`) respectively.
- `try: import math; math.sqrt(-1) except ValueError as e:` — a second example: `math.sqrt` only accepts non-negative numbers by design (it's not the type of `-1` that's wrong, `-1` is a perfectly valid `int`/`float` — it's that no real square root exists for a negative number), so this also raises `ValueError`, not `TypeError`.

#### Full code block

```python
# ValueError: type is correct but value is inappropriate

try:
    num = int("hello")   # str type, but not a valid integer string
except ValueError as e:
    print(f"ValueError: {e}")
    print("Fix: Validate input before conversion")

# Safe conversion pattern
def safe_int(s):
    try:
        return int(s)
    except ValueError:
        return None

print(safe_int("42"))     # 42
print(safe_int("hello"))  # None

try:
    import math
    math.sqrt(-1)          # Can't take sqrt of negative (use complex() for that)
except ValueError as e:
    print(f"ValueError: {e}")
```

### 17.5 — IndexError & KeyError

#### Why this section exists
These two errors are the sequence/mapping equivalents of each other: `IndexError` for accessing a numeric position that doesn't exist in a list/tuple, `KeyError` for accessing a key that doesn't exist in a dict. Both have the same core fix pattern — check before you access, or use a safe accessor with a default.

#### Line-by-line

- `my_list = [10, 20, 30]` then `try: print(my_list[5]) except IndexError as e:` — the list only has valid indices `0`, `1`, `2`; asking for index `5` is out of range, raising `IndexError`.
- `print(f"Fix: List has {len(my_list)} items, valid indices: 0 to {len(my_list)-1}")` — the printed guidance shows how to compute the valid index range programmatically, using `len()`.
- `config = {"batch_size": 32, "epochs": 10}` then `try: lr = config["learning_rate"] except KeyError as e:` — `"learning_rate"` was never added to `config`, so bracket access raises `KeyError` (whereas a list would need an *out-of-range integer* to raise the sequence equivalent).
- `lr = config.get("learning_rate", 0.001)` — the fix: `.get(key, default)` (introduced in Section 4) never raises for a missing key — it just returns the given default instead.
- `if "learning_rate" in config: lr = config["learning_rate"] else: print(...)` — an alternative fix pattern: explicitly test membership with `in` before doing the direct bracket access, useful when you want different logic for the "missing" case beyond just substituting a default value.

#### Full code block

```python
# IndexError — list index out of range
my_list = [10, 20, 30]
try:
    print(my_list[5])    # Index 5 doesn't exist
except IndexError as e:
    print(f"IndexError: {e}")
    print(f"Fix: List has {len(my_list)} items, valid indices: 0 to {len(my_list)-1}")

print()

# KeyError — dictionary key not found
config = {"batch_size": 32, "epochs": 10}
try:
    lr = config["learning_rate"]   # Key doesn't exist
except KeyError as e:
    print(f"KeyError: {e}")

# ✅ Fix: Use .get() with a default value
lr = config.get("learning_rate", 0.001)
print(f"Learning rate (with default): {lr}")

# ✅ Fix: Check existence first
if "learning_rate" in config:
    lr = config["learning_rate"]
else:
    print("Key not found, using default")
```

### 17.6 — AttributeError

#### Why this section exists
`AttributeError` fires when you try to access a method or attribute that a particular object's type simply doesn't have. One of the most common real-world triggers is calling a method on a variable that turned out to be `None` — usually because an earlier function silently returned `None` instead of the value you expected.

#### Line-by-line

- `try: x = 42; x.append(1) except AttributeError as e:` — `.append()` is a `list` method; `int` objects have no such method at all, so Python raises `AttributeError` the moment it looks up `.append` on `x` and doesn't find it.
- `print(f"  type(x) = {type(x)}")` — the suggested debugging step: print the actual `type()` of the offending variable to confirm your assumption about what it should be was wrong.
- `try: result = None; result.upper() except AttributeError as e:` — a **very** common real-world pattern: some earlier function call returned `None` (perhaps because it hit an edge case or failed silently), and calling `.upper()` — a `str` method — on `None` raises `AttributeError` because `NoneType` has no `.upper()` method.
- `def process(text): if text is None: return ""; return text.upper()` — the fix pattern: explicitly guard against `None` before calling any method on a value that might be missing, using `is None` (the idiomatic identity check from Section 5/16.3).
- `print(process(None))` and `print(process("hello"))` — show the guarded function handling both the "missing" case (returns `""` safely) and the normal case (`"HELLO"`).

#### Full code block

```python
# AttributeError: 'type' object has no attribute 'name'

try:
    x = 42
    x.append(1)   # int has no .append() method — that's for lists
except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Fix: Check that your variable is the type you expect")
    print(f"  type(x) = {type(x)}")

try:
    result = None
    result.upper()   # NoneType has no .upper() — common when a function returns None
except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Fix: Check that your variable is not None before calling methods")

# Pattern for handling potential None values
def process(text):
    if text is None:
        return ""
    return text.upper()

print(process(None))     # '' — safe
print(process("hello"))  # 'HELLO'
```

### 17.7 — ImportError & ModuleNotFoundError

#### Why this section exists
`ModuleNotFoundError` is a subclass of the broader `ImportError` — the former means the module itself can't be located at all, the latter (in its narrower sense) means the module was found but a specific name you tried to import from it doesn't exist. Both are also useful for building **optional dependency** patterns.

#### Line-by-line

- `try: import nonexistent_library except ModuleNotFoundError as e:` — `nonexistent_library` isn't installed (or doesn't exist at all), so Python can't locate it on `sys.path`, raising `ModuleNotFoundError`. The suggested fix is simply installing it with `pip install <package_name>`.
- `try: from math import nonexistent_function except ImportError as e:` — this time the *module* `math` is found successfully, but it has no attribute called `nonexistent_function`, so the narrower `ImportError` fires (not `ModuleNotFoundError`, since the module itself did exist) — the fix here is to check the module's actual documented API for the correct name.
- `try: import numpy as np; HAS_NUMPY = True except ImportError: HAS_NUMPY = False; print(...)` — the **graceful optional import** pattern: wrap an import for an optional dependency in `try/except`, set a boolean flag reflecting whether it succeeded, and let the rest of the program branch on that flag rather than crashing outright if the dependency isn't installed.
- `if HAS_NUMPY: arr = np.array([1, 2, 3]); print(...) else: print("Using plain Python list instead")` — demonstrates the payoff of the pattern above: the code adapts its behavior at runtime based on what's actually available, rather than requiring every dependency to be installed just to run at all.

#### Full code block

```python
# ModuleNotFoundError is a subclass of ImportError
# Occurs when Python cannot find the module you're trying to import

try:
    import nonexistent_library
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Fix: Install the package with:  pip install <package_name>")

# ImportError — module exists but specific name does not
try:
    from math import nonexistent_function
except ImportError as e:
    print(f"ImportError: {e}")
    print("Fix: Check the module's documentation for the correct name")

# Best practice: graceful optional import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("numpy not available — some features will be disabled")

if HAS_NUMPY:
    arr = np.array([1, 2, 3])
    print(f"NumPy array: {arr}")
else:
    print("Using plain Python list instead")
```

### 17.8 — RecursionError

#### Why this section exists
Recursion (a function calling itself) needs a **base case** — a condition under which it stops calling itself and simply returns. Without one, each call stacks another frame on Python's call stack until it hits the interpreter's built-in recursion limit and raises `RecursionError`.

#### Line-by-line

- `def count_down_wrong(n): print(n); count_down_wrong(n - 1)` — defined but **never actually called** in this cell (it's shown purely as a "don't do this" illustration) — it has no base case, so calling it would recurse forever (in practice, until `RecursionError` is raised) because `n` just keeps decreasing without ever stopping the recursive calls.
- `def count_down(n): if n < 0: return; print(n, end=" "); count_down(n - 1)` — the fix: `if n < 0: return` is the **base case** — the condition that eventually becomes true and stops the recursion instead of calling itself again.
- `count_down(5)` — actually executed; recurses `count_down(5) → count_down(4) → ... → count_down(-1)`, where the base case finally triggers and unwinds the stack, printing `5 4 3 2 1 0`.
- `print(f"Default recursion limit: {sys.getrecursionlimit()}")` — `sys.getrecursionlimit()` reports Python's configured maximum call-stack depth (commonly 1000 by default), the threshold beyond which any recursive function — even a correct one on sufficiently deep input — raises `RecursionError`.
- The commented `# sys.setrecursionlimit(5000)` shows that the limit **can** be raised, with an explicit "use with caution" warning, since raising it too far risks crashing the whole Python process with a genuine C-level stack overflow instead of a clean, catchable `RecursionError`.
- `def fib_iterative(n): a, b = 0, 1; for _ in range(n): a, b = b, a + b; return a` — the recommended alternative for cases where recursion depth would be a problem: an **iterative** version of Fibonacci using a simple loop and no function-call stack growth at all, capable of handling much larger `n` safely.
- `print(f"fib(50) = {fib_iterative(50)}")` — computes the 50th Fibonacci number instantly, something a naive (non-memoized) recursive Fibonacci implementation would take an impractically long time to do, since its call count grows exponentially with `n`.

#### Full code block

```python
import sys

# ❌ Infinite recursion — no base case!
def count_down_wrong(n):
    print(n)
    count_down_wrong(n - 1)   # Never stops!

# ✅ Recursion with a proper base case
def count_down(n):
    if n < 0:         # Base case — stop recursion
        return
    print(n, end=" ")
    count_down(n - 1)

count_down(5)
print()

# Python's default recursion limit
print(f"Default recursion limit: {sys.getrecursionlimit()}")

# For very deep recursion, consider iterative approach or increase limit
# sys.setrecursionlimit(5000)  # Use with caution!

# ✅ Better: iterative fibonacci for large n
def fib_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(f"fib(50) = {fib_iterative(50)}")
```

---

## 18. Group Coding Challenges

### Why this section exists
The remaining code cells in the notebook are **student exercise stubs**, not worked solutions. Each one pairs with a markdown cell (shown here summarized under "Why this section exists") describing the task; the code cell itself contains only a header comment, occasionally some starter data the challenge specifies, and a `# Your group's code here` placeholder for students to fill in during the live session. This walkthrough documents exactly what starter code ships in each stub and what the challenge asks students to build on top of it — there is no implementation logic to walk through because none is provided.

### Topic 1 — Variables & Data Types

#### Challenge 1.1 — Type Inspector

##### Why this section exists
Reinforces Section 1: students create one variable of each core type (`int`, `float`, `str`, `bool`, `NoneType`) and loop over them printing value, `type()`, and an `isinstance(x, (int, float))` numeric check with aligned f-string output.

##### Line-by-line
- `# Challenge 1.1 — Type Inspector` — a header comment identifying which challenge this cell belongs to; every challenge stub in the notebook follows this same naming pattern.
- `# Your group's code here` — the placeholder line where students write their solution; no starter data is provided for this particular challenge, since the task is to *create* the five variables from scratch.

##### Full code block

```python
# Challenge 1.1 — Type Inspector
# Your group's code here
```

#### Challenge 1.2 — Dynamic Swap Chain

##### Why this section exists
Reinforces multiple assignment, one-line variable rotation (`a, b, c = b, c, a`-style unpacking), and dynamic typing by reassigning one variable through three different types in sequence.

##### Line-by-line
- `# Challenge 1.2 — Dynamic Swap Chain` — header comment.
- `# Your group's code here` — placeholder; students are expected to start by writing `a = 10`, `b = 20`, `c = 30` themselves as step 1 of the instructions, since no starter data is pre-supplied here either.

##### Full code block

```python
# Challenge 1.2 — Dynamic Swap Chain
# Your group's code here
```

#### Challenge 1.3 — Type Conversion Pipeline

##### Why this section exists
Reinforces `try`/`except`-based type coercion: attempt `int()`, fall back to `float()`, fall back to `str`, and tally how many values land in each bucket.

##### Line-by-line
- `# Challenge 1.3 — Type Conversion Pipeline` — header comment.
- `raw = ["42", "3.14", "True", "0", "99", "False"]` — the starter data supplied directly in the stub (matching the spec exactly), a list of raw strings simulating sensor-log values with mixed underlying types.
- `# Your group's code here` — placeholder for the conversion-pipeline logic operating on `raw`.

##### Full code block

```python
# Challenge 1.3 — Type Conversion Pipeline
raw = ["42", "3.14", "True", "0", "99", "False"]
# Your group's code here
```

#### Challenge 1.4 — Variable Alias Detective

##### Why this section exists
Reinforces the mutable-alias-vs-copy concept from Section 16.5 (shallow vs deep copy) in miniature: comparing `y = x` (alias) against `z = x[:]` (shallow copy) after mutating `x`, then verifying with `is`.

##### Line-by-line
- `# Challenge 1.4 — Variable Alias Detective` — header comment.
- `# Your group's code here` — placeholder; students are expected to write the `x = [1, 2, 3]; y = x; z = x[:]; x.append(4)` setup themselves per the instructions, then predict and print `x`, `y`, `z`, and `is` comparisons between them.

##### Full code block

```python
# Challenge 1.4 — Variable Alias Detective
# Your group's code here
```

### Topic 2 — Strings & String Operations

#### Challenge 2.1 — Username Normalizer

##### Why this section exists
Reinforces `.strip()`, `.title()`, and `.replace()` chained together to build a `normalize(name)` function that cleans messy, inconsistently-cased user input.

##### Line-by-line
- `# Challenge 2.1 — Username Normalizer` — header comment.
- `raw_names = ["  Alan Turing  ", "ADA LOVELACE", "grace hopper ", "  LINUS torvalds"]` — starter data: a list of names with inconsistent casing and stray whitespace, exactly as specified in the challenge markdown.
- `# Your group's code here` — placeholder for the `normalize()` function definition and its application to each name in `raw_names`.

##### Full code block

```python
# Challenge 2.1 — Username Normalizer
raw_names = ["  Alan Turing  ", "ADA LOVELACE", "grace hopper ", "  LINUS torvalds"]
# Your group's code here
```

#### Challenge 2.2 — Palindrome Checker

##### Why this section exists
Reinforces string cleaning (removing non-alphanumeric characters, lowercasing) and slicing/reversal (`s[::-1]` from Section 2) to detect palindromes across full sentences, not just single words.

##### Line-by-line
- `# Challenge 2.2 — Palindrome Checker` — header comment.
- `# Your group's code here` — placeholder; the challenge supplies its test strings (`"racecar"`, `"A man a plan a canal Panama"`, etc.) only in the markdown instructions, not as pre-populated code, so students must write the test calls themselves alongside the `is_palindrome()` function.

##### Full code block

```python
# Challenge 2.2 — Palindrome Checker
# Your group's code here
```

#### Challenge 2.3 — f-String Report Card

##### Why this section exists
Reinforces f-string field-width alignment and decimal rounding (`:<10`, `:.1f`-style format specs) to render a formatted, aligned report card from a list of student dicts.

##### Line-by-line
- `# Challenge 2.3 — f-String Report Card` — header comment.
- `students = [{"name": "Maria", "score": 94.678, "passed": True}, {"name": "James", "score": 58.3, "passed": False}, {"name": "Yuki", "score": 76.0, "passed": True}]` — starter data: a list of dicts, each representing one student's name, raw (unrounded) score, and pass/fail status.
- `# Your group's code here` — placeholder for the formatted-printing logic operating on `students`.

##### Full code block

```python
# Challenge 2.3 — f-String Report Card
students = [
    {"name": "Maria",  "score": 94.678, "passed": True},
    {"name": "James",  "score": 58.3,   "passed": False},
    {"name": "Yuki",   "score": 76.0,   "passed": True},
]
# Your group's code here
```

#### Challenge 2.4 — Word Frequency Counter

##### Why this section exists
Reinforces `.split()`, `.lower()`, dict-based counting (or `collections.Counter` from Section 13), and `sorted()` with a `key=` to rank words by frequency — a direct precursor to real text-preprocessing work.

##### Line-by-line
- `# Challenge 2.4 — Word Frequency Counter` — header comment.
- `sentence = "To be or not to be that is the question whether tis nobler in the mind to suffer"` — the starter sentence to tokenize and count, provided verbatim in the stub.
- `# Your group's code here` — placeholder for the splitting/counting/sorting logic.

##### Full code block

```python
# Challenge 2.4 — Word Frequency Counter
sentence = "To be or not to be that is the question whether tis nobler in the mind to suffer"
# Your group's code here
```

### Topic 3 — Numbers & Arithmetic

#### Challenge 3.1 — Unit Converter

##### Why this section exists
Reinforces writing small, single-purpose functions (Section 7) that each perform one arithmetic conversion, plus `round()` and f-string tables.

##### Line-by-line
- `# Challenge 3.1 — Unit Converter` — header comment.
- `# Your group's code here` — placeholder; students define the three conversion functions (`celsius_to_fahrenheit`, `km_to_miles`, `kg_to_pounds`) and the table-printing loop themselves, with no starter data pre-populated since the input list is left to their choice.

##### Full code block

```python
# Challenge 3.1 — Unit Converter
# Your group's code here
```

#### Challenge 3.2 — FizzBuzz (with a twist)

##### Why this section exists
Reinforces `%` (modulus) and `//` (floor division) plus multi-branch `if`/`elif` logic (Section 5) in the classic FizzBuzz exercise, extended with a fourth rule ("Boom" on multiples of 7) and a running tally of each label.

##### Line-by-line
- `# Challenge 3.2 — FizzBuzz with a twist` — header comment.
- `# Your group's code here` — placeholder for the loop over `1..50` implementing the Fizz/Buzz/FizzBuzz/Boom rules and the final label counts.

##### Full code block

```python
# Challenge 3.2 — FizzBuzz with a twist
# Your group's code here
```

#### Challenge 3.3 — Circle Stats Calculator

##### Why this section exists
Reinforces `math.pi`, returning multiple values as a tuple (Section 4), and rounding, by computing area/circumference/diameter for several radii.

##### Line-by-line
- `# Challenge 3.3 — Circle Stats Calculator` — header comment.
- `import math` — pre-supplied import, since the challenge specifically requires `math.pi`.
- `# Your group's code here` — placeholder for the `circle_stats(radius)` function and the loop over `[1, 5, 10, 0.5, 100]`.

##### Full code block

```python
# Challenge 3.3 — Circle Stats Calculator
import math
# Your group's code here
```

#### Challenge 3.4 — Compound Interest Calculator

##### Why this section exists
Reinforces translating a mathematical formula directly into a Python function with multiple parameters, plus `round()` and formatted output — a very common real pattern in financial/engineering calculations.

##### Line-by-line
- `# Challenge 3.4 — Compound Interest Calculator` — header comment.
- `# Your group's code here` — placeholder for the `compound_interest(P, r, n, t)` function and its three test calls with the given parameter sets.

##### Full code block

```python
# Challenge 3.4 — Compound Interest Calculator
# Your group's code here
```

### Topic 4 — Collections: Lists, Tuples, Sets & Dictionaries

#### Challenge 4.1 — AI Model Registry

##### Why this section exists
Reinforces nested dictionaries, filtering/searching a dict by iterating `.items()`, and the `|=` dict-update operator (Section 4) in a realistic "config registry" shape.

##### Line-by-line
- `# Challenge 4.1 — AI Model Registry` — header comment.
- `# Your group's code here` — placeholder; students build the `model_registry` nested dict themselves (populating it with at least 5 models of their choice), so no starter data is pre-supplied.

##### Full code block

```python
# Challenge 4.1 — AI Model Registry
# Your group's code here
```

#### Challenge 4.2 — Set Operations for Tag Deduplication

##### Why this section exists
Reinforces the set operators (`&`, `-`, `|`, and subset checking) from Section 4 applied to a realistic "article tags" deduplication scenario across three sets.

##### Line-by-line
- `# Challenge 4.2 — Set Operations` — header comment.
- `article_a_tags = {"python", "ml", "tutorial", "beginner", "data"}`, `article_b_tags = {"python", "deep-learning", "tutorial", "advanced", "neural-networks"}`, `article_c_tags = {"javascript", "frontend", "tutorial", "beginner"}` — three starter sets provided exactly as specified in the challenge.
- `# Your group's code here` — placeholder for the intersection/difference/union/subset logic across the three tag sets.

##### Full code block

```python
# Challenge 4.2 — Set Operations
article_a_tags = {"python", "ml", "tutorial", "beginner", "data"}
article_b_tags = {"python", "deep-learning", "tutorial", "advanced", "neural-networks"}
article_c_tags = {"javascript", "frontend", "tutorial", "beginner"}
# Your group's code here
```

#### Challenge 4.3 — Student Gradebook

##### Why this section exists
Reinforces working with a list of tuples (Section 4) without pandas — computing per-student averages, finding a maximum, grouping by a field, and sorting by score, all with built-in tools only.

##### Line-by-line
- `# Challenge 4.3 — Student Gradebook` — header comment.
- `gradebook = [("Alice", "Math", 88), ("Alice", "Science", 92), ("Alice", "English", 76), ("Bob", "Math", 73), ("Bob", "Science", 81), ("Bob", "English", 90), ("Carol", "Math", 95), ("Carol", "Science", 87), ("Carol", "English", 84)]` — the starter data: nine `(student_name, subject, score)` tuples, provided exactly as specified.
- `# Your group's code here` — placeholder for the averaging, top-student lookup, per-subject max, and descending sort logic.

##### Full code block

```python
# Challenge 4.3 — Student Gradebook
gradebook = [
    ("Alice", "Math", 88), ("Alice", "Science", 92), ("Alice", "English", 76),
    ("Bob",   "Math", 73), ("Bob",   "Science", 81), ("Bob",   "English", 90),
    ("Carol", "Math", 95), ("Carol", "Science", 87), ("Carol", "English", 84),
]
# Your group's code here
```

#### Challenge 4.4 — Inventory Management System

##### Why this section exists
Reinforces nested dictionaries plus writing small functions that operate on shared mutable state (Section 7 + Section 4 combined) — a `total_value`, a `restock` that raises `KeyError` intentionally, and a threshold filter.

##### Line-by-line
- `# Challenge 4.4 — Inventory Management System` — header comment.
- `inventory = {"apples": {"price": 0.5, "stock": 100}, "bananas": {"price": 0.25, "stock": 150}, "cherries": {"price": 3.0, "stock": 40}, "dates": {"price": 5.0, "stock": 20}}` — starter data: a dict of dicts modeling a store's stock, provided exactly as specified.
- `# Your group's code here` — placeholder for the three required functions (`total_value`, `restock`, `items_below_threshold`) and their test calls.

##### Full code block

```python
# Challenge 4.4 — Inventory Management System
inventory = {
    "apples":  {"price": 0.5,  "stock": 100},
    "bananas": {"price": 0.25, "stock": 150},
    "cherries":{"price": 3.0,  "stock": 40},
    "dates":   {"price": 5.0,  "stock": 20},
}
# Your group's code here
```

### Topic 5 — Control Flow & Loops

#### Challenge 5.1 — Number Classifier

##### Why this section exists
Reinforces multi-branch `if`/`elif` (Section 5) plus a hand-rolled primality check requiring a nested loop or early `break`.

##### Line-by-line
- `# Challenge 5.1 — Number Classifier` — header comment.
- `# Your group's code here` — placeholder for the `classify_number(n)` function and its test loop over `[-5, 0, 4, 7, 13, 15, 2, 100]`.

##### Full code block

```python
# Challenge 5.1 — Number Classifier
# Your group's code here
```

#### Challenge 5.2 — Password Strength Validator

##### Why this section exists
Reinforces string membership checks, length checks, and accumulating a score plus a list of failure reasons — a very common real-world validation pattern.

##### Line-by-line
- `# Challenge 5.2 — Password Strength Validator` — header comment.
- `# Your group's code here` — placeholder for `check_password(password)` and its four test calls.

##### Full code block

```python
# Challenge 5.2 — Password Strength Validator
# Your group's code here
```

#### Challenge 5.3 — Pattern Printer

##### Why this section exists
Reinforces nested `for` loops (Section 6) — the classic exercise for building an intuition for how outer/inner loop iteration counts interact to produce 2D visual patterns.

##### Line-by-line
- `# Challenge 5.3 — Pattern Printer` — header comment.
- `# Your group's code here` — placeholder for the three pattern-printing functions (triangle, multiplication table, diamond).

##### Full code block

```python
# Challenge 5.3 — Pattern Printer
# Your group's code here
```

#### Challenge 5.4 — Collatz Conjecture

##### Why this section exists
Reinforces `while` loops with a non-obvious termination condition (Section 6), and tracking auxiliary state (a running maximum) alongside a step counter.

##### Line-by-line
- `# Challenge 5.4 — Collatz Conjecture` — header comment.
- `# Your group's code here` — placeholder for `collatz_steps(n)` and the `range(1, 101)` search for the slowest-converging starting number.

##### Full code block

```python
# Challenge 5.4 — Collatz Conjecture
# Your group's code here
```

### Topic 6 — Functions & Lambda Functions

#### Challenge 6.1 — Flexible Stats Function

##### Why this section exists
Reinforces `*args` combined with a keyword-only-style default parameter (Section 7) to build a variadic statistics function.

##### Line-by-line
- `# Challenge 6.1 — Flexible Stats Function` — header comment.
- `# Your group's code here` — placeholder for `stats(*numbers, precision=2)` and its three test calls.

##### Full code block

```python
# Challenge 6.1 — Flexible Stats Function
# Your group's code here
```

#### Challenge 6.2 — Function as First-Class Object

##### Why this section exists
Reinforces that functions (including unbound methods like `str.upper`) can be stored in a list and applied dynamically, plus closures — `make_multiplier(factor)` returning a new function that "remembers" `factor`.

##### Line-by-line
- `# Challenge 6.2 — Functions as First-Class Objects` — header comment.
- `transforms = [str.upper, str.lower, str.title, str.strip]` — starter data: a list of the (unbound) `str` methods themselves as first-class function objects, provided exactly as specified.
- `# Your group's code here` — placeholder for `apply_all(text, funcs)` and the `make_multiplier`/`double`/`triple`/`halve` closures.

##### Full code block

```python
# Challenge 6.2 — Functions as First-Class Objects
transforms = [str.upper, str.lower, str.title, str.strip]
# Your group's code here
```

#### Challenge 6.3 — Lambda Sort-Off

##### Why this section exists
Reinforces `sorted(..., key=lambda ...)` (Section 8) across four different sort criteria on the same dataset, including a computed (derived) sort key.

##### Line-by-line
- `# Challenge 6.3 — Lambda Sort-Off` — header comment.
- `papers = [{"title": "Attention Is All You Need", "year": 2017, "citations": 90000}, {"title": "BERT", "year": 2018, "citations": 55000}, {"title": "GPT-3", "year": 2020, "citations": 30000}, {"title": "ResNet", "year": 2015, "citations": 120000}, {"title": "AlphaFold", "year": 2021, "citations": 18000}]` — starter data: five AI paper records, provided exactly as specified.
- `# Your group's code here` — placeholder for the four `sorted()` calls (by citations, by year, by title, by citations-per-year).

##### Full code block

```python
# Challenge 6.3 — Lambda Sort-Off
papers = [
    {"title": "Attention Is All You Need", "year": 2017, "citations": 90000},
    {"title": "BERT",                      "year": 2018, "citations": 55000},
    {"title": "GPT-3",                     "year": 2020, "citations": 30000},
    {"title": "ResNet",                    "year": 2015, "citations": 120000},
    {"title": "AlphaFold",                 "year": 2021, "citations": 18000},
]
# Your group's code here
```

#### Challenge 6.4 — Recursive Fibonacci with Memoization

##### Why this section exists
Reinforces recursion (Section 17.8), a hand-rolled memoization cache (a precursor to the `@memoize` decorator in Challenge 10.4), and timing comparisons with the `time` module.

##### Line-by-line
- `# Challenge 6.4 — Recursive Fibonacci with Memoization` — header comment.
- `import time` — pre-supplied import, since the challenge specifically requires timing each version.
- `# Your group's code here` — placeholder for `fib_recursive(n)`, `fib_memo(n, cache={})`, and the timing/comparison logic. (Note: `cache={}` as specified in the instructions is itself an instance of the mutable-default-argument pattern from Section 16.1 — here used *intentionally* as a persistent cache rather than accidentally, illustrating that the "gotcha" can also be a deliberate technique when used knowingly.)

##### Full code block

```python
# Challenge 6.4 — Recursive Fibonacci with Memoization
import time
# Your group's code here
```

### Topic 7 — List Comprehensions

#### Challenge 7.1 — Data Cleaning Pipeline

##### Why this section exists
Reinforces chaining multiple comprehension steps (Section 9) — strip, filter-valid, convert, filter-by-value — first separately, then combined into one comprehension.

##### Line-by-line
- `# Challenge 7.1 — Data Cleaning Pipeline` — header comment.
- `raw_scores = ["85", " 90 ", "72", "N/A", "88", "", "95", "invalid", "61"]` — starter data: a deliberately messy list of score strings, provided exactly as specified.
- `# Your group's code here` — placeholder for the multi-step and single-line comprehension solutions.

##### Full code block

```python
# Challenge 7.1 — Data Cleaning Pipeline
raw_scores = ["85", " 90 ", "72", "N/A", "88", "", "95", "invalid", "61"]
# Your group's code here
```

#### Challenge 7.2 — Dictionary & Set Comprehensions

##### Why this section exists
Reinforces dict, set, and nested list comprehensions (Section 9) applied to grouping and pairing words by shared starting letters.

##### Line-by-line
- `# Challenge 7.2 — Dictionary & Set Comprehensions` — header comment.
- `words = ["apple", "banana", "cherry", "avocado", "blueberry", "apricot", "coconut"]` — starter data: a list of fruit names, provided exactly as specified.
- `# Your group's code here` — placeholder for the four comprehension tasks (lengths dict, first-letters set, grouped-by-letter dict, same-letter word pairs).

##### Full code block

```python
# Challenge 7.2 — Dictionary & Set Comprehensions
words = ["apple", "banana", "cherry", "avocado", "blueberry", "apricot", "coconut"]
# Your group's code here
```

#### Challenge 7.3 — Matrix Operations with Comprehensions

##### Why this section exists
Reinforces nested list comprehensions (Section 9) for classic 2D-matrix tasks: an identity matrix, a multiplication table, a transpose, and flattening.

##### Line-by-line
- `# Challenge 7.3 — Matrix Operations with Comprehensions` — header comment.
- `matrix = [[1,2,3],[4,5,6],[7,8,9]]` — starter data: the 3×3 matrix to transpose and flatten, provided exactly as specified.
- `# Your group's code here` — placeholder for the identity-matrix, multiplication-table, transpose, and flattening comprehensions.

##### Full code block

```python
# Challenge 7.3 — Matrix Operations with Comprehensions
matrix = [[1,2,3],[4,5,6],[7,8,9]]
# Your group's code here
```

#### Challenge 7.4 — FizzBuzz, Comprehension-Style

##### Why this section exists
Revisits Challenge 3.2's FizzBuzz logic but forces it into a single list comprehension (Section 9), including a conditional *expression* (ternary, Section 5) inside the comprehension body — a good test of whether students can compress branching logic into an expression.

##### Line-by-line
- `# Challenge 7.4 — FizzBuzz, Comprehension-Style` — header comment.
- `# Your group's code here` — placeholder for the single-comprehension FizzBuzz solution and the follow-up "Fizz" counting comprehension.

##### Full code block

```python
# Challenge 7.4 — FizzBuzz, Comprehension-Style
# Your group's code here
```

### Topic 8 — Object-Oriented Programming

#### Challenge 8.1 — BankAccount Class

##### Why this section exists
Reinforces class design (Section 12): instance attributes with defaults, methods that validate and mutate state, and a `__repr__` for debugging.

##### Line-by-line
- `# Challenge 8.1 — BankAccount Class` — header comment.
- `# Your group's code here` — placeholder for the full `BankAccount` class definition (attributes, `deposit`, `withdraw`, `get_balance`, `print_statement`, `__repr__`) and the two-account demo.

##### Full code block

```python
# Challenge 8.1 — BankAccount Class
# Your group's code here
```

#### Challenge 8.2 — Shape Hierarchy

##### Why this section exists
Reinforces inheritance and polymorphism (Section 12): a base `Shape` class with methods that intentionally `raise NotImplementedError`, and three subclasses that each provide their own `area()`/`perimeter()`.

##### Line-by-line
- `# Challenge 8.2 — Shape Hierarchy` — header comment.
- `import math` — pre-supplied import, needed for `Circle.area()`'s use of `math.pi`.
- `# Your group's code here` — placeholder for the `Shape` base class and the `Circle`, `Rectangle`, `Triangle` subclasses, plus the sort-by-area demo.

##### Full code block

```python
# Challenge 8.2 — Shape Hierarchy
import math
# Your group's code here
```

#### Challenge 8.3 — Student Management System

##### Why this section exists
Reinforces composing two related classes together (Section 12) — a `Classroom` that holds a list of `Student` objects and aggregates data across all of them.

##### Line-by-line
- `# Challenge 8.3 — Student Management System` — header comment.
- `# Your group's code here` — placeholder for both the `Student` and `Classroom` class definitions and the four-student demo.

##### Full code block

```python
# Challenge 8.3 — Student Management System
# Your group's code here
```

#### Challenge 8.4 — Dunder Methods & Operator Overloading

##### Why this section exists
Reinforces dunder/magic methods (Section 12) beyond `__repr__`/`__str__` — implementing `__add__`, `__sub__`, `__mul__`, `__eq__`, and `__abs__` to make a custom `Vector2D` class work naturally with Python's built-in operators.

##### Line-by-line
- `# Challenge 8.4 — Dunder Methods & Operator Overloading` — header comment.
- `import math` — pre-supplied import, needed for `__abs__`'s square-root magnitude calculation.
- `# Your group's code here` — placeholder for the full `Vector2D` class and the `v1`/`v2` operator demo.

##### Full code block

```python
# Challenge 8.4 — Dunder Methods & Operator Overloading
import math
# Your group's code here
```

### Topic 9 — Exception Handling

#### Challenge 9.1 — Safe Division Calculator

##### Why this section exists
Reinforces the full `try`/`except`/`finally` structure (Section 11) with two distinct exception types and a guaranteed cleanup message.

##### Line-by-line
- `# Challenge 9.1 — Safe Division Calculator` — header comment.
- `# Your group's code here` — placeholder for `safe_divide(a, b)` and its four test calls.

##### Full code block

```python
# Challenge 9.1 — Safe Division Calculator
# Your group's code here
```

#### Challenge 9.2 — Custom Exception Hierarchy

##### Why this section exists
Reinforces custom exception classes and inheritance (Section 11 + Section 12) — a base `BankError` with three subclasses, each catchable individually or all together via the shared base class.

##### Line-by-line
- `# Challenge 9.2 — Custom Exception Hierarchy` — header comment.
- `from datetime import datetime` — pre-supplied import, needed to timestamp each custom exception as the spec requires.
- `# Your group's code here` — placeholder for the four exception class definitions and the `process_transaction(balance, amount, locked=False)` function.

##### Full code block

```python
# Challenge 9.2 — Custom Exception Hierarchy
from datetime import datetime
# Your group's code here
```

#### Challenge 9.3 — Robust File Reader

##### Why this section exists
Reinforces combining file I/O (Section 10) with exception handling (Section 11) — catching `FileNotFoundError` and `json.JSONDecodeError` separately around a JSON-reading operation, plus timing the operation.

##### Line-by-line
- `# Challenge 9.3 — Robust File Reader` — header comment.
- `import json, time` — pre-supplied imports: `json` for parsing, `time` for the required duration measurement via `time.time()`.
- `# Your group's code here` followed by `# Hint: use open() to first write a temp file with invalid JSON for testing` — placeholder plus a hint nudging students toward creating their own broken-JSON fixture file before testing `safe_read_json()` against it.

##### Full code block

```python
# Challenge 9.3 — Robust File Reader
import json, time
# Your group's code here
# Hint: use open() to first write a temp file with invalid JSON for testing
```

#### Challenge 9.4 — Input Validation Loop

##### Why this section exists
Reinforces looping until valid input is received (Section 6 + Section 11) using a pre-defined list to simulate user input (since a notebook can't easily do live interactive `input()` prompts in this context), plus a custom `OutOfRangeError`.

##### Line-by-line
- `# Challenge 9.4 — Input Validation Loop` — header comment.
- `# Your group's code here` — placeholder for the custom `OutOfRangeError` class, the `get_valid_integer(prompt, min_val, max_val)` function, and its test run against `["abc", "-5", "200", "42"]`.

##### Full code block

```python
# Challenge 9.4 — Input Validation Loop
# Your group's code here
```

### Topic 10 — Decorators & Generators

#### Challenge 10.1 — Timer & Logger Decorators

##### Why this section exists
Reinforces writing decorators from scratch (Section 14) and, specifically, **stacking** two decorators on one function to observe bottom-up application order.

##### Line-by-line
- `# Challenge 10.1 — Timer & Logger Decorators` — header comment.
- `import time, functools` — pre-supplied imports needed for the timing and `@functools.wraps` metadata-preservation patterns from Section 14.
- `# Your group's code here` — placeholder for the `@timer` and `@logger` decorators and the stacked `slow_sum(numbers)` function.

##### Full code block

```python
# Challenge 10.1 — Timer & Logger Decorators
import time, functools
# Your group's code here
```

#### Challenge 10.2 — Retry Decorator with Backoff

##### Why this section exists
Reinforces the parameterized decorator (decorator factory) pattern from Section 14's `@retry` example — students rebuild essentially the same three-layer nested-function structure independently.

##### Line-by-line
- `# Challenge 10.2 — Retry Decorator with Backoff` — header comment.
- `import time, random, functools` — pre-supplied imports mirroring exactly what Section 14's worked `@retry` example needed.
- `# Your group's code here` — placeholder for the `@retry(max_attempts=3, delay=0.5)` decorator factory and the `flaky_api_call()` test function.

##### Full code block

```python
# Challenge 10.2 — Retry Decorator with Backoff
import time, random, functools
# Your group's code here
```

#### Challenge 10.3 — Infinite Generator Pipeline

##### Why this section exists
Reinforces generator functions and chaining generators together (Section 15) — an infinite `fibonacci()`, a `take(n, iterable)` limiter, and an `only_even(iterable)` filter, composed into a lazy pipeline with zero intermediate lists.

##### Line-by-line
- `# Challenge 10.3 — Infinite Generator Pipeline` — header comment.
- `# Your group's code here` — placeholder for the three generator functions and the chained call that produces the first 10 even Fibonacci numbers.

##### Full code block

```python
# Challenge 10.3 — Infinite Generator Pipeline
# Your group's code here
```

#### Challenge 10.4 — Memoize Decorator from Scratch

##### Why this section exists
Reinforces decorators (Section 14) combined with dict-based caching (Section 16.1/6.4) by having students reimplement the core idea behind `functools.lru_cache` themselves, including exposing cache statistics on the wrapper function.

##### Line-by-line
- `# Challenge 10.4 — Memoize Decorator from Scratch` — header comment.
- `# Your group's code here` — placeholder for the `@memoize` decorator (with an attached `cache_info()` reporting hits/misses/size) and its application to a recursive `fib(n)`.

##### Full code block

```python
# Challenge 10.4 — Memoize Decorator from Scratch
# Your group's code here
```

---

## 19. Bonus Challenge — Mini Data Processing Pipeline

### Why this section exists
The capstone exercise: a single realistic scenario (a list of AI research paper records) that requires combining nearly every topic in the notebook — class design, comprehensions, sorting, a stats function, a generator, custom exceptions, and a decorator — into one working pipeline. Unlike the Topic 1–10 stubs, this cell pre-populates its full dataset since the task is entirely about processing it, not creating it.

### Line-by-line

- `# 🚀 Bonus Challenge — Mini Data Processing Pipeline` — header comment identifying the cell.
- `papers_data = [...]` — the starter dataset: six dicts, each describing one AI research paper with `title`, `authors` (a list), `year`, `citations`, and `open_access` (bool) fields — provided in full so every group works from identical source data.
- `# Your group's code here` — placeholder for the entire 8-part pipeline described in the challenge markdown: a `Paper` class with a `citation_rate` property and `__repr__`/`__lt__`; building `Paper` objects via a list comprehension; filtering/sorting with lambdas; a `paper_stats(papers)` aggregation function; a `papers_by_year(papers, start, end)` generator; a `safe_get_paper(papers, title)` function raising a custom `PaperNotFoundError`; a `@log_call` decorator applied to `paper_stats`; and a final formatted f-string report.

### Full code block

```python
# 🚀 Bonus Challenge — Mini Data Processing Pipeline
papers_data = [
    {"title": "Attention Is All You Need", "authors": ["Vaswani", "Shazeer", "Parmar"], "year": 2017, "citations": 90000, "open_access": True},
    {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "authors": ["Devlin", "Chang"], "year": 2018, "citations": 55000, "open_access": True},
    {"title": "GPT-3", "authors": ["Brown", "Mann", "Ryder"], "year": 2020, "citations": 30000, "open_access": False},
    {"title": "Deep Residual Learning for Image Recognition", "authors": ["He", "Zhang", "Ren"], "year": 2015, "citations": 120000, "open_access": True},
    {"title": "AlphaFold", "authors": ["Jumper", "Evans"], "year": 2021, "citations": 18000, "open_access": False},
    {"title": "Generative Adversarial Nets", "authors": ["Goodfellow", "Pouget-Abadie"], "year": 2014, "citations": 65000, "open_access": True},
]

# Your group's code here
```

### Line-by-line (Traceback reading demo)

### Why this section exists
The final code cell is a standalone teaching demo (not a student exercise) that manufactures a real multi-frame traceback on purpose, so students can practice the "read from the bottom up" technique taught earlier in Section 17.

### Line-by-line

- `import traceback` — the standard library module used to print exception tracebacks programmatically (rather than letting Python print one automatically on an uncaught error).
- `def level_3(): return 1 / 0` — the innermost function, containing the actual bug (division by zero).
- `def level_2(): return level_3()` and `def level_1(): return level_2()` — two wrapper functions that call each other in a chain, purely to build up several stack frames before the real error, simulating a realistic multi-layered call stack (e.g. a request handler calling a service function calling a data-access function).
- `try: level_1() except ZeroDivisionError:` — calls the top of the chain; the exception raised three calls deep in `level_3()` propagates all the way up through `level_2()` and `level_1()` (since neither of them catches it) until it reaches this `except` block.
- `traceback.print_exc()` — prints the full traceback of the exception currently being handled, showing all three stack frames (`level_1` → `level_2` → `level_3`) exactly as Python would print them if the exception had gone completely uncaught.
- The subsequent `print()` statements restate the four-step "how to read a traceback" method from Section 17's introduction: start at the bottom (the actual error), move up one line (where it occurred), work upward through the call stack (how execution got there), and note that the topmost frame is where execution began.

### Full code block

```python
import traceback

def level_3():
    return 1 / 0              # The actual error

def level_2():
    return level_3()          # Calls the buggy function

def level_1():
    return level_2()          # Calls level_2

try:
    level_1()
except ZeroDivisionError:
    print("=" * 60)
    print("HOW TO READ A TRACEBACK:")
    print("=" * 60)
    traceback.print_exc()
    print()
    print("Reading order:")
    print("  1. Start at the BOTTOM — that's the actual error type & message")
    print("  2. The line just above is WHERE the error occurred")
    print("  3. Work UPWARD through the call stack to trace the origin")
    print("  4. The TOPMOST frame is where execution started")
```

---

## Closing reference material

The notebook ends with two markdown-only cells: a condensed "Error → Fix Cheat Sheet" table (the same content already covered per-error in Section 17) and a "Next Steps for AI Engineers" roadmap pointing toward NumPy, Pandas, Matplotlib/Seaborn, Scikit-learn, PyTorch/TensorFlow, Hugging Face Transformers, and FastAPI as the natural next topics after this foundation. Neither cell contains code, so there is nothing further to walk through — this concludes the notebook.
