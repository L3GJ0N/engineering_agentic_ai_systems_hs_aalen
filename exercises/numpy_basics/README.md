# Exercise: NumPy basics

NumPy is the library behind `gpt2.py`: one type, the n-dimensional array, and the functions around it, written in C. This exercise teaches what you need to read and write NumPy code: arrays with a shape and a dtype, indexing, the two products `*` and `@`, broadcasting with `axis` and `keepdims`, and the four shapes `(5,)`, `(5, 1)`, `(1, 5)` and `(1, 1)`, behind which many bugs hide. At the end you time NumPy on your laptop, read the table, and see where NumPy comes from. The exercise [NumPy for the attention lab](../numpy_for_attention/README.md) then applies all of it to GPT-2; do this one first. [Inside NumPy](../numpy_inside/README.md) is an optional deep dive: what an array is in memory, and why NumPy is fast.

You work alone. Six parts, A to F, take about 85 to 100 minutes in all; the five ★ tasks are optional and add about 20 minutes. Finish at home what the session leaves. Everything runs on your laptop's CPU, with small arrays and without any download.

All you change is one file: [my_numpy_basics.py](../../attention_lab/src/eaas_attention_lab/exercises/my_numpy_basics.py) in `attention_lab`, six small functions. Everything else happens in the Python REPL and in the terminal.

## How you work

- **Predict first.** Before you run a line, write down what you expect it to print, in a file or on paper. The surprises are the point.
- **The REPL.** In a terminal in `attention_lab`, `uv run python` starts Python with NumPy; `>>>` is its prompt. Type one line, press Enter, and Python prints the result. `exit()` ends it. Every REPL starts empty: type `import numpy as np` first. Each task gives all the other lines it needs.
- **The checks.** `uv run my-numpy-basics` checks the six functions in `my_numpy_basics.py` and prints one line for each: `True`, `not written yet`, or `False` with a hint at what to change. The checks compare your results with what NumPy computes, or with a plain loop.
- **The timings.** `uv run numpy-timings` computes one product of GPT-2 in nine ways and times each on your laptop; part F reads its table.
- **The boxes.** Each task ends with a closed box, **Check your reasoning**. Open it after you tried. Its REPL sessions are real runs, with NumPy 2.5.4 and Python 3.13. The boxes stay closed only where the sheet is shown as a page: on GitHub, or in VS Code's preview, Ctrl+Shift+V (Cmd+Shift+V on a Mac). In the editor, every box is open.

Write the code yourself: a coding agent or chatbot that writes these few lines takes the point away. Asking one for an explanation is fine.

## Get started

1. Update your copy of the course repository: `git pull`, or download it again. This exercise came on 26 November. If `git pull` stops because you changed a file of an earlier exercise, read [Your changes and git pull](#your-changes-and-git-pull) first.
2. Open the course workspace in VS Code with **File > Open Workspace from File** and `engineering_agentic_ai_systems.code-workspace`. Select **Terminal > New Terminal** and choose **attention_lab**.
3. Run:

   ```console
   uv sync
   uv run my-numpy-basics
   ```

The first run checks six functions, and all six say `not written yet`.

## Part A: arrays, shape and dtype

About 15 minutes.

A NumPy array is an object of the class `numpy.ndarray`. Its numbers sit side by side in one block of memory, all of one type, and a small header says how to read the block:

- **The shape** has one length per axis, as a tuple. Axis 0 runs down the rows, axis 1 across the columns: `(2, 3)` is 2 rows of 3 numbers. A 1-D array of three numbers has the shape `(3,)`: the comma makes it a tuple with one length. `ndim` is the number of axes, `size` the number of all entries.
- **The dtype** is the type of all entries. `int64` for whole numbers and `float64` for decimals take 8 bytes per number; GPT-2's weights are `float32`, 4 bytes per number, half the memory. `a.astype(np.float32)` returns a copy of `a` in another dtype.

Attributes such as `a.shape` are read without parentheses; methods such as `a.sum()` are called with them.

**How to make arrays.** `np.array([[1, 2, 3], [4, 5, 6]])` from a nested list; `np.zeros((2, 3))` and `np.ones((2, 3))`, where the shape is one tuple; `np.arange(6)`, the whole numbers 0 to 5; `np.arange(6).reshape(2, 3)`, the same numbers read as 2 rows of 3, row by row. `np.arange` with whole numbers makes integers, with a decimal point floats, as `np.arange(6.0)` does; `np.zeros` and `np.ones` make `float64` unless you pass `dtype=np.float32`.

**A list is not an array.** For Python lists, `+` joins and `* 2` repeats. For arrays, both work on every entry. Turn a list into an array with `np.array(...)` before you compute with it.

### A1. Shapes and dtypes

Predict what each line prints, then type it into your REPL:

<!-- input: a1 -->
```python
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape, a.ndim, a.size, a.dtype
np.zeros(3).shape
np.zeros((3, 1)).shape
np.zeros((1, 3)).shape
np.arange(6).reshape(3, 2)
np.ones((2, 3), dtype=np.float32).dtype
[1, 2] + [3, 4]
np.array([1, 2]) + np.array([3, 4])
a.shape()
```

<details>
<summary>Check your reasoning</summary>

<!-- session: a1 setup=np -->
```pycon
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> a.shape, a.ndim, a.size, a.dtype
((2, 3), 2, 6, dtype('int64'))
>>> np.zeros(3).shape
(3,)
>>> np.zeros((3, 1)).shape
(3, 1)
>>> np.zeros((1, 3)).shape
(1, 3)
>>> np.arange(6).reshape(3, 2)
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> np.ones((2, 3), dtype=np.float32).dtype
dtype('float32')
>>> [1, 2] + [3, 4]
[1, 2, 3, 4]
>>> np.array([1, 2]) + np.array([3, 4])
array([4, 6])
>>> a.shape()
Traceback (most recent call last):
  ...
TypeError: 'tuple' object is not callable
```

`a` has two axes, 2 rows of 3 numbers, 6 entries in all, whole numbers in `int64`. `(3,)`, `(3, 1)` and `(1, 3)` all hold three zeros, but they are three different shapes: one axis of 3 numbers; 3 rows of one number, a column; one row of 3 numbers. Part E comes back to them. `reshape` keeps the numbers and their order and reads them as 3 rows of 2. Lists join, arrays add. `a.shape()` tries to call the tuple `(2, 3)`, and a tuple cannot be called.

</details>

### A2. Random weights: `make_weights()`

Random numbers come from a generator: `rng = np.random.default_rng(42)` makes one from the seed 42, and `rng.normal(0.0, 1.0, size=(2, 3))` draws a 2 × 3 array from a normal distribution with mean 0 and standard deviation 1. `np.array_equal(a, b)` says whether two arrays have the same shape and the same numbers; `a == b` would compare them entry by entry and give an array of `True` and `False`. Predict, then run:

<!-- input: a2 -->
```python
rng = np.random.default_rng(5)
a = rng.normal(0.0, 1.0, size=3)
b = rng.normal(0.0, 1.0, size=3)
np.array_equal(a, b)
c = np.random.default_rng(5).normal(0.0, 1.0, size=3)
np.array_equal(a, c)
a.dtype
```

Then write `make_weights(seed, rows, columns)` in `my_numpy_basics.py`: a weight matrix [rows, columns] in float32, drawn from a normal distribution with mean 0 and standard deviation 0.02, the spread that GPT-2's configuration gives for its initial weights (`"initializer_range": 0.02`). The same seed must give the same matrix, whatever ran before, and another seed another one. Run `uv run my-numpy-basics` until `make_weights` says `True`.

<details>
<summary>Check your reasoning</summary>

<!-- session: a2 setup=np -->
```pycon
>>> rng = np.random.default_rng(5)
>>> a = rng.normal(0.0, 1.0, size=3)
>>> b = rng.normal(0.0, 1.0, size=3)
>>> np.array_equal(a, b)
False
>>> c = np.random.default_rng(5).normal(0.0, 1.0, size=3)
>>> np.array_equal(a, c)
True
>>> a.dtype
dtype('float64')
```

A generator draws a fixed sequence from its seed, and every call continues it: `b` differs from `a`. A new generator with the same seed starts the sequence again: `c` equals `a`. So `make_weights` makes its generator itself, from `seed`, inside the function: a generator made once outside would continue its sequence, and the third call of the check, seed 0 again, would get other numbers than the first. The second argument of `normal` is the standard deviation, 0.02, not the variance, 0.02² = 0.0004. The draws are float64; the matrix becomes float32 with `astype`. The check also calls `make_weights(0, 3, 5)` and expects 3 rows of 5 numbers.

</details>

### ★ One tuple

Predict, then run:

<!-- input: astar -->
```python
np.zeros(2, 3)
```

<details>
<summary>Check your reasoning</summary>

<!-- session: astar setup=np -->
```pycon
>>> np.zeros(2, 3)
Traceback (most recent call last):
  ...
TypeError: Cannot interpret '3' as a data type
```

The second argument of `np.zeros` is the dtype, so NumPy tries to read 3 as a type. The shape must be one tuple: `np.zeros((2, 3))`. `reshape(2, 3)` is the exception: it accepts the lengths one by one.

</details>

## Part B: indexing

About 10 minutes.

- **Index and slice** work as for lists: `a[0]` is the first entry, `a[-1]` the last, `a[2:5]` the entries 2, 3 and 4 (the stop is not included), `a[::2]` every second one.
- **Two axes:** `m[i, j]` is row i, column j. `m[i]`, the same as `m[i, :]`, is row i: in GPT-2, the vector of token i. `m[:, j]` is column j: number j of every token.
- **A list of indices** picks those rows, in that order and as often as listed: `wte[token_ids]` turns the token IDs of a prompt into their vectors, one row per token. This is the embedding lookup of every transformer.

An index removes its axis; a slice keeps it, even a slice of length one.

### B1. Index or slice?

Predict the value or the shape, then run:

<!-- input: b1 -->
```python
m = np.arange(12).reshape(3, 4)
m[1, 2]
m[1]
m[1].shape
m[1:2].shape
m[:, 2]
m[:, 2].shape
m[[2, 0, 2]]
```

<details>
<summary>Check your reasoning</summary>

<!-- session: b1 setup=np -->
```pycon
>>> m = np.arange(12).reshape(3, 4)
>>> m[1, 2]
np.int64(6)
>>> m[1]
array([4, 5, 6, 7])
>>> m[1].shape
(4,)
>>> m[1:2].shape
(1, 4)
>>> m[:, 2]
array([ 2,  6, 10])
>>> m[:, 2].shape
(3,)
>>> m[[2, 0, 2]]
array([[ 8,  9, 10, 11],
       [ 0,  1,  2,  3],
       [ 8,  9, 10, 11]])
```

`m[1, 2]` is one number. The REPL shows NumPy's own numbers with their type, `np.int64(6)`; `print` shows `6`. `m[1]` is row 1, of shape `(4,)`: the index removed axis 0. The slice `m[1:2]` keeps axis 0 with length 1: `(1, 4)`, a matrix of one row. `m[:, 2]` is column 2, but it comes out as a 1-D array, `(3,)`: the index removed axis 1, and nothing in the result says that it was a column. `m[[2, 0, 2]]` picks rows 2, 0 and 2 again, a matrix of 3 rows.

</details>

### B2. A column that stays a column: `column()`

Write `column(m, j)`: column j of the matrix `m`, as a column, of shape (rows, 1). B1 kept an axis for a row; do the same for a column. Run `uv run my-numpy-basics` until `column` says `True`.

<details>
<summary>Check your reasoning</summary>

A slice of length one along axis 1 keeps the axis, and so does a list that holds the one index; `reshape` to (rows, 1) works as well. A plain index, `m[:, j]`, gives `(rows,)`. Why would you want the column? It broadcasts against the matrix, row by row: `m - column(m, 0)` subtracts from each row its own entry in column 0. Part D does the same with `keepdims`.

</details>

## Part C: element-wise or matrix product

About 10 to 15 minutes.

- **Element-wise:** `+`, `-`, `*`, `/`, `**` and functions such as `np.exp` and `np.sqrt` work on every entry on its own. The shapes must fit: equal, or broadcast (part D). A single number works with every entry, as in `2 * x`.
- **The matrix product `@`:** `a @ b` combines the rows of `a` with the columns of `b`: (m, k) @ (k, n) gives (m, n), and the inner sizes must be equal. Entry (i, j) is the dot product of row i of `a` and column j of `b`. In GPT-2, `x @ w` sends the row of every token through the same weights.
- **The transpose:** `.T` swaps rows and columns: (4, 3) becomes (3, 4).

Using `*` where `@` was meant is a classic bug in NumPy code: when the shapes happen to fit, it raises no error.

### C1. `*` or `@`?

`x` holds 4 tokens with 3 numbers each, `w` a weight matrix of 3 rows and 2 columns. Predict each shape, or whether a line raises an error, then run:

<!-- input: c1 -->
```python
x = np.arange(12.0).reshape(4, 3)
w = np.ones((3, 2))
(x @ w).shape
(x * x).shape
(x.T @ x).shape
(x @ x.T).shape
x @ w.T
x * w
```

After the run: in the last line of the error of `x @ w.T`, which two sizes does NumPy compare?

<details>
<summary>Check your reasoning</summary>

<!-- session: c1 setup=np -->
```pycon
>>> x = np.arange(12.0).reshape(4, 3)
>>> w = np.ones((3, 2))
>>> (x @ w).shape
(4, 2)
>>> (x * x).shape
(4, 3)
>>> (x.T @ x).shape
(3, 3)
>>> (x @ x.T).shape
(4, 4)
>>> x @ w.T
Traceback (most recent call last):
  ...
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
>>> x * w
Traceback (most recent call last):
  ...
ValueError: operands could not be broadcast together with shapes (4,3) (3,2)
```

`x @ w`: 4 rows, 2 columns, two new numbers for every token. `x * x` keeps the shape and squares every entry. `x.T @ x` is (3, 4) @ (4, 3), so (3, 3); `x @ x.T` is (4, 4), one number for each pair of tokens, as the scores of attention are. `x @ w.T` puts the 3 columns of `x` against the 2 rows of `w.T`: operand 1, the right factor counted from 0, has 2 where 3 were needed. `x * w` is element-wise and needs shapes that broadcast; (4, 3) and (3, 2) do not.

</details>

### C2. The shape of a product: `result_shape()`

Write `result_shape(a_shape, b_shape)`, in plain Python: the shape of `a @ b` for two matrices of these shapes, as a tuple. When the inner sizes differ, raise a `ValueError`, as `@` does. Run `uv run my-numpy-basics` until `result_shape` says `True`.

<details>
<summary>Check your reasoning</summary>

The inner sizes meet and vanish: the rows of `a` and the columns of `b` remain. The check compares your shapes with those of NumPy's own products. Two of them come back in part E: (1, 5) @ (5, 1) is (1, 1), one number that is still a matrix, and (4, 1) @ (1, 4) is (4, 4), every number of the column times every number of the row. `raise ValueError("...")` stops the function with an error, as NumPy does for `x @ w.T`.

</details>

### ★ A 1-D array has no direction

Mathematics writes vectors as columns or rows. NumPy's 1-D array is neither. Predict, then run:

<!-- input: cstar -->
```python
v = np.array([1.0, 2.0, 3.0])
m3 = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
v @ m3
m3 @ v
v.T.shape
v @ v
```

<details>
<summary>Check your reasoning</summary>

<!-- session: cstar setup=np -->
```pycon
>>> v = np.array([1.0, 2.0, 3.0])
>>> m3 = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
>>> v @ m3
array([3., 2., 6.])
>>> m3 @ v
array([1., 3., 6.])
>>> v.T.shape
(3,)
>>> v @ v
np.float64(14.0)
```

`v @ m3` combines `v` with the columns of `m3`, `m3 @ v` with its rows: both work, both give a 1-D array, and they differ. The position in `@` decides, not a direction of `v`. `.T` changes nothing for a 1-D array. `v @ v` is the dot product, one number: no shape to care about, which part E comes back to.

</details>

## Part D: broadcasting, axis and keepdims

About 15 to 20 minutes.

- **Broadcasting:** for an element-wise operation, NumPy lines up the two shapes from the right. Two lengths fit if they are equal or if one of them is 1, and a missing axis counts as 1. A length of 1 is stretched to the other length, without copying numbers. `(4, 3) + (3,)` adds the same row to every row, as a bias is added to every token; `(4, 3) * (4, 1)` multiplies every row by its own number. `(4, 3) + (4,)` does not fit: lined up from the right, 3 meets 4.
- **axis:** a reduction such as `sum`, `mean`, `max` or `var` (the variance, the mean squared distance from the mean) removes the axis that you give it. `a.sum(axis=1)` adds along each row and gives one sum per row; `axis=0` gives one result per column; `axis=-1` is the last axis; without `axis` you get one number for the whole array.
- **keepdims:** `keepdims=True` keeps the removed axis, with length 1. The maxima of the rows of a (4, 3) are then (4, 1), a column, instead of (4,), and they broadcast back against (4, 3): one number for each row.

### D1. Which axis disappears?

Predict the values and shapes, then run:

<!-- input: d1 -->
```python
a = np.array([[1.0, 5.0], [3.0, 2.0], [4.0, 4.0]])
a.sum(axis=1)
a.sum(axis=0)
a.sum()
a.max(axis=1).shape
a.max(axis=1, keepdims=True).shape
a - a.max(axis=1, keepdims=True)
a - a.max(axis=1)
a - a.max(axis=0)
```

<details>
<summary>Check your reasoning</summary>

<!-- session: d1 setup=np -->
```pycon
>>> a = np.array([[1.0, 5.0], [3.0, 2.0], [4.0, 4.0]])
>>> a.sum(axis=1)
array([6., 5., 8.])
>>> a.sum(axis=0)
array([ 8., 11.])
>>> a.sum()
np.float64(19.0)
>>> a.max(axis=1).shape
(3,)
>>> a.max(axis=1, keepdims=True).shape
(3, 1)
>>> a - a.max(axis=1, keepdims=True)
array([[-4.,  0.],
       [ 0., -1.],
       [ 0.,  0.]])
>>> a - a.max(axis=1)
Traceback (most recent call last):
  ...
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
>>> a - a.max(axis=0)
array([[-3.,  0.],
       [-1., -3.],
       [ 0., -1.]])
```

`a` has 3 rows of 2 numbers. `axis=1` adds along each row: 6, 5 and 8, shape `(3,)`. `axis=0` gives one sum per column, 8 and 11, and no axis one number for all, 19. The maxima of the rows, 5, 3 and 4, are `(3,)`, with `keepdims` `(3, 1)`. As a column, they subtract from each row its own maximum, so the largest entry of every row becomes 0: the softmax in `gpt2.py` starts this way. Without `keepdims`, (3, 2) meets (3,), and lined up from the right, 2 meets 3: an error. `a - a.max(axis=0)` works without `keepdims`: the maxima of the columns, `(2,)`, line up with the last axis, so each column loses its own maximum. Results per column fit without `keepdims`; results per row need it.

</details>

### D2. Rows that sum to 1: `normalize_rows()`

Write `normalize_rows(x)`: every row of `x` divided by its own sum, so that each row sums to 1, as each row of attention weights does. No loop. The check uses a square matrix, 3 × 3, where a missing `keepdims` raises no error. Run `uv run my-numpy-basics` until `normalize_rows` says `True`.

<details>
<summary>Check your reasoning</summary>

To divide each row by its own sum, the sums must be a column, (3, 1): sum along axis 1, or -1, with `keepdims=True`. Without `keepdims`, the sums are `(3,)`, and broadcasting lines them up with the last axis, as a row: entry j of every row is divided by the sum of row j. That is wrong, and no error says so, because the matrix is square; a (4, 3) matrix would have raised one. A line without an error may still have broadcast in a way you did not mean: check the shapes.

</details>

### D3. Columns with mean 0: `standardize_columns()`

Write `standardize_columns(x, eps)`: every column of `x` with mean 0 and standard deviation 1. Subtract each column's mean, then divide by the square root of the column's variance with `eps` added under the root, `np.sqrt(variance + eps)`. No loop. Do you need `keepdims` here? Why `eps`? The check includes a column whose numbers are all equal, and passes `eps = 0.01`. Run `uv run my-numpy-basics` until `standardize_columns` says `True`.

<details>
<summary>Check your reasoning</summary>

The means and variances of the columns remove axis 0 and have the shape (columns,), which lines up with the last axis of (rows, columns): every row gets the same row of means subtracted. So the columns need no `keepdims`; with `keepdims=True`, (1, columns) works as well. A column whose numbers are all equal has the variance 0, and 0 divided by 0 is `nan`, not a number; `eps` under the root keeps the divisor above 0. The check passes 0.01, enough to tell `np.sqrt(variance + eps)` from `np.sqrt(variance) + eps`. The layer norm in `gpt2.py` does the same for each row, along the last axis, and there `keepdims=True` is needed.

</details>

### ★ Which shapes fit?

`np.broadcast_shapes` gives the shape that two shapes broadcast to, or an error. Predict each line, then run:

<!-- input: dstar -->
```python
np.broadcast_shapes((4, 3), (3,))
np.broadcast_shapes((4, 3), (4, 1))
np.broadcast_shapes((5, 1), (1, 5))
np.broadcast_shapes((12, 9, 9), (9, 9))
np.broadcast_shapes((4, 3), (4,))
```

<details>
<summary>Check your reasoning</summary>

<!-- session: dstar setup=np -->
```pycon
>>> np.broadcast_shapes((4, 3), (3,))
(4, 3)
>>> np.broadcast_shapes((4, 3), (4, 1))
(4, 3)
>>> np.broadcast_shapes((5, 1), (1, 5))
(5, 5)
>>> np.broadcast_shapes((12, 9, 9), (9, 9))
(12, 9, 9)
>>> np.broadcast_shapes((4, 3), (4,))
Traceback (most recent call last):
  ...
ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (4, 3) and arg 1 with shape (4,).
```

From the right: 3 meets 3, and the missing axis counts as 1. 3 meets 1, then 4 meets 4. A column and a row stretch each other to a table, 5 × 5. The scores of twelve heads for 9 tokens, (12, 9, 9), take one mask of (9, 9) for all heads at once, as `gpt2.py` adds its causal mask. The last pair puts 3 against 4.

</details>

## Part E: (5,), (5, 1), (1, 5) and (1, 1)

About 20 minutes.

Five numbers can come in three shapes:

- `(5,)`: a 1-D array, five numbers along one axis, neither a row nor a column.
- `(5, 1)`: a matrix of 5 rows with one number each, a column.
- `(1, 5)`: a matrix of one row of 5 numbers, a row.

Why would you want `(5,)`? Most results are a list of numbers: one score per token, one loss per example, the probabilities of the next token. A 1-D array is indexed with one number, `p[3]`; it sums and sorts as a list would; it fits on either side of `@`; and a reduction without `keepdims` returns one. `(5, 1)` and `(1, 5)` are for broadcasting against a matrix, on purpose: `keepdims` or `reshape` makes them. So keep a list of numbers 1-D, go 2-D on purpose, and come back with `.ravel()`, which turns any array into a 1-D array of all its numbers, row by row (`.reshape(-1)` does the same). `.item()` turns an array that holds one number into a Python number.

### E1. Three shapes of five numbers

Predict, then run:

<!-- input: e1 -->
```python
v = np.arange(5.0)
col = v.reshape(5, 1)
row = v.reshape(1, 5)
print(col)
print(row)
(col + row).shape
(col + v).shape
(v + v).shape
v.T.shape, col.T.shape
row.ravel().shape
```

<details>
<summary>Check your reasoning</summary>

<!-- session: e1 setup=np -->
```pycon
>>> v = np.arange(5.0)
>>> col = v.reshape(5, 1)
>>> row = v.reshape(1, 5)
>>> print(col)
[[0.]
 [1.]
 [2.]
 [3.]
 [4.]]
>>> print(row)
[[0. 1. 2. 3. 4.]]
>>> (col + row).shape
(5, 5)
>>> (col + v).shape
(5, 5)
>>> (v + v).shape
(5,)
>>> v.T.shape, col.T.shape
((5,), (1, 5))
>>> row.ravel().shape
(5,)
```

The column prints as 5 rows of one number, the row as one row in two pairs of brackets: the number of brackets at the start is the number of axes. `col + row` stretches both to a table, 5 × 5: every number of the column plus every number of the row. `col + v` gives (5, 5) as well, because `v`, lined up from the right, acts as a row. `v + v` stays `(5,)`. `.T` leaves a 1-D array as it is and turns the column into a row. `.ravel()` brings the row back to `(5,)`.

</details>

### E2. The silent table

The mean squared error of a prediction, the mean of `(y_true - y_pred) ** 2` over all examples, is the loss of many models. Here the predictions come as a column, as a model with one output per row of its input returns them. Predict, then run:

<!-- input: e2 -->
```python
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = (y_true + 0.1).reshape(4, 1)
diff = y_true - y_pred
diff.shape
(diff ** 2).mean()
```

Every prediction is off by 0.1, so the error should be 0.1² = 0.01. Why is it not, and why did NumPy not complain?

<details>
<summary>Check your reasoning</summary>

<!-- session: e2 setup=np -->
```pycon
>>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
>>> y_pred = (y_true + 0.1).reshape(4, 1)
>>> diff = y_true - y_pred
>>> diff.shape
(4, 4)
>>> (diff ** 2).mean()
np.float64(2.51)
```

`(4,)` minus `(4, 1)` broadcasts to (4, 4), as `col + v` did in E1: every true value minus every prediction, 16 differences instead of 4, and their mean square is 2.51. No error: the shapes fit, only not in the way that was meant. A model trained on this loss runs without an error and learns the wrong thing. Bring both arrays to the same shape before you subtract them.

</details>

### E3. A (1, 1) result is not a number

One example with 3 numbers, as a row, times weights as a column: a model with one output. Predict, then run:

<!-- input: e3 -->
```python
x = np.array([[1.0, 2.0, 3.0]])
w = np.array([[0.5], [0.25], [0.125]])
r = x @ w
r.shape
f"{r:.2f}"
round(r, 2)
r.item()
f"{r.item():.2f}"
r.squeeze().shape
float(r.squeeze())
x[0] @ w[:, 0]
```

<details>
<summary>Check your reasoning</summary>

<!-- session: e3 setup=np -->
```pycon
>>> x = np.array([[1.0, 2.0, 3.0]])
>>> w = np.array([[0.5], [0.25], [0.125]])
>>> r = x @ w
>>> r.shape
(1, 1)
>>> f"{r:.2f}"
Traceback (most recent call last):
  ...
TypeError: unsupported format string passed to numpy.ndarray.__format__
>>> round(r, 2)
Traceback (most recent call last):
  ...
TypeError: type numpy.ndarray doesn't define __round__ method
>>> r.item()
1.375
>>> f"{r.item():.2f}"
'1.38'
>>> r.squeeze().shape
()
>>> float(r.squeeze())
1.375
>>> x[0] @ w[:, 0]
np.float64(1.375)
```

(1, 3) @ (3, 1) is (1, 1): one number, but still a matrix with two axes. Python's own tools want a number. Format strings, `round()` and `json.dumps()` have never taken an array; `float()` and `math.exp()` did, silently up to NumPy 1.24 and with a `DeprecationWarning` from 1.25 to 2.3, and raise a `TypeError` since NumPy 2.4 (we tried all five with NumPy 1.24.4, 2.3.5, 2.4.6 and 2.5.4 on 26 November 2026). Code and answers on the web that call `float()` on such a result may come from those years. `.item()` returns a Python float. `squeeze()` removes every axis of length 1 and leaves the shape `()`, a 0-dimensional array, which `float()` accepts. `r[0, 0]` would give NumPy's own number, `np.float64`, which works wherever a Python float does: NumPy made it a subclass of `float` (`np.float32` is not one, and `json.dumps` rejects it). Row 0 of `x` and column 0 of `w` are 1-D, and with two 1-D arrays `@` gives the number at once: one more reason to keep vectors 1-D.

</details>

### E4. One number from any two shapes: `mse()`

Write `mse(y_true, y_pred)`: the mean squared error of two arrays that hold the same n numbers, each as (n,), (n, 1) or (1, n). Return one number, not an array. The check tries four combinations of shapes, among them the one of E2, with five numbers. Run `uv run my-numpy-basics` until `mse` says `True`.

<details>
<summary>Check your reasoning</summary>

Bring both arrays to one shape first, the simplest one being 1-D; then subtract, square and take the mean. The mean of a whole array, without `axis`, is already one number, an `np.float64`; `float()` or `.item()` makes it a Python float. A version that brought both to (1, n) and kept the axes with `keepdims` would return (1, 1) again: the check says so.

</details>

### ★ Down to shape ()

Predict, then run:

<!-- input: estar -->
```python
z = np.ones((3, 4))
z.sum(keepdims=True).shape
z.sum().shape, z.sum().ndim
z[0:1, 0:1].shape
z[0, 0].shape
float(z.sum())
```

<details>
<summary>Check your reasoning</summary>

<!-- session: estar setup=np -->
```pycon
>>> z = np.ones((3, 4))
>>> z.sum(keepdims=True).shape
(1, 1)
>>> z.sum().shape, z.sum().ndim
((), 0)
>>> z[0:1, 0:1].shape
(1, 1)
>>> z[0, 0].shape
()
>>> float(z.sum())
12.0
```

A sum over all axes with `keepdims=True` keeps both axes: (1, 1). Without it, the sum has the shape `()`, no axis at all: a 0-dimensional value, and that is what `float()` accepts. For an array with axes, `float()` says so itself: only 0-dimensional arrays can be converted to Python scalars (first aid). Slices of length one keep both axes; two indices remove both.

</details>

## Part F: fast and slow, and where NumPy comes from

About 15 to 20 minutes.

Why is NumPy fast? A loop in Python runs about ten bytecode instructions of the interpreter for every multiplication, checks the types each time and makes a new float object for every result. One call of NumPy runs the whole loop in C, over the raw numbers of one dtype. And `@` on float32 and float64 numbers goes to BLAS, a library tuned for matrix products: OpenBLAS in the NumPy that `uv sync` installs, Apple's Accelerate on Macs with macOS 14 or newer.

### F1. Predict, then measure

`uv run numpy-timings` computes the scores of one attention head, S = Q Kᵀ for T = 1024 tokens with d_h = 64 numbers each, in nine ways: from three loops in plain Python to one `@` on float32, the way `gpt2.py` does it. Read the nine ways in [numpy_timings.py, lines 102 to 252](../../attention_lab/src/eaas_attention_lab/commands/numpy_timings.py#L102-L252): each is a function of a few lines. Slide 18 showed the table of our laptop. Before you run it, predict which rows will come out differently on yours, and why: another processor, another BLAS, more or fewer cores, a laptop on battery. Then run it in the terminal, not in the REPL; it takes about 15 seconds:

```console
uv run numpy-timings
```

<details>
<summary>Check your reasoning</summary>

Our run, on a laptop with an Intel Core i7-12700H (14 cores, 20 threads), Windows 11, on 26 November 2026; slide 18 shows the same table:

<!-- timings: e1 -->
```text
Way                                  Steps in Python       Time   vs. one @  Same
Python: three loops over lists            67,108,864     1.99 s     2,290 x  True
NumPy, one dot product per pair            1,048,576     1.03 s     1,192 x  True
NumPy, one row per step                        1,024     4.8 ms       5.5 x  True
NumPy, one outer product per step                 64    75.6 ms        87 x  True
NumPy, broadcasting, then sum                      1    82.3 ms        95 x  True
np.einsum                                          1    6.21 ms       7.2 x  True
one @ on int32 (no BLAS)                           1    28.3 ms        33 x  True
one @ on float64 (BLAS)                            1    1.59 ms       1.8 x  True
one @ on float32 (BLAS)                            1   0.868 ms       1.0 x  True
```

Your numbers differ with your laptop, its power settings and what else runs; the order of magnitude stays. "Steps in Python" counts how often a line of Python runs in the innermost loop: 67 million in plain Python, a million calls of NumPy for the pairs, 1,024 rows, 64 outer products, and one call for the rest. "vs. one @" divides each time by the time of one `@` on float32. "Same" compares each result with the exact product. The command then times more: BLAS threads, twelve heads at once, `np.exp`, and the memory of lists and arrays; [Inside NumPy](../numpy_inside/README.md), part E, reads the threads and the heads.

</details>

### F2. Read the table

Answer from the table of F1, yours and ours; the ratios in the questions are ours. Then open the box:

1. How many times as long as one `@` on float32 did the three loops in plain Python take on your laptop?
2. "One dot product per pair" lets NumPy compute each of the million dot products in C. Why is it only about twice as fast as plain Python?
3. "One row per step" calls NumPy only 1,024 times. Why does that bring it within about 5 times one `@`?
4. Why does one `@` on int32 numbers take about 30 times as long as on float32, and on float64 about twice as long?

<details>
<summary>Check your reasoning</summary>

1. On ours, about 2,300 times as long. Yours differs with the processor and the BLAS, but stays in the thousands: a loop in Python is the wrong tool for numbers.
2. Every call of NumPy costs time of its own, about as much as a short loop in Python. Here a call does only 64 multiplications, so a million calls saved only half of the time. A call of NumPy pays off when it gets a lot of work.
3. Each call computes a whole row of S, 1,024 dot products, 65,536 multiplications: the cost of the 1,024 calls hardly counts any more.
4. NumPy hands only float32 and float64 numbers (and complex ones) to BLAS; for whole numbers it runs a plain loop in C of its own. A float64 number takes 8 bytes instead of 4: twice the memory to move, and one vector instruction handles half as many of them at once.

</details>

### F3. Where NumPy comes from

NumPy is open source, under a BSD license, and its code can be read: [github.com/numpy/numpy](https://github.com/numpy/numpy/tree/v2.5.4), pinned here to version 2.5.4, which `uv sync` installs for Python 3.12 and newer. The array and its loops are written in C, in [numpy/_core/src](https://github.com/numpy/numpy/tree/v2.5.4/numpy/_core/src): `multiarray` holds the array type, `umath` the ufuncs, the functions that run on every entry such as `np.exp`, and the loops behind `@`.

Its history: in 1995 Jim Hugunin, a graduate student at MIT, wrote most of Numeric, an array type for Python. Around 2001 the Space Telescope Science Institute wrote Numarray for large images of the Hubble telescope; it was slow for small arrays, and the community split between the two packages. In 2005 Travis Oliphant merged them into NumPy; NumPy 1.0 came in October 2006, and NumPy 2.0 in June 2024. pandas, SciPy and scikit-learn build on it, and the tensors of PyTorch have a `shape` and a `dtype` too.

Run `np.show_config()` in your REPL and look for `blas`: which library does your NumPy use? Then open [ndarraytypes.h, lines 790 to 839](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/include/numpy/ndarraytypes.h#L790-L839), the C struct that every array is: which of its fields `nd`, `dimensions` and `descr` match the attributes you used in part A, and what do `data` and `strides` hold?

<details>
<summary>Check your reasoning</summary>

Ours, NumPy 2.5.4 on Windows, on 26 November 2026: BLAS `scipy-openblas` 0.3.34.106.0, an OpenBLAS that NumPy's packages bring along (NumPy 2.4.6, for Python 3.11, brings 0.3.31.188.0). On a Mac with macOS 14 or newer it says `accelerate`. `data` is the address where the block of numbers starts, `nd` is `ndim`, `dimensions` the shape, `descr` the dtype. `strides` says how many bytes one step along each axis jumps in the block: that is how one block can be read as 3 rows of 2 or as 2 rows of 3, and how `.T` swaps rows and columns without moving a number. [Inside NumPy](../numpy_inside/README.md) starts there.

</details>

### ★ Other sizes

Predict: with 256 tokens instead of 1,024, does plain Python fall further behind one `@`, or less far? And one row per step? Then run `uv run numpy-timings --tokens 256`.

<details>
<summary>Check your reasoning</summary>

Our run on 26 November 2026, the nine ways only:

<!-- timings: size256 -->
```text
The scores of one head, S = Q K^T: Q and K are [256, 64], float32, S is [256, 256]
Way                                  Steps in Python       Time   vs. one @  Same
Python: three loops over lists             4,194,304     123 ms     1,871 x  True
NumPy, one dot product per pair               65,536    60.9 ms       930 x  True
NumPy, one row per step                          256   0.498 ms       7.6 x  True
NumPy, one outer product per step                 64    1.51 ms        23 x  True
NumPy, broadcasting, then sum                      1    4.67 ms        71 x  True
np.einsum                                          1   0.335 ms       5.1 x  True
one @ on int32 (no BLAS)                           1    1.52 ms        23 x  True
one @ on float64 (BLAS)                            1    0.11 ms       1.7 x  True
one @ on float32 (BLAS)                            1    65.5 µs       1.0 x  True
```

Plain Python and one `@` both do T × T × d_h multiplications, so their ratio stays in the thousands, a little lower than with 1,024 tokens: one `@` has 16 times less work, and its fixed cost weighs more. A smaller array does not rescue a loop in Python. One row per step falls further behind, from 5.5 to 7.6 times one `@`: each of its 256 calls gets a row of 256 dot products instead of 1,024, so the cost of a call counts more, as in F2, question 3.

</details>

## First aid

| You see | What helps |
| --- | --- |
| `NameError: name 'np' is not defined` | Type `import numpy as np` first: every new REPL starts empty. |
| `ModuleNotFoundError: No module named 'numpy'` | Start the REPL with `uv run python` in the `attention_lab` folder, not with `python`. There, `np.__version__` says 2.5.4 (2.4.6 with Python 3.11). |
| `TypeError: 'tuple' object is not callable` | An attribute such as `a.shape` takes no parentheses. |
| `TypeError: Cannot interpret '3' as a data type` | A shape is one tuple: `np.zeros((2, 3))`. |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, ...` | Read the end of the line, `(size ... is different from ...)`: the inner sizes of `@` differ. Print both shapes; maybe a `.T` is missing, or the order is wrong. |
| `ValueError: operands could not be broadcast together with shapes ...` | Compare the two shapes from the right. A result per row needs `keepdims=True`. |
| `TypeError: only 0-dimensional arrays can be converted to Python scalars` | An array stands where Python wants a number. If it holds one number, such as a (1, 1) result, use `.item()` (E3); if it holds several, a reduction or an index is missing. |
| `ValueError: The truth value of an array with more than one element is ambiguous. ...` | `if`, `and` and `or` need one truth value: compare whole arrays with `np.array_equal(a, b)`, or ask `(a == b).all()` or `.any()`. |
| `TypeError: unsupported format string passed to numpy.ndarray.__format__` | The same in a format string: format `r.item()`, not `r`. |
| `not written yet` | The function still raises `NotImplementedError`: write it in place of that line. |
| Your function works in the file, but the REPL still raises `NotImplementedError` | A running REPL keeps the `my_numpy_basics.py` it imported: `exit()` and start a new REPL. |

## Your changes and git pull

Your work stays in your copy of the repository. If a later update changes `my_numpy_basics.py` too, `git pull` stops instead of overwriting your work. Before you pull, either keep your solution on a branch of your own (`git switch -c my-numpy-basics`, `git commit -am "NumPy basics"`, then `git switch main`), or undo your changes to `my_numpy_basics.py` with **Source Control > Discard Changes**.
