# Exercise: NumPy for the attention lab

`attention_lab` computes GPT-2 in about 100 lines of NumPy, in [gpt2.py](../../attention_lab/src/eaas_attention_lab/gpt2.py). This exercise teaches the NumPy those lines use, on the lab's own arrays. Afterwards you can read every line of `gpt2.py` and of the three commands, and solve tasks 3, 4 and 6 of [one_head](../one_head/README.md) on your own.

Do [Inside NumPy](../numpy_inside/README.md) first: it shows what an array is and why NumPy is fast; this exercise applies it to GPT-2.

You work alone. Parts A to D follow the four rounds of Part II of the session on 26 November, slides 24 to 35. Most need 50 to 70 minutes: finish them at home. Part E, reading `gpt2.py`, takes about 20 to 30 minutes more. Tasks marked ★ are optional. Everything runs on the tiny GPT-2 with random weights, on your laptop's CPU, without any download.

All you change is one file: [my_numpy.py](../../attention_lab/src/eaas_attention_lab/exercises/my_numpy.py) in `attention_lab`. Everything else happens in the Python REPL.

## How you work

- **Predict first.** Before you run a line, write down what you expect it to print, in a file or on paper. The surprises are the point.
- **The REPL.** In a terminal in `attention_lab`, `uv run python` starts Python with NumPy and the lab; `>>>` is its prompt. Type one line, press Enter, and Python prints the result. `exit()` ends it. Start every REPL with these four lines:

  <!-- input: start -->
  ```python
  import numpy as np
  from eaas_attention_lab.exercises.my_numpy import tiny_trace
  trace = tiny_trace("the cat sat on it")
  layer = trace.layers[5]
  ```

  `trace` holds everything the tiny GPT-2 computed for "the cat sat on it" with `<|endoftext|>` in front, 6 tokens; `layer` is block 5 of its 12 blocks. The tasks of parts A and B continue in this REPL.
- **The checks.** `uv run my-numpy` checks the functions in `my_numpy.py` and prints one line for each: `True`, `not written yet`, or `False` with a hint at what to change.
- **The boxes.** Each task ends with a closed box, **Check your reasoning**. Open it after you tried. Its REPL sessions are real runs. Your REPL writes in front of a warning where it came from, such as `<python-input-3>:1:`; the boxes leave that out. The boxes stay closed only where the sheet is shown as a page: on GitHub, or in VS Code's preview, Ctrl+Shift+V (Cmd+Shift+V on a Mac). In the editor, every box is open.

Write the code yourself: a coding agent or chatbot that writes these few lines takes the point away. Asking one for an explanation is fine.

<details>
<summary>The four start lines, as the REPL shows them</summary>

<!-- session: start -->
```pycon
>>> import numpy as np
>>> from eaas_attention_lab.exercises.my_numpy import tiny_trace
>>> trace = tiny_trace("the cat sat on it")
>>> layer = trace.layers[5]
```

They print nothing: an assignment shows no result. Type `layer` alone to see what it is, a long printout.

</details>

## Get started

1. Update your copy of the course repository: `git pull`, or download it again. This exercise came on 26 November. If `git pull` stops because you changed `my_head.py`, read [Your changes and git pull](#your-changes-and-git-pull) first.
2. Open the course workspace in VS Code with **File > Open Workspace from File** and `engineering_agentic_ai_systems.code-workspace`. Select **Terminal > New Terminal** and choose **attention_lab**.
3. Run:

   ```console
   uv sync
   uv run my-numpy
   ```

The first run checks eight functions. Seven say `not written yet`: six you write in parts B, C and D, the seventh, `merge`, is the ★ task of part D. The first one, `scaled_scores`, says `False`: that is task A3.

## Part A: one block of numbers

About 10 to 15 minutes.

<details>
<summary>How to make arrays (to look up)</summary>

<!-- session: howto setup=np -->
```pycon
>>> np.array([[1.0, 2.0], [3.0, 4.0]])
array([[1., 2.],
       [3., 4.]])
>>> np.zeros((2, 3), dtype=np.float32)
array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
>>> np.full((2, 2), -np.inf)
array([[-inf, -inf],
       [-inf, -inf]])
>>> np.arange(6).reshape(2, 3)
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.eye(3, dtype=bool)
array([[ True, False, False],
       [False,  True, False],
       [False, False,  True]])
```

A shape is a tuple: `np.zeros((2, 3))`, with two pairs of parentheses. Without `dtype`, numbers with a decimal point become float64 and whole numbers int64; the weights of GPT-2 are float32.

</details>

### A1. The shapes of a trace

The comments of `gpt2.py`, lines 62 to 90, give every shape in brackets, such as `[T, d]` or `[heads, T, d_h]`: T tokens, d numbers per token, d_h numbers per head, V entries in the vocabulary. In the tiny GPT-2, d = 48, there are 12 heads of d_h = 4, and V = 50. `.ndim` is the number of axes, `.size` the number of numbers. Predict what each line prints, then type it into your REPL:

<!-- input: a1 -->
```python
trace.x0.shape
layer.q.shape
layer.scores.shape
trace.attention.shape
trace.logits.shape
layer.q.dtype
len(layer.q), layer.q.ndim, layer.q.size
```

<details>
<summary>Check your reasoning</summary>

<!-- session: a1 setup=trace -->
```pycon
>>> trace.x0.shape
(6, 48)
>>> layer.q.shape
(12, 6, 4)
>>> layer.scores.shape
(12, 6, 6)
>>> trace.attention.shape
(12, 12, 6, 6)
>>> trace.logits.shape
(6, 50)
>>> layer.q.dtype
dtype('float32')
>>> len(layer.q), layer.q.ndim, layer.q.size
(12, 3, 288)
```

A shape lists the length of every axis, from the first to the last. `layer.q` is a stack: 12 heads, each a 6 × 4 matrix. `trace.attention` puts the 12 layers in front of that. `len()` counts along the first axis only, `ndim` is the number of axes, and `size` the number of numbers, 12 × 6 × 4 = 288.

</details>

### A2. A head is a window into one block

An array is one block of numbers in memory plus a header: its shape, its dtype and its strides, the number of bytes from one element to the next along each axis. `k` below holds the keys of head 4. When an array only looks into the block of another one, `.base` is that other array. Predict, then run:

<!-- input: a2 -->
```python
k = layer.k[4]
k.shape, k.strides
k.T.shape, k.T.strides
np.shares_memory(k, k.T)
k.base.shape
```

After the run: why is the step from one row of `k` to the next so much larger than a row of 4 numbers of 4 bytes? And why does `k.T` cost nothing?

<details>
<summary>Check your reasoning</summary>

<!-- session: a2 setup=trace -->
```pycon
>>> k = layer.k[4]
>>> k.shape, k.strides
((6, 4), (576, 4))
>>> k.T.shape, k.T.strides
((4, 6), (4, 576))
>>> np.shares_memory(k, k.T)
True
>>> k.base.shape
(6, 144)
```

Along a row, the next number is 4 bytes further: a float32. The next row is 576 bytes further, because `k` is no matrix of its own: it is a window into the block `qkv` of `gpt2.py`, line 159, which `k.base` shows. Each row of `qkv` holds 144 numbers, 144 × 4 = 576 bytes: the queries, keys and values of all 12 heads side by side. `k.T` writes a new header with the two strides swapped. No number moves, so `k.T` shares its memory with `k` and costs nothing, however large `k` is. An array that is only a new header over another array's block, such as `k` and `k.T`, is called a view.

</details>

### A3. Fix the float64 leak

`scaled_scores(q, k)` in `my_numpy.py` computes S = Q Kᵀ / √d_h of one head. The first `uv run my-numpy` says its numbers are right, but its dtype is float64. Find out why: `gpt2.py`, lines 118 and 119, has the answer. Change the formula, not the type of its result (no `.astype`), so that every step stays float32, and run until the check says `True`.

<details>
<summary>Check your reasoning</summary>

`np.sqrt(d_h)` returns a NumPy float64 number, and a NumPy number takes part in choosing the dtype: float32 divided by float64 gives float64. A Python number, such as the result of `math.sqrt(d_h)`, adapts to the array, and the result stays float32. This is NumPy 2's rule for mixing numbers and arrays (NEP 50). It matters because float64 doubles the memory of every array computed from it, and `tests/test_gpt2.py` checks that the forward pass stays float32.

</details>

### ★ Which numbers turn float32 into float64?

Predict the dtype of each result, then check:

<!-- input: astar -->
```python
(k / 2).dtype
(k / 2.0).dtype
(k / np.float64(2)).dtype
(k / np.float32(2)).dtype
(k * np.ones(4)).dtype
```

<details>
<summary>Check your reasoning</summary>

<!-- session: astar setup=k -->
```pycon
>>> (k / 2).dtype
dtype('float32')
>>> (k / 2.0).dtype
dtype('float32')
>>> (k / np.float64(2)).dtype
dtype('float64')
>>> (k / np.float32(2)).dtype
dtype('float32')
>>> (k * np.ones(4)).dtype
dtype('float64')
```

Python numbers, whole or not, keep float32. A NumPy float64 number or a float64 array, such as `np.ones(4)` without `dtype`, turns the result into float64.

</details>

## Part B: rows are tokens

About 10 to 15 minutes.

### B1. Rows, columns, heads and layers

`a` below is A of head 4 in layer 5: row i belongs to token i, the token that reads, column j to token j, the token that is read. Predict, then run:

<!-- input: b1 -->
```python
a = layer.weights[4]
a[0]
a[-1].round(2)
a[:, 0].round(2)
a[2:4].shape
np.array_equal(trace.attention[5, 4], a)
trace.attention[:, :, -1, :].shape
trace.attention[..., -1, :].shape
```

<details>
<summary>Check your reasoning</summary>

<!-- session: b1 setup=trace -->
```pycon
>>> a = layer.weights[4]
>>> a[0]
array([1., 0., 0., 0., 0., 0.], dtype=float32)
>>> a[-1].round(2)
array([0.7 , 0.03, 0.04, 0.01, 0.23, 0.  ], dtype=float32)
>>> a[:, 0].round(2)
array([1.  , 0.9 , 0.64, 0.69, 0.32, 0.7 ], dtype=float32)
>>> a[2:4].shape
(2, 6)
>>> np.array_equal(trace.attention[5, 4], a)
True
>>> trace.attention[:, :, -1, :].shape
(12, 12, 6)
>>> trace.attention[..., -1, :].shape
(12, 12, 6)
```

`a[0]` is row 0: token 0 may read only itself. `a[-1]` is the last row, "it" reading every token. `a[:, 0]` is column 0: how much every token reads token 0. A slice such as `a[2:4]` keeps its axis, rows 2 and 3, so it stays 2-D. With four axes, `[:, :, -1, :]` keeps all layers, all heads and all columns of the last row; `...` stands for "all axes before", so `[..., -1, :]` is the same.

</details>

### B2. Look up the tokens: `embed()`

`forward()` in `gpt2.py` starts with the input of the first block: each token's row of the embedding table, `weights["wte.weight"]`, [V, d], plus the row of the position table, `weights["wpe.weight"]`, for its position, 0 for the first token, 1 for the second. Write `embed(weights, ids)` in `my_numpy.py` without looking at `forward()`, and without a loop; slide 27 shows how a list of indices picks rows. To try it in the REPL:

<!-- input: b2 -->
```python
from eaas_attention_lab import checks
model = checks.tiny_model()
ids = checks.encode(model, "the cat sat on it")
ids
model.weights["wte.weight"].shape, model.weights["wpe.weight"].shape
```

Then run `uv run my-numpy`. When `embed` says `True`, compare your lines with lines 203 and 204 of `gpt2.py`. To call your own `embed` in the REPL, start a new one: a running REPL keeps the `my_numpy.py` it imported with the start lines.

<details>
<summary>Check your reasoning</summary>

<!-- session: b2 setup=trace -->
```pycon
>>> from eaas_attention_lab import checks
>>> model = checks.tiny_model()
>>> ids = checks.encode(model, "the cat sat on it")
>>> ids
[0, 1, 4, 16, 26, 31]
>>> model.weights["wte.weight"].shape, model.weights["wpe.weight"].shape
((50, 48), (32, 48))
```

A list of indices picks rows, in its order and with repeats, and copies them: one row of the embedding table per token, (6, 48). The positions are a slice instead, rows 0 to T-1 of the position table, whatever the IDs are. The two have the same shape, so `+` adds them row by row. Picking the position rows by ID would be wrong without any error for this text, whose largest ID, 31, is still one of the tiny GPT-2's 32 positions; the check text has larger IDs, and there it fails.

</details>

### B3. How much does every token read the first one? `first_token_weights()`

Write `first_token_weights(trace)`: for every layer, every head and every reading token i, the weight A[i, 0], as one array [layers, heads, T]. One line, no loop. `uv run my-numpy` checks it.

<details>
<summary>Check your reasoning</summary>

The first token is the token that is read, so it is column 0 of every A, in all layers and heads at once. Row 0 would be the first token reading, which is always 1, 0, 0 and so on. In GPT-2 small, many heads put most of a row's weight on the first token, the attention sink of `attention_sink.py`; that command averages such values over the heads of each layer. In the tiny GPT-2 the weights are random, so its values mean nothing.

</details>

### ★ A boolean index copies, a row picked by an index writes through

<!-- input: bstar -->
```python
from eaas_attention_lab import show
future = show.future(6)
a[future]
b = a.copy()
first = b[0]
first[:] = 0.5
b[0]
np.shares_memory(b, b[future])
```

<details>
<summary>Check your reasoning</summary>

<!-- session: bstar setup=a -->
```pycon
>>> from eaas_attention_lab import show
>>> future = show.future(6)
>>> a[future]
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      dtype=float32)
>>> b = a.copy()
>>> first = b[0]
>>> first[:] = 0.5
>>> b[0]
array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float32)
>>> np.shares_memory(b, b[future])
False
```

A boolean index picks the cells where the mask is `True`, the future, as one flat array; they are all 0 in A. It copies them, so it shares no memory with `b`. A row picked by an index, such as `b[0]`, is a view, like a slice such as `b[1:]`: writing into it changes `b`. That is why `b` is a copy of `a`, and why the checks of `my_numpy.py` copy every array they hand to your functions.

</details>

## Part C: one line for all rows

About 15 to 25 minutes.

### C1. Which axis disappears?

Predict each result, shape and numbers, then run:

<!-- input: c1 -->
```python
x = np.arange(6.0).reshape(2, 3)
x
x.sum(axis=0)
x.sum(axis=-1)
x.sum(axis=-1, keepdims=True)
x - x.mean(axis=-1, keepdims=True)
x + np.array([10.0, 20.0, 30.0])
x + np.array([[100.0], [200.0]])
(np.zeros((4, 2, 3)) + x).shape
(x > 2).sum()
x + np.array([10.0, 20.0])
```

<details>
<summary>Check your reasoning</summary>

<!-- session: c1 setup=np -->
```pycon
>>> x = np.arange(6.0).reshape(2, 3)
>>> x
array([[0., 1., 2.],
       [3., 4., 5.]])
>>> x.sum(axis=0)
array([3., 5., 7.])
>>> x.sum(axis=-1)
array([ 3., 12.])
>>> x.sum(axis=-1, keepdims=True)
array([[ 3.],
       [12.]])
>>> x - x.mean(axis=-1, keepdims=True)
array([[-1.,  0.,  1.],
       [-1.,  0.,  1.]])
>>> x + np.array([10.0, 20.0, 30.0])
array([[10., 21., 32.],
       [13., 24., 35.]])
>>> x + np.array([[100.0], [200.0]])
array([[100., 101., 102.],
       [203., 204., 205.]])
>>> (np.zeros((4, 2, 3)) + x).shape
(4, 2, 3)
>>> (x > 2).sum()
np.int64(3)
>>> x + np.array([10.0, 20.0])
Traceback (most recent call last):
  ...
ValueError: operands could not be broadcast together with shapes (2,3) (2,)
```

A reduction removes the axis it runs along: `axis=0` sums down the columns, `axis=-1` along each row. `keepdims=True` leaves that axis with length 1, a column (2, 1), which fits the rows of `x` again. Broadcasting compares shapes from the right: (3,) fits (2, 3) as a row added to every row, (2, 1) as a column added to every column, and (2, 3) fits a stack (4, 2, 3). (2,) does not fit (2, 3), because the last axes, 2 and 3, differ: the message names both shapes. Summing booleans counts the `True`s.

</details>

### C2. LayerNorm: `layer_norm_rows()`

LayerNorm gives each token's row mean 0 and variance 1, adds a small `eps` to the variance before the square root, then multiplies by a learned gain and adds a learned bias, one number per column. Write `layer_norm_rows(x, gain, bias, eps)` without looking at `layer_norm()` in `gpt2.py`, and without a loop. The check uses the residual stream entering block 5, with one row whose numbers are all equal.

<details>
<summary>Check your reasoning</summary>

Mean and variance run along the last axis, one value per row, and `keepdims=True` keeps them as a column (T, 1) that fits (T, d). Gain and bias, (d,), are rows that broadcasting adds to every token's row. A row whose numbers are all equal has variance 0: without `eps`, its numbers become 0 / 0, which is nan. Compare your function with lines 109 to 113 of `gpt2.py`.

</details>

### C3. A window instead of the whole past: `window_mask()`

Some models let each token read only a window of nearby tokens, to save work on long texts. In Mistral 7B (Jiang et al. 2023, arXiv 2310.06825), each token reads itself and a window of the tokens just before it; Longformer (Beltagy et al. 2020, arXiv 2004.05150) uses a window on both sides of each token, plus a few tokens that read everything. Write `window_mask(t, w)` for the first kind: an additive mask [t, t], float32, with 0 where token i may read token j, and minus infinity where it may not, in the future and more than w tokens back. No loop. `np.triu` and `np.tril`, whose `k` argument moves the diagonal, and `np.where` may help.

Once `window_mask` says `True`, every run prints A of head 4 in layer 5 with this mask. Change `WINDOW` at the top of `my_numpy.py` and run again: what happens to the rows?

<details>
<summary>Check your reasoning</summary>

The mask is added to the scores before the softmax: 0 leaves a score as it is, minus infinity turns its weight into exactly 0, because exp(minus infinity) is 0. A mask of 0 and 1 would only shift the scores, and every token would still read every other. Each row of A then spreads its weight over at most w + 1 tokens, itself and the w before it, and still sums to 1. With `WINDOW` as large as the text, the mask is the causal mask of `gpt2.py`. The mask must be float32: a float64 mask would turn the float32 scores into float64.

</details>

### ★ When `exp` overflows

<!-- input: cstar -->
```python
np.finfo(np.float32).max
np.exp(np.float32(88.0))
np.exp(np.float32(89.0))
s = np.array([1000.0, 1001.0], dtype=np.float32)
np.exp(s) / np.exp(s).sum()
np.exp(s - s.max()) / np.exp(s - s.max()).sum()
np.isfinite(np.exp(s))
hidden = np.array([-np.inf, -np.inf], dtype=np.float32)
hidden - hidden.max()
```

<details>
<summary>Check your reasoning</summary>

<!-- session: cstar setup=np -->
```pycon
>>> np.finfo(np.float32).max
np.float32(3.4028235e+38)
>>> np.exp(np.float32(88.0))
np.float32(1.6516363e+38)
>>> np.exp(np.float32(89.0))
RuntimeWarning: overflow encountered in exp
np.float32(inf)
>>> s = np.array([1000.0, 1001.0], dtype=np.float32)
>>> np.exp(s) / np.exp(s).sum()
RuntimeWarning: overflow encountered in exp
RuntimeWarning: invalid value encountered in divide
array([nan, nan], dtype=float32)
>>> np.exp(s - s.max()) / np.exp(s - s.max()).sum()
array([0.2689414, 0.7310586], dtype=float32)
>>> np.isfinite(np.exp(s))
RuntimeWarning: overflow encountered in exp
array([False, False])
>>> hidden = np.array([-np.inf, -np.inf], dtype=np.float32)
>>> hidden - hidden.max()
RuntimeWarning: invalid value encountered in subtract
array([nan, nan], dtype=float32)
```

float32 holds numbers up to about 3.4 × 10³⁸, the first line, and exp passes that between 88 and 89: the result is `inf`, with a warning. inf divided by inf is nan. Subtracting the row's maximum first changes no weight, because every exp is multiplied by the same factor, but the largest exp becomes exp(0) = 1. A row in which every score is minus infinity has nothing to read: minus infinity minus minus infinity is nan. `softmax()` in `gpt2.py` subtracts each row's largest score after the mask, so such a row would give nan: a mask must leave every token at least itself.

</details>

## Part D: heads as a stack

About 10 to 15 minutes.

### D1. Stacks

Predict, then run:

<!-- input: d1 -->
```python
x = np.arange(24.0).reshape(2, 3, 4)
x.T.shape
x.transpose(0, 2, 1).shape
(x @ x.transpose(0, 2, 1)).shape
np.arange(8).reshape(2, 4).reshape(4, 2)
y = np.arange(12.0).reshape(3, 4)
y.reshape(3, 2, 2).transpose(1, 0, 2)[1]
[part.shape for part in np.split(y, 2, axis=-1)]
np.stack([y, y, y]).shape
(y[0] @ y.T).shape
```

<details>
<summary>Check your reasoning</summary>

<!-- session: d1 setup=np -->
```pycon
>>> x = np.arange(24.0).reshape(2, 3, 4)
>>> x.T.shape
(4, 3, 2)
>>> x.transpose(0, 2, 1).shape
(2, 4, 3)
>>> (x @ x.transpose(0, 2, 1)).shape
(2, 3, 3)
>>> np.arange(8).reshape(2, 4).reshape(4, 2)
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7]])
>>> y = np.arange(12.0).reshape(3, 4)
>>> y.reshape(3, 2, 2).transpose(1, 0, 2)[1]
array([[ 2.,  3.],
       [ 6.,  7.],
       [10., 11.]])
>>> [part.shape for part in np.split(y, 2, axis=-1)]
[(3, 2), (3, 2)]
>>> np.stack([y, y, y]).shape
(3, 3, 4)
>>> (y[0] @ y.T).shape
(3,)
```

`.T` reverses all axes; `transpose(0, 2, 1)` swaps only the last two. `@` multiplies the matrices in the last two axes and treats every axis before them as a stack: two 3 × 4 matrices times two 4 × 3 matrices give two 3 × 3 matrices. `reshape` reads and writes row by row. `y.reshape(3, 2, 2).transpose(1, 0, 2)` is `split_heads` of `gpt2.py` for 2 heads of 2 numbers: head 1 gets columns 2 and 3 of every row. `np.split` returns a list of views; `np.stack` adds a new first axis. A 1-D operand of `@` loses its axis: a row times a matrix gives a row.

</details>

### D2. All heads at once: `all_scores()`

Write `all_scores(q, k)`: S = Q Kᵀ / √d_h for all heads at once, [heads, T, T], float32, from `q` and `k` of shape [heads, T, d_h]. One `@`, no loop. `uv run my-numpy` checks it on `layer.q` and `layer.k`; compare with line 168 of `gpt2.py`.

<details>
<summary>Check your reasoning</summary>

`k.T` turns [heads, T, d_h] into [d_h, T, heads], so `q @ k.T` fails with the matmul message: read its end, the two sizes that differ. Swapping only the last two axes gives [heads, d_h, T], and `@` computes all 12 products at once. Divide by the square root of d_h, the last axis of `q`, with a Python number, so that the result stays float32.

</details>

### D3. Where is the largest score? `largest_score()`

Write `largest_score(s)`: the cell (i, j) of the largest score in one head's S, [T, T]. No loop; `argmax()` and `np.unravel_index` may help. Once it says `True`, every run prints the cell: does the largest score lie in the future, where the mask hides it?

<details>
<summary>Check your reasoning</summary>

`argmax()` without an axis counts through the matrix row by row and returns one flat index; `np.unravel_index` turns it into (row, column) for any shape. `argmax(axis=-1)` answers another question, one column per row. The cell is (row, column): row i is the token that reads. `int()` turns the NumPy integers into Python ints, which print as plain numbers. Whether the largest score lies in the future depends on the head and the text; task 1 of one_head asked about the highest score of a row in GPT-2 small.

</details>

### ★ Put the heads back together: `merge()`

Write `merge(heads)`: [heads, T, d_h] to [T, d], each token's row holding all heads' results side by side, without `merge_heads()` of `gpt2.py`.

<details>
<summary>Check your reasoning</summary>

`reshape` reads the stack head by head, so `reshape` alone fills each row with one head's results for several tokens. Transposing first, to [T, heads, d_h], brings each token's results of all heads together, and `reshape` then lays them side by side. For the head outputs, a fresh block, the transposed stack no longer reads row by row, so this `reshape` has to copy. Whether it always does is a question of task E2.

</details>

### ★ A loop against one `@`

How much faster is one `@` than a Python loop? Time both on your laptop, with the sizes of slide 2, 1024 tokens and d_h = 64. The loop takes about a second:

```python
import time, timeit
rng = np.random.default_rng(0)
q = rng.standard_normal((1024, 64)).astype(np.float32)
k = rng.standard_normal((1024, 64)).astype(np.float32)
start = time.perf_counter(); loop = [[q[i] @ k[j] for j in range(1024)] for i in range(1024)]; time.perf_counter() - start
min(timeit.repeat(lambda: q @ k.T, number=1, repeat=20))
```

<details>
<summary>Check your reasoning</summary>

The loop runs a Python step for each of the 1024 × 1024 pairs; `@` hands the whole product to a compiled library, BLAS, which can use several cores (`np.show_config()` names the one your NumPy uses). The first `@` also has to start BLAS, so the last line takes the fastest of 20 runs, as slide 2 did. Your numbers differ from the slide with the laptop and its power settings, but the loop stays slower by about three orders of magnitude. Time `q @ k.T`, not `q @ q.T`: for a matrix times its own transpose NumPy calls another routine, which took about three times as long on our laptop (part E of Inside NumPy). `uv run numpy-timings` measures this loop, one `@` and seven ways in between. Your REPL imported NumPy before the lab, so OpenBLAS uses one thread per logical CPU there, not the lab's 4 (Inside NumPy, E3).

</details>

## Part E: read gpt2.py

About 20 to 30 minutes, at home if need be.

### E1. The shapes of `block()`

Fill in the shape after each line of `gpt2.py` for "the cat sat on it" with `<|endoftext|>` in front, T = 6, in the tiny GPT-2 (d = 48, 12 heads of d_h = 4, V = 50) and in GPT-2 small (d = 768, 12 heads of d_h = 64, V = 50,257). Then open the box.

| Line | Code | tiny GPT-2 | GPT-2 small |
| --- | --- | --- | --- |
| 203 | `x0 = weights["wte.weight"][token_ids]` | | |
| 155 | `x = layer_norm(h, w("ln_1.weight"), w("ln_1.bias"), config.eps)` | | |
| 159 | `qkv = x @ w("attn.c_attn.weight") + w("attn.c_attn.bias")` | | |
| 160 | `q_all, k_all, v_all = np.split(qkv, 3, axis=-1)`, each | | |
| 161 | `q = split_heads(q_all, config.n_heads)` | | |
| 168 | `scores = q @ k.transpose(0, 2, 1) / math.sqrt(config.d_h)` | | |
| 169 | `attention_weights = softmax(scores + causal_mask(len(h)))` | | |
| 170 | `head_outputs = attention_weights @ v` | | |
| 174 | `concat = merge_heads(head_outputs)` | | |
| 175 | `attn_out = concat @ w("attn.c_proj.weight") + w("attn.c_proj.bias")` | | |
| 180 | `hidden = gelu(x @ w("mlp.c_fc.weight") + w("mlp.c_fc.bias"))` | | |
| 181 | `h_out = h_mid + hidden @ w("mlp.c_proj.weight") + w("mlp.c_proj.bias")` | | |
| 215 | `logits = final @ weights["wte.weight"].T` | | |

<details>
<summary>Check your reasoning</summary>

<!-- table: e1 -->
| Line | Code | tiny GPT-2 | GPT-2 small |
| --- | --- | --- | --- |
| 203 | `x0 = weights["wte.weight"][token_ids]` | (6, 48) | (6, 768) |
| 155 | `x = layer_norm(h, w("ln_1.weight"), w("ln_1.bias"), config.eps)` | (6, 48) | (6, 768) |
| 159 | `qkv = x @ w("attn.c_attn.weight") + w("attn.c_attn.bias")` | (6, 144) | (6, 2304) |
| 160 | `q_all, k_all, v_all = np.split(qkv, 3, axis=-1)`, each | (6, 48) | (6, 768) |
| 161 | `q = split_heads(q_all, config.n_heads)` | (12, 6, 4) | (12, 6, 64) |
| 168 | `scores = q @ k.transpose(0, 2, 1) / math.sqrt(config.d_h)` | (12, 6, 6) | (12, 6, 6) |
| 169 | `attention_weights = softmax(scores + causal_mask(len(h)))` | (12, 6, 6) | (12, 6, 6) |
| 170 | `head_outputs = attention_weights @ v` | (12, 6, 4) | (12, 6, 64) |
| 174 | `concat = merge_heads(head_outputs)` | (6, 48) | (6, 768) |
| 175 | `attn_out = concat @ w("attn.c_proj.weight") + w("attn.c_proj.bias")` | (6, 48) | (6, 768) |
| 180 | `hidden = gelu(x @ w("mlp.c_fc.weight") + w("mlp.c_fc.bias"))` | (6, 192) | (6, 3072) |
| 181 | `h_out = h_mid + hidden @ w("mlp.c_proj.weight") + w("mlp.c_proj.bias")` | (6, 48) | (6, 768) |
| 215 | `logits = final @ weights["wte.weight"].T` | (6, 50) | (6, 50257) |

Every matrix keeps one row per token, from the lookup to the logits. GPT-2 stores its weights as [inputs, outputs], so `x @ W` needs no transpose: c_attn is [d, 3d], the feed-forward network goes from d to 4d and back. The scores and weights are T × T in both models: they depend on the text, not on the model's width. The output layer reuses the embedding table: `wte.T` is [d, V], one score per vocabulary entry.

</details>

### E2. View or copy?

Which of these lines write a new block of numbers, and which only a new header over an existing block? Predict, then check in your REPL:

<!-- input: e2 -->
```python
from eaas_attention_lab import gpt2
qkv = np.zeros((6, 144), dtype=np.float32)
q_all = np.split(qkv, 3, axis=-1)[0]
np.shares_memory(qkv, q_all)
q = gpt2.split_heads(q_all, 12)
np.shares_memory(qkv, q)
np.shares_memory(q, q.transpose(0, 2, 1))
outputs = np.zeros((12, 6, 4), dtype=np.float32)
np.shares_memory(outputs, gpt2.merge_heads(outputs))
np.shares_memory(q, gpt2.merge_heads(q))
table = np.zeros((50, 48), dtype=np.float32)
np.shares_memory(table, table[[3, 1, 4]])
np.shares_memory(table, table[:6])
trace.attention is trace.attention
```

<details>
<summary>Check your reasoning</summary>

<!-- session: e2 setup=trace -->
```pycon
>>> from eaas_attention_lab import gpt2
>>> qkv = np.zeros((6, 144), dtype=np.float32)
>>> q_all = np.split(qkv, 3, axis=-1)[0]
>>> np.shares_memory(qkv, q_all)
True
>>> q = gpt2.split_heads(q_all, 12)
>>> np.shares_memory(qkv, q)
True
>>> np.shares_memory(q, q.transpose(0, 2, 1))
True
>>> outputs = np.zeros((12, 6, 4), dtype=np.float32)
>>> np.shares_memory(outputs, gpt2.merge_heads(outputs))
False
>>> np.shares_memory(q, gpt2.merge_heads(q))
True
>>> table = np.zeros((50, 48), dtype=np.float32)
>>> np.shares_memory(table, table[[3, 1, 4]])
False
>>> np.shares_memory(table, table[:6])
True
>>> trace.attention is trace.attention
False
```

`np.split`, `split_heads` (a `reshape` and a `transpose`) and `transpose` only write new headers: q of every head is still a window into `qkv`. `merge_heads` copies the head outputs, a fresh block like `outputs`, in which each head's numbers lie together and a token's numbers lie apart: no header can read them as rows. Merging `q` needs no copy, because transposing back restores the layout of `qkv`, in which a token's 48 numbers lie side by side. Whether `reshape` copies depends on how the numbers lie, not on the function. A list of indices copies, a slice does not. `trace.attention` is a property that stacks the 12 layers anew at every access: two accesses give two different arrays, with the same numbers. In the comment of `tests/test_gpt2.py`, line 40, "stacked views" means the different ways to look at the trace, not NumPy views.

</details>

### E3. A reading map of the commands

For each line, say what it computes and the shape of its result, for "the cat sat on it" in the tiny GPT-2. Then open the box and run its lines in your REPL.

1. `one_head.py`, line 75: `w_q, w_k, w_v = np.split(w[prefix + "attn.c_attn.weight"], 3, axis=1)`
2. `one_head.py`, lines 77 and 78: `head = slice(HEAD * d_h, (HEAD + 1) * d_h)`, then `x @ w_q[:, head] + b_q[head]`
3. `one_head.py`, line 185: `order = np.argsort(-a[-1])`
4. `attention_sink.py`, lines 51 and 52: `last = trace.attention[:, :, -1, :]`, then `int((last.argmax(axis=-1) == 0).sum())`
5. `attention_sink.py`, lines 87 and 88: `lengths = np.linalg.norm(trace.values, axis=-1)`, then `lengths[..., 1:].mean(axis=-1)`
6. `attention_sink.py`, line 91: `np.linalg.norm(last[..., np.newaxis] * trace.values, axis=-1)`
7. `model_weights.py`, lines 146 and 147: `np.tril(np.ones((n, n), np.float32))`, compared with GPT-2's stored mask by `np.array_equal`
8. `model_weights.py`, line 165: `int(np.prod(shape))`
9. `one_head.py`, line 164: `late = softmax(s) * ~show.future(t)`
10. `gpt2.py`, line 38: `Array: TypeAlias = NDArray[np.floating[Any]]`

<details>
<summary>Check your reasoning</summary>

<!-- session: e3 setup=trace -->
```pycon
>>> from eaas_attention_lab import checks, show
>>> w = checks.tiny_model().weights
>>> [part.shape for part in np.split(w["h.5.attn.c_attn.weight"], 3, axis=1)]
[(48, 48), (48, 48), (48, 48)]
>>> a = layer.weights[4]
>>> np.argsort(-a[-1])
array([0, 4, 2, 1, 3, 5])
>>> last = trace.attention[:, :, -1, :]
>>> last.shape, (last.argmax(axis=-1) == 0).shape
((12, 12, 6), (12, 12))
>>> lengths = np.linalg.norm(trace.values, axis=-1)
>>> lengths.shape, lengths[..., 1:].mean(axis=-1).shape
((12, 12, 6), (12, 12))
>>> (last[..., np.newaxis] * trace.values).shape
(12, 12, 6, 4)
>>> np.tril(np.ones((3, 3), np.float32))
array([[1., 0., 0.],
       [1., 1., 0.],
       [1., 1., 1.]], dtype=float32)
>>> int(np.prod((48, 144)))
6912
>>> ~show.future(3)
array([[ True, False, False],
       [ True,  True, False],
       [ True,  True,  True]])
```

1. Three [d, d] matrices, W_Q, W_K and W_V, from the [d, 3d] block of a layer. 2. `slice(...)` is a stored `start:stop`: the d_h columns of one head, so the head's queries are [T, d_h]. 3. The tokens by weight in the last row, largest first: `argsort` sorts ascending, so it sorts the negated weights. 4. The last row of every head, [layers, heads, T], then for every head whether its largest weight goes to token 0, [layers, heads], and the number of such heads, one Python int. 5. The length of every value vector, [layers, heads, T], then their mean over the tokens after the first, [layers, heads]. 6. `np.newaxis` adds an axis of length 1, so [layers, heads, T, 1] times [layers, heads, T, d_h] multiplies each value vector by its weight; the norm along the last axis then gives the length of each weighted vector, [layers, heads, T]. 7. GPT-2's stored mask has 1 where a token may read and 0 above the diagonal: `np.tril` keeps the lower triangle. 8. The number of numbers of a tensor from its shape. 9. `~` turns `True` into `False` and back: `True` where a token may read. In arithmetic, `True` counts as 1 and `False` as 0, so the product sets every weight in the future to 0, after the softmax: the rows then sum to less than 1, which step 6 of `one_head.py` prints. 10. A type hint only: an array of floating-point numbers; Python does not check it.

</details>

### ★ show.py and the Debug Console

`show.py` draws the figures of the commands. In `heatmap()`, `np.ma.masked_array` hides the masked cells, which `imshow` then paints grey, and `np.ndindex(t, t)` runs through all cells (i, j). In `head_grid()`, `fig.subplots(...)` returns a NumPy array of Axes, and `.flat` runs through it row by row.

To see the shapes of E1 in the running model: in `gpt2.py`, set a breakpoint on the line `attention_weights = softmax(scores + causal_mask(len(h)))` (click left of its line number), select the **Run and Debug** view, choose **Attention: My NumPy (exercise)** and click **Start Debugging**. The debugger first stops at the top of `my_numpy.py`: click **Continue**. At the breakpoint, open the **Debug Console** next to **Terminal** in the bottom panel and type `scores.shape`, `q.strides` or `x.dtype`. The run uses the check text of 9 tokens, so T is 9 there, not 6. The breakpoint stops once per block: click **Stop** when you are done.

## First aid

| You see | What helps |
| --- | --- |
| `NameError: name 'np' is not defined` | Type `import numpy as np` first: every new REPL starts empty. |
| `ModuleNotFoundError: No module named 'eaas_attention_lab'` | Start the REPL with `uv run python` in the `attention_lab` folder, not with `python`. |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, ...` | Read the end of the line, `(size ... is different from ...)`: the inner sizes of `@` differ. Print both shapes. |
| `ValueError: operands could not be broadcast together with shapes ...` | Compare the two shapes from the right; a reduction may need `keepdims=True`. |
| `IndexError: index 6 is out of bounds for axis 0 with size 6` | Indices count from 0: the last of 6 rows is 5, or -1. |
| `RuntimeWarning: overflow encountered in exp` | A number above about 88 went into `exp` in float32: subtract the row's largest value first. The REPL of Python 3.13, which the lab uses, repeats a warning at every call; older ones show the same warning only once. |
| `The tiny model does not know: ...` | The tiny GPT-2 knows 49 words; the message lists them. |
| `not written yet` | The function still raises `NotImplementedError`: write it in place of that line. |
| Your function works in the file, but the REPL still raises `NotImplementedError` | A running REPL keeps the `my_numpy.py` it imported at the start: `exit()` and start a new REPL. |

## Your changes and git pull

Your work stays in your copy of the repository. If a later update changes `my_numpy.py` too, `git pull` stops instead of overwriting your work. Before you pull, either keep your solution on a branch of your own (`git switch -c my-numpy`, `git commit -am "My NumPy"`, then `git switch main`), or undo your changes to `my_numpy.py` with **Source Control > Discard Changes**.
