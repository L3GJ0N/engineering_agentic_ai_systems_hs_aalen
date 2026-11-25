# Exercise: Inside NumPy

NumPy is the library behind `gpt2.py`: one type, the n-dimensional array, written in C, and the functions around it. This exercise looks inside it. What an array is in memory, how its header reads its block of numbers, which lines copy and which do not, and why one `@` is a thousand times faster than a Python loop, measured on your laptop. Afterwards you can say what NumPy does for every line you write, and why it is fast or slow. The exercise [NumPy for the attention lab](../numpy_for_attention/README.md) then applies all of it to GPT-2; do this one first.

You work alone. Six parts, A to F, take about 100 to 120 minutes in all; finish at home what the session leaves. Tasks marked ★ are optional. Everything runs on your laptop's CPU, with random numbers and without any download.

All you change is one file: [my_numpy_inside.py](../../attention_lab/src/eaas_attention_lab/exercises/my_numpy_inside.py) in `attention_lab`. Everything else happens in the Python REPL and in the terminal.

## How you work

- **Predict first.** Before you run a line, write down what you expect it to print, in a file or on paper. The surprises are the point.
- **The REPL.** In a terminal in `attention_lab`, `uv run python` starts Python with NumPy and the lab; `>>>` is its prompt. Type one line, press Enter, and Python prints the result. `exit()` ends it. Every REPL starts empty: type `import numpy as np` first. Each task gives all the other lines it needs.
- **The checks.** `uv run my-numpy-inside` checks the functions in `my_numpy_inside.py` and prints one line for each: `True`, `not written yet`, or `False` with a hint at what to change. The checks compare your results with what NumPy itself computes. Once a function of part D works, the run also times it.
- **The timings.** `uv run numpy-timings` computes the scores of one attention head in nine ways and times each on your laptop; part E reads its table.
- **The boxes.** Each task ends with a closed box, **Check your reasoning**. Open it after you tried. Its REPL sessions are real runs, with NumPy 2.5.4 and Python 3.13.

Write the code yourself: a coding agent or chatbot that writes these few lines takes the point away. Asking one for an explanation is fine.

## Get started

1. Update your copy of the course repository: `git pull`, or download it again. This exercise came on 26 November. If `git pull` stops because you changed a file of an earlier exercise, read [Your changes and git pull](#your-changes-and-git-pull) first.
2. Open the course workspace in VS Code with **File > Open Workspace from File** and `engineering_agentic_ai_systems.code-workspace`. Select **Terminal > New Terminal** and choose **attention_lab**.
3. Run:

   ```console
   uv sync
   uv run my-numpy-inside
   ```

The first run checks eight functions, and all eight say `not written yet`: seven you write in parts A to D, the eighth, `softmax_python`, is the ★ task of part D.

## Part A: an array is a header and a block

About 20 to 25 minutes.

A NumPy array is an object of the class `numpy.ndarray`, written in C. It holds its numbers side by side in one block of memory, all of one type, the dtype, and a small header that says how to read the block: the number of axes (`ndim`), the length of each axis (`shape`), the dtype, and the strides, the number of bytes from one number to the next along each axis.

### A1. The header

Predict what each line prints, then type it into your REPL:

<!-- input: a1 -->
```python
a = np.zeros((4, 2, 3))
type(a)
a.ndim, a.shape, a.dtype
a.itemsize, a.size, a.nbytes
a.strides
a.shape()
```

<details>
<summary>Check your reasoning</summary>

<!-- session: a1 setup=np -->
```pycon
>>> a = np.zeros((4, 2, 3))
>>> type(a)
<class 'numpy.ndarray'>
>>> a.ndim, a.shape, a.dtype
(3, (4, 2, 3), dtype('float64'))
>>> a.itemsize, a.size, a.nbytes
(8, 24, 192)
>>> a.strides
(48, 24, 8)
>>> a.shape()
Traceback (most recent call last):
  ...
TypeError: 'tuple' object is not callable
```

`np.zeros` is a function; it returns an object of the class `numpy.ndarray`. Without `dtype`, its numbers are float64, 8 bytes each (`itemsize`): three axes of lengths 4, 2 and 3, a stack of four 2 × 3 matrices, so 24 numbers (`size`) in 192 bytes (`nbytes`). The strides follow from the shape, from the last axis to the first: 8 bytes to the next number of a row, 3 × 8 = 24 to the next row, 2 × 24 = 48 to the next matrix. Attributes such as `a.shape` are read without parentheses; methods such as `a.sum()` are called with them, and `a.shape()` tries to call the tuple `(4, 2, 3)`.

</details>

### A2. GPT-2's tables

The checks of the lab use a tiny GPT-2 with random weights: the layout of GPT-2 small, 12 blocks with 12 heads, but 48 numbers per token instead of 768, and 50 vocabulary entries instead of 50,257. Its weights are a dict of arrays, with the names of GPT-2 small's file. Predict, then run:

<!-- input: a2 -->
```python
from eaas_attention_lab import checks
w = checks.tiny_model().weights
wte = w["wte.weight"]
wte.shape, wte.dtype, wte.strides
wte.nbytes
sum(x.nbytes for x in w.values())
50257 * 768 * 4
```

After the run: how many bytes does the embedding table of GPT-2 small take, [50257, 768] in float32, and what are its strides?

<details>
<summary>Check your reasoning</summary>

<!-- session: a2 setup=np -->
```pycon
>>> from eaas_attention_lab import checks
>>> w = checks.tiny_model().weights
>>> wte = w["wte.weight"]
>>> wte.shape, wte.dtype, wte.strides
((50, 48), dtype('float32'), (192, 4))
>>> wte.nbytes
9600
>>> sum(x.nbytes for x in w.values())
1373184
>>> 50257 * 768 * 4
154389504
```

One row per vocabulary entry, 48 numbers of 4 bytes each: the next row starts 192 bytes further, the next number 4 bytes. The table takes 50 × 192 = 9,600 bytes, and all weights of the tiny model together about 1.4 MB. GPT-2 small's table, [50257, 768], has the strides (3072, 4) and takes 154,389,504 bytes, about 154 MB; all 124,439,808 weights of GPT-2 small take about 498 MB in float32. `nbytes` counts the numbers in the block only, not the header.

</details>

### A3. The strides of a new array: `c_strides()`

When NumPy makes a new array, it lays the numbers out row by row, the last axis fastest, in what is called C order: the next number along the last axis is one item further, and one step along any other axis skips everything that the axes after it hold. Write `c_strides(shape, itemsize)` in `my_numpy_inside.py`: the strides NumPy gives a new array of this shape, in bytes, in plain Python. Run `uv run my-numpy-inside` until it says `True`.

<details>
<summary>Check your reasoning</summary>

The loop runs from the last axis to the first: the last stride is `itemsize`, and each stride before it is the stride after it times the length of the axis after it. For (2, 3) and 8 bytes: 8, then 3 × 8 = 24. For a stack of 12 heads, (12, 6, 4) in float32: 4, then 4 × 4 = 16 to the next row, then 6 × 16 = 96 to the next head. NumPy does the same in C, in `_array_fill_strides` (part F links it). An axis of length 1 gets a stride too, although no step along it ever happens.

</details>

### A4. Where a number lies: `byte_offset()`

`a[1, 2]` lies one step along axis 0 and two steps along axis 1 from `a[0, 0]`. Write `byte_offset(strides, index)`: how many bytes after the array's first number `a[index]` lies, for any number of axes, in plain Python. The checks include views whose strides are not those of a new array, and one whose stride is negative.

<details>
<summary>Check your reasoning</summary>

Each position of the index times the stride of its axis, summed: for the strides (24, 8) and the index (1, 2), 1 × 24 + 2 × 8 = 40. The formula does not care whether the strides are those of a new array or of a view: a window into a wider block, such as one head's keys, or a transposed matrix, or rows read backwards with a negative stride. NumPy computes the same in C when you index with integers, in `get_item_pointer` (part F links it). This is why a header with other strides can show the same block in another way without moving a number.

</details>

### ★ A list is not an array

How much memory does a number take in a Python list, and in an array? Predict, then run:

<!-- input: astar -->
```python
import sys, tracemalloc
sys.getsizeof(1.0)
np.zeros(1, dtype=np.float32).itemsize
tracemalloc.start()
floats = [float(i) for i in range(1_000_000)]
tracemalloc.get_traced_memory()[0] // 1_000_000
tracemalloc.stop()
np.zeros(1_000_000, dtype=np.float32).nbytes // 1_000_000
[1.0, 2.0] + [3.0, 4.0], np.array([1.0, 2.0]) + np.array([3.0, 4.0])
```

<details>
<summary>Check your reasoning</summary>

<!-- session: astar setup=np -->
```pycon
>>> import sys, tracemalloc
>>> sys.getsizeof(1.0)
24
>>> np.zeros(1, dtype=np.float32).itemsize
4
>>> tracemalloc.start()
>>> floats = [float(i) for i in range(1_000_000)]
>>> tracemalloc.get_traced_memory()[0] // 1_000_000
32
>>> tracemalloc.stop()
>>> np.zeros(1_000_000, dtype=np.float32).nbytes // 1_000_000
4
>>> [1.0, 2.0] + [3.0, 4.0], np.array([1.0, 2.0]) + np.array([3.0, 4.0])
([1.0, 2.0, 3.0, 4.0], array([4., 6.]))
```

A Python float is an object of its own: 8 bytes of reference count, 8 bytes for a pointer to its type, 8 bytes for the number. A list holds one pointer of 8 bytes per element, so a list of a million floats takes about 32 bytes per number, as `tracemalloc` counts what Python asks for; the allocator rounds each float up to 32 bytes, so the real cost is higher still. A float32 array takes 4 bytes per number, side by side. GPT-2 small's 124,439,808 weights would take at least 4 GB as a list of floats instead of 498 MB. And `+` means something else: it joins two lists, but adds two arrays number by number.

</details>

## Part B: making arrays

About 10 to 15 minutes.

### B1. Constructors and their dtypes

Predict, then run:

<!-- input: b1 -->
```python
np.zeros((2, 3))
np.ones(3, dtype=np.float32)
np.full((2, 2), -np.inf)
np.arange(5), np.arange(0.0, 1.0, 0.25)
np.linspace(0, 1, 5)
np.eye(3)
np.arange(12).reshape(3, 4)
np.arange(12).reshape(2, -1).shape
np.array([1, 2, 3]).dtype, np.array([1, 2.5]).dtype, np.array([True, False]).dtype
np.zeros(2, 3)
```

<details>
<summary>Check your reasoning</summary>

<!-- session: b1 setup=np -->
```pycon
>>> np.zeros((2, 3))
array([[0., 0., 0.],
       [0., 0., 0.]])
>>> np.ones(3, dtype=np.float32)
array([1., 1., 1.], dtype=float32)
>>> np.full((2, 2), -np.inf)
array([[-inf, -inf],
       [-inf, -inf]])
>>> np.arange(5), np.arange(0.0, 1.0, 0.25)
(array([0, 1, 2, 3, 4]), array([0.  , 0.25, 0.5 , 0.75]))
>>> np.linspace(0, 1, 5)
array([0.  , 0.25, 0.5 , 0.75, 1.  ])
>>> np.eye(3)
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
>>> np.arange(12).reshape(3, 4)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> np.arange(12).reshape(2, -1).shape
(2, 6)
>>> np.array([1, 2, 3]).dtype, np.array([1, 2.5]).dtype, np.array([True, False]).dtype
(dtype('int64'), dtype('float64'), dtype('bool'))
>>> np.zeros(2, 3)
Traceback (most recent call last):
  ...
TypeError: Cannot interpret '3' as a data type
```

A shape is one tuple: `np.zeros(2, 3)` passes 3 as the second argument, the dtype, which NumPy cannot read. `arange` stops before its end, like `range`; `linspace` includes it and counts points, not steps. `reshape` fills the new shape row by row, and `-1` lets NumPy compute one length from the others. Without `dtype`, whole numbers become int64, numbers with a decimal point float64, and `True` and `False` bool; one float among ints makes the whole array float64. Most results print without `dtype=`, because float64 and int64 are the defaults; a float32 array names its dtype.

</details>

### B2. Random numbers with a seed

Predict which lines print the same numbers, then run:

<!-- input: b2 -->
```python
rng = np.random.default_rng(42)
rng.normal(size=3)
rng.normal(size=3)
np.random.default_rng(42).normal(size=3)
rng.normal(0.0, 0.02, size=(2, 3)).astype(np.float32)
```

<details>
<summary>Check your reasoning</summary>

<!-- session: b2 setup=np -->
```pycon
>>> rng = np.random.default_rng(42)
>>> rng.normal(size=3)
array([ 0.30471708, -1.03998411,  0.7504512 ])
>>> rng.normal(size=3)
array([ 0.94056472, -1.95103519, -1.30217951])
>>> np.random.default_rng(42).normal(size=3)
array([ 0.30471708, -1.03998411,  0.7504512 ])
>>> rng.normal(0.0, 0.02, size=(2, 3)).astype(np.float32)
array([[ 0.00255681, -0.00632485, -0.00033602],
       [-0.01706088,  0.01758796,  0.01555584]], dtype=float32)
```

A generator made from a seed draws a fixed sequence: every call continues it, so the second line differs from the first, and a new generator with the same seed starts the sequence again. `normal(0.0, 0.02, size=...)` draws with mean 0 and standard deviation 0.02, the spread GPT-2's configuration gives for its initial weights (`"initializer_range": 0.02`), and `.astype(np.float32)` makes float32 numbers of them.

</details>

### B3. A head at random: `init_head()`

Write `init_head(seed, d, d_h)`: W_Q, W_K and W_V of one head, [d, d_h] each, float32, drawn with mean 0 and standard deviation 0.02 from one generator made from `seed`. The same seed must give the same three matrices, whatever ran before, another seed other ones, and the three must differ from each other. The check calls it with the seeds 0, 1 and 0 again, for d = 768 and d_h = 64.

<details>
<summary>Check your reasoning</summary>

One generator, made from `seed` inside the function, and three draws from it: each draw continues the sequence, so the three matrices differ, and the next call with the same seed starts again. A generator made outside the function would continue its sequence across calls; three generators made from the same seed would draw the same matrix three times. Code that draws weights this way runs the same on every laptop, which is what tests and experiments need.

</details>

## Part C: views

About 20 minutes.

A slice, a transpose and most reshapes write a new header and keep the block: the result is a view, which shows the same numbers in another way. `np.shares_memory(x, y)` says whether two arrays use the same bytes, and `x.base` is the array whose block a view uses.

### C1. View or copy?

Predict `True` or `False` for each line, then run:

<!-- input: c1 -->
```python
a = np.arange(12, dtype=np.float32).reshape(3, 4)
np.shares_memory(a, a[1])
np.shares_memory(a, a[:, 1:3])
np.shares_memory(a, a.T)
np.shares_memory(a, a.reshape(4, 3))
np.shares_memory(a, a[[0, 2]])
np.shares_memory(a, a[a > 5])
np.shares_memory(a, a.T.reshape(12))
np.shares_memory(a, a.copy())
a.T.flags["C_CONTIGUOUS"], a.T.flags["F_CONTIGUOUS"]
```

<details>
<summary>Check your reasoning</summary>

<!-- session: c1 setup=np -->
```pycon
>>> a = np.arange(12, dtype=np.float32).reshape(3, 4)
>>> np.shares_memory(a, a[1])
True
>>> np.shares_memory(a, a[:, 1:3])
True
>>> np.shares_memory(a, a.T)
True
>>> np.shares_memory(a, a.reshape(4, 3))
True
>>> np.shares_memory(a, a[[0, 2]])
False
>>> np.shares_memory(a, a[a > 5])
False
>>> np.shares_memory(a, a.T.reshape(12))
False
>>> np.shares_memory(a, a.copy())
False
>>> a.T.flags["C_CONTIGUOUS"], a.T.flags["F_CONTIGUOUS"]
(False, True)
```

A row, a slice, a transpose and the reshape of a new array are views: each needs only another start, other lengths or other strides. A list of indices and a boolean mask pick numbers that no single stride per axis can reach, so NumPy copies them into a new block. `a.T.reshape(12)` must read the numbers column by column, 0, 4, 8, 1, ..., and no stride can do that over a block that lies row by row: this `reshape` copies. Whether a reshape copies depends on how the numbers lie, which the flags show: `a.T` reads its block column by column, Fortran order, not row by row, C order.

</details>

### C2. Writing through a view

Predict what `a` holds after each change, then run:

<!-- input: c2 -->
```python
a = np.zeros((2, 3), dtype=np.float32)
row = a[0]
row[:] = 7
a
picked = a[[1]]
picked[:] = 9
a
row.base is a, np.shares_memory(a, picked)
```

<details>
<summary>Check your reasoning</summary>

<!-- session: c2 setup=np -->
```pycon
>>> a = np.zeros((2, 3), dtype=np.float32)
>>> row = a[0]
>>> row[:] = 7
>>> a
array([[7., 7., 7.],
       [0., 0., 0.]], dtype=float32)
>>> picked = a[[1]]
>>> picked[:] = 9
>>> a
array([[7., 7., 7.],
       [0., 0., 0.]], dtype=float32)
>>> row.base is a, np.shares_memory(a, picked)
(True, False)
```

`row` is a view: writing into it writes into `a`. `picked`, made by a list of indices, is a copy: writing into it leaves `a` as it was. `row[:] = 7` writes into the numbers; `row = 7` would only rebind the name. Both are common sources of bugs: a function that changes a view of its input changes the caller's array, and a change to a copy is lost.

</details>

### C3. The header of a transpose: `transposed()`

`a.transpose(axes)` moves the axes into a new order: axis i of the result is axis `axes[i]` of `a`. `split_heads` in `gpt2.py` uses it to turn [T, heads, d_h] into [heads, T, d_h]. Write `transposed(shape, strides, axes)`: the shape and the strides of `a.transpose(axes)`, from those of `a`, in plain Python.

<details>
<summary>Check your reasoning</summary>

Shape and strides are reordered the same way: each axis keeps its length and its stride, only their order changes. For split_heads, (6, 12, 4) with the strides (576, 16, 4) and the axes (1, 0, 2) give (12, 6, 4) with (16, 576, 4): head h starts h × 16 bytes into each row of the block, and its next row is 576 bytes further. No number moves, so a transpose costs the same for 6 tokens as for 6,000.

</details>

### C4. Row by row without a gap: `is_contiguous()`

An array is C-contiguous when its header reads the block row by row without a gap: its strides are those of a new array of its shape, except on axes of length 1, along which NumPy never steps. Write `is_contiguous(shape, strides, itemsize)`, the answer that `a.flags["C_CONTIGUOUS"]` gives, in plain Python.

<details>
<summary>Check your reasoning</summary>

Walk from the last axis to the first, as in A3, and compare each stride with the stride a new array would have; an axis of length 1 is skipped, whatever its stride. The checks show which views are not contiguous: a transpose, a window such as one head's columns, rows read backwards, the stack that split_heads returns. Some functions need contiguous numbers: `reshape` copies when no strides can do it, and code in C that walks the block from its first byte, as many libraries do, asks for `np.ascontiguousarray(x)`, which copies only when it must.

</details>

### ★ Where a view starts

`x.__array_interface__["data"][0]` is the address of an array's first number in memory. Predict the differences, then run:

<!-- input: cstar -->
```python
a = np.arange(12, dtype=np.float32).reshape(3, 4)
start = a.__array_interface__["data"][0]
a[1:].__array_interface__["data"][0] - start
a[:, 2:].__array_interface__["data"][0] - start
a[::-1].__array_interface__["data"][0] - start
a[::-1].strides
```

<details>
<summary>Check your reasoning</summary>

<!-- session: cstar setup=np -->
```pycon
>>> a = np.arange(12, dtype=np.float32).reshape(3, 4)
>>> start = a.__array_interface__["data"][0]
>>> a[1:].__array_interface__["data"][0] - start
16
>>> a[:, 2:].__array_interface__["data"][0] - start
8
>>> a[::-1].__array_interface__["data"][0] - start
32
>>> a[::-1].strides
(-16, 4)
```

A view is a header with its own start: `a[1:]` starts one row, 16 bytes, later; `a[:, 2:]` two numbers, 8 bytes, later. `a[::-1]` starts at the last row, 32 bytes later, and steps back 16 bytes per row: a negative stride. The addresses themselves differ from run to run, the differences do not.

</details>

## Part D: element-wise or matrix product

About 20 minutes.

### D1. `*` or `@`?

`*`, `+` and `np.exp` work number by number, element-wise; `@` is the matrix product. Predict the shape and the numbers of each result, then run:

<!-- input: d1 -->
```python
x = np.array([[1.0, 2.0], [3.0, 4.0]])
x * x
x @ x
np.array([1.0, 2.0]) @ x
x @ np.array([1.0, 2.0])
np.array([1.0, 2.0]) @ np.array([3.0, 4.0])
np.ones((2, 3)) @ np.ones((2, 3))
np.ones((2, 3)) * np.ones((3, 2))
```

<details>
<summary>Check your reasoning</summary>

<!-- session: d1 setup=np -->
```pycon
>>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> x * x
array([[ 1.,  4.],
       [ 9., 16.]])
>>> x @ x
array([[ 7., 10.],
       [15., 22.]])
>>> np.array([1.0, 2.0]) @ x
array([ 7., 10.])
>>> x @ np.array([1.0, 2.0])
array([ 5., 11.])
>>> np.array([1.0, 2.0]) @ np.array([3.0, 4.0])
np.float64(11.0)
>>> np.ones((2, 3)) @ np.ones((2, 3))
Traceback (most recent call last):
  ...
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
>>> np.ones((2, 3)) * np.ones((3, 2))
Traceback (most recent call last):
  ...
ValueError: operands could not be broadcast together with shapes (2,3) (3,2)
```

`x * x` squares each number; `x @ x` combines rows with columns. A 1-D array on the left of `@` acts as a row, on the right as a column, and two 1-D arrays give their dot product, one number. Read an error from the end of its last line: `@` needs the last axis of the left operand to match the first axis of the right one (`size 2 is different from 3`), and `*` needs shapes that broadcast, which (2, 3) and (3, 2) do not.

</details>

### D2. A dot product in plain Python: `dot_python()`

Write `dot_python(x, y)`: the dot product of two lists of floats, one loop, no NumPy. Once it works, the check computes the scores of one head with it, one call per pair of tokens, for T = 1024 and d_h = 64, and prints how long that takes against one `@`.

<details>
<summary>Check your reasoning</summary>

One running total, and for each pair of numbers, taken side by side from both lists, their product added to it. The timing runs 1024 × 1024 calls, each with a loop of 64 steps: 67 million steps of Python. Part E explains why that takes thousands of times as long as one `@`, although both do the same multiplications.

</details>

### D3. One column per step: `scores_columns()`

S = Q Kᵀ can be built one column at a time: column j of S holds every query's score for key j, Q k_j. Write `scores_columns(q, k)`: a Python loop over the T columns of S, each computed with one `@` and written into its column of a float32 array. Once it works, the check times it against one `@` and against the same loop over the rows, one `@` per row, which `numpy_timings.py` contains.

<details>
<summary>Check your reasoning</summary>

First a float32 array for S, then for each key j its column: the product of the whole query matrix with that one key, [T, d_h] times [d_h], is a vector of T scores, written into column j. Columns and rows take the same number of steps, 1,024, and the same multiplications, yet on our laptop the columns took nearly twice as long. A row of S lies side by side in memory; a column of S is spread over T rows, 4 KB apart, so writing it touches a different stretch of memory for every number. Into an array that lies column by column, made with `order="F"` (A3 calls this Fortran order), the same loop is as fast as the rows: try it.

</details>

### ★ The softmax in plain Python: `softmax_python()`

Write `softmax_python(row)`: the softmax of one row of scores, a list of floats, with the `math` module. Like `softmax()` in `gpt2.py`, subtract the row's largest number first. A score of `-inf`, where the mask hides a token, must get the weight 0. Once it works, the check runs it on all 1,024 rows of one head's scores and times it against `softmax()` of `gpt2.py`, which does all rows at once.

<details>
<summary>Check your reasoning</summary>

Find the row's largest score, take exp of each score minus it, and divide each of these by their sum. Without subtracting the largest number, `math.exp(1000)` raises an `OverflowError`; NumPy's `np.exp` would return `inf` with a warning instead. `math.exp(-inf)` is 0.0, so masked scores need no special case. The timing compares a million calls of `math.exp` in a loop with one `np.exp` on the whole array: part E.

</details>

## Part E: fast and slow

About 15 to 20 minutes.

### E1. Predict, then measure

`uv run numpy-timings` computes the scores of one head, S = Q Kᵀ for T = 1024 tokens and d_h = 64, in nine ways, from three loops in plain Python to one `@` on float32, the way `gpt2.py` does it. Read the nine ways in [numpy_timings.py](../../attention_lab/src/eaas_attention_lab/commands/numpy_timings.py). Then, before you run it, rank them from fastest to slowest and guess how many times slower than one `@` the slowest is. Then run it in the terminal, not in the REPL; it takes about 15 seconds:

```console
uv run numpy-timings
```

<details>
<summary>Check your reasoning</summary>

Our run, on a laptop with an Intel Core i7-12700H (14 cores, 20 threads), Windows 11, on 26 November 2026:

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

Your numbers differ with your laptop, its power settings and what else runs; the order of magnitude stays. Column "Steps in Python" counts how often a line of Python runs in the innermost loop: 67 million in plain Python, a million calls of NumPy for the pairs, 1,024 rows, 64 outer products, and one call for the rest. "Same" compares each result with the exact product.

</details>

### E2. Read the table

Answer these questions from the table, then open the box:

1. NumPy computes each dot product of "one dot product per pair" in C. Why is that way only about twice as fast as plain Python?
2. One row per step takes 1,024 steps, one outer product per step only 64. Why are the outer products slower?
3. Broadcasting has no Python loop at all. Why is it slower than one row per step?
4. Why does one `@` on int32 numbers take about 30 times as long as on float32?
5. Why does float64 take about twice as long as float32?

<details>
<summary>Check your reasoning</summary>

1. Every step of a Python loop costs time of its own: for each multiplication, the interpreter runs about ten bytecode instructions, checks the types and makes a new float object. One call of NumPy costs about as much as a short loop of Python: here, a call for 64 multiplications took about half as long as the 64 steps in plain Python, so a million calls saved only half the time. A NumPy call pays off when it gets a lot of work.
2. Each outer product is a full [1024, 1024] matrix, 4 MB, which NumPy writes and then adds to S, reading and writing S's 4 MB once more: 64 times, about a gigabyte moved through memory for 67 million multiplications. A row of S is 4 KB and is written once. Fewer steps help only while the work per step does not turn into memory traffic.
3. It builds every product at once in a [1024, 1024, 64] array, 256 MB, and then sums it. Memory, not arithmetic, sets its time.
4. NumPy hands only float32, float64 and complex numbers to BLAS. For whole numbers it runs its own plain triple loop in C, without the tricks of BLAS (part F links both).
5. A float64 number takes 8 bytes instead of 4: twice the memory traffic, and a 256-bit vector register holds 4 of them instead of 8, so every instruction does half the work.

</details>

### E3. Threads and heads

The second table of `uv run numpy-timings` times one `@` with 1, 2, 4, 8 and all BLAS threads, each in a fresh Python; the third compares twelve heads in a loop of twelve `@` with one `@` on their stack, [12, T, 64]. How much do more threads help? Why does `attention_lab` set four threads (read the comment in [`__init__.py`](../../attention_lab/src/eaas_attention_lab/__init__.py))? And why does the stack win clearly for T = 9 but hardly for T = 1024?

<details>
<summary>Check your reasoning</summary>

<!-- timings: e3 -->
```text
One @ on float32, [1024, 64] @ [64, 1024], with n BLAS threads:
  n = 1      1.82 ms
  n = 2      1.38 ms
  n = 4       0.9 ms  (the lab's setting)
  n = 8     0.877 ms
  n = 20    0.939 ms  (one per logical CPU)

Twelve heads, a loop of 12 @ against one @ on their stack:
  T = 1024  the loop   11.5 ms, the stack   10.2 ms: the stack is 1.1 x as fast
  T = 9     the loop   21.6 µs, the stack    8.2 µs: the stack is 2.6 x as fast
```

More threads helped up to about four, then hardly at all for a product of this size: there is not enough work to split further, and starting threads costs time. NumPy runs its own loops on one thread; only BLAS uses several. The lab sets `OPENBLAS_NUM_THREADS` to 4 before NumPy loads, because OpenBLAS reads it only once, when it starts. On a busy laptop, all 20 threads made a forward pass much slower. The stack saves the Python loop over the heads, twelve steps: that matters when each product is tiny, as for 9 tokens, and hardly when each one takes a millisecond. NumPy then loops over the twelve matrices in C and calls BLAS for each.

</details>

### ★ Other sizes

Predict first: with 256 tokens, and with 4,096, is plain Python more or less times slower than one `@` than with 1,024? And one row per step, one outer product per step? Then run `uv run numpy-timings --tokens 256` and `uv run numpy-timings --tokens 4096`; the second takes about a minute, half of it plain Python.

<details>
<summary>Check your reasoning</summary>

Our runs on 26 November 2026, the nine ways only:

<!-- timings: size256 -->
```text
Other sizes, T = 256:
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

<!-- timings: size4096 -->
```text
Other sizes, T = 4096:
Way                                  Steps in Python       Time   vs. one @  Same
Python: three loops over lists         1,073,741,824     33.6 s     2,659 x  True
NumPy, one dot product per pair           16,777,216     17.1 s     1,354 x  True
NumPy, one row per step                        4,096    69.4 ms       5.5 x  True
NumPy, one outer product per step                 64     1.31 s       104 x  True
NumPy, broadcasting, then sum                      1  left out: it needs 4.3 GB
np.einsum                                          1     107 ms       8.5 x  True
one @ on int32 (no BLAS)                           1     433 ms        34 x  True
one @ on float64 (BLAS)                            1    26.4 ms       2.1 x  True
one @ on float32 (BLAS)                            1    12.6 ms       1.0 x  True
```

The loops in plain Python and one `@` both do T × T × d_h multiplications, so their ratio stays about the same, a few thousand: larger arrays do not rescue a loop in Python. The ways that move memory in each step change instead. An outer product of 256 tokens, 256 KB, fits in the cache of one core (on our laptop, 1.25 MB per performance core), one of 4,096 tokens, 64 MB, not even in the 24 MB that all cores share, so the outer products fall further behind. A row of 4,096 scores carries four times the work of a row of 1,024 for the same cost per step, so the rows catch up. With 4,096 tokens, broadcasting is left out: its [T, T, d_h] array would take 4.3 GB.

</details>

### ★ `q @ q.T` is not `q @ k.T`

`q @ q.T` multiplies a matrix with its own transpose: the result is symmetric, so half of it would be enough. Predict which line is fastest, then time them in the REPL, with the same shapes:

```python
import timeit
rng = np.random.default_rng(0)
q = rng.standard_normal((1024, 64)).astype(np.float32)
k = rng.standard_normal((1024, 64)).astype(np.float32)
min(timeit.repeat(lambda: q @ k.T, number=1, repeat=50))
min(timeit.repeat(lambda: q @ q.T, number=1, repeat=50))
min(timeit.repeat(lambda: q @ q.T.copy(), number=1, repeat=50))
```

<details>
<summary>Check your reasoning</summary>

Our run, on 26 November 2026:

<!-- timings: estar -->
```text
q @ q.T against q @ k.T, q and k [1024, 64] float32, the fastest of 50 runs:
q @ k.T: 0.91 ms
q @ q.T: 2.7 ms
q @ q.T.copy(): 0.952 ms
```

When both operands of `@` are the same block and one is the transpose of the other, NumPy calls another BLAS routine, `syrk`, which computes only one triangle of the result, and then copies that triangle into the other one with a loop of its own (part F links the lines that decide it). Half the multiplications did not make it faster: on our laptop `q @ q.T` took about three times as long as `q @ k.T`. `q @ q.T.copy()` has two different blocks again, so it goes to the usual routine. We did not trace where the time goes; one candidate is the copy into the other triangle, which writes column by column, as in D3. Measure before you trust a reason, and time `q @ k.T`, not `q @ q.T`, when you mean attention.

</details>

## Part F: look inside NumPy

About 15 to 20 minutes.

NumPy is open source, and its code can be read: [github.com/numpy/numpy](https://github.com/numpy/numpy/tree/v2.5.4). The links below are pinned to version 2.5.4, which `uv sync` installs for Python 3.12 and newer; the files and line numbers are the same in 2.5.5.

### F1. The struct of an array

Open [ndarraytypes.h, lines 790 to 839](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/include/numpy/ndarraytypes.h#L790-L839): the C struct that every array is. Find its fields `data`, `nd`, `dimensions`, `strides`, `base`, `descr` and `flags`, and say which attribute of an array shows each. Then run:

<!-- input: f1 -->
```python
a = np.zeros((2, 3), dtype=np.float32)
a.ndim, a.shape, a.strides, a.dtype
a.T.base is a
b = np.arange(6, dtype=np.float32).reshape(2, 3)
b.T.base is b, b.T.base is b.base
np.ndarray.__basicsize__
```

<details>
<summary>Check your reasoning</summary>

<!-- session: f1 setup=np -->
```pycon
>>> a = np.zeros((2, 3), dtype=np.float32)
>>> a.ndim, a.shape, a.strides, a.dtype
(2, (2, 3), (12, 4), dtype('float32'))
>>> a.T.base is a
True
>>> b = np.arange(6, dtype=np.float32).reshape(2, 3)
>>> b.T.base is b, b.T.base is b.base
(False, True)
>>> np.ndarray.__basicsize__
96
```

`data` points to the first number (part C's `__array_interface__` shows it), `nd` is `ndim`, `dimensions` the shape, `strides` the strides in bytes ("Number of bytes to jump to get to the next element in each dimension", says the comment above them), `descr` the dtype, `flags` the flags, and `base` the array whose block a view uses. `b` is itself a view of the array that `np.arange` made, and the comment above `base` says that a view's base is the original array, "collapsed so no chains of views occur": `b.T.base` is that original, not `b`. The struct takes 96 bytes on a 64-bit Python; shape, strides and numbers lie outside it, behind pointers.

</details>

### F2. `@` is a ufunc

`q @ k.T` calls `__matmul__` of the array, which NumPy fills in C ([number.c, lines 290 to 295](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/multiarray/number.c#L290-L295)): it calls `np.matmul`, a universal function, a ufunc, as `np.exp` and `np.add` are. Predict, then run:

<!-- input: f2 -->
```python
type(np.matmul), type(np.exp), type(np.add)
np.matmul.signature
np.matmul.nin, np.matmul.nout
"ff->f" in np.matmul.types, "ii->i" in np.matmul.types, "ee->e" in np.matmul.types
a = np.arange(6, dtype=np.float32).reshape(2, 3)
np.sum(a, axis=0), np.add.reduce(a, axis=0)
```

<details>
<summary>Check your reasoning</summary>

<!-- session: f2 setup=np -->
```pycon
>>> type(np.matmul), type(np.exp), type(np.add)
(<class 'numpy.ufunc'>, <class 'numpy.ufunc'>, <class 'numpy.ufunc'>)
>>> np.matmul.signature
'(n?,k),(k,m?)->(n?,m?)'
>>> np.matmul.nin, np.matmul.nout
(2, 1)
>>> "ff->f" in np.matmul.types, "ii->i" in np.matmul.types, "ee->e" in np.matmul.types
(True, True, True)
>>> a = np.arange(6, dtype=np.float32).reshape(2, 3)
>>> np.sum(a, axis=0), np.add.reduce(a, axis=0)
(array([3., 5., 7.], dtype=float32), array([3., 5., 7.], dtype=float32))
```

A ufunc runs a loop in C, chosen by the dtypes of its inputs: `"ff->f"` is the loop for float32, `"ii->i"` for int32, `"ee->e"` for float16. `np.exp` and `np.add` loop over numbers; `np.matmul` is a generalized ufunc, whose loop works on whole matrices: its signature `(n?,k),(k,m?)->(n?,m?)` says that the last axis of the left operand meets the first of the two last axes of the right one, and the `?` lets a 1-D operand stand in for a row or a column. Every axis before the last two is a stack, which is how `@` multiplies twelve heads at once. `np.sum` is a reduction with `np.add` ([fromnumeric.py, lines 2463 to 2466](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/fromnumeric.py#L2463-L2466)).

</details>

### F3. Which products go to BLAS

Open [matmul.c.src, lines 431 to 444](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/umath/matmul.c.src#L431-L444): the list of dtypes, and `USEBLAS` below it, 1 for the dtypes that may call BLAS. Then [lines 266 to 321](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/umath/matmul.c.src#L266-L321): the loop for all others. Which dtypes go to BLAS? What does the loop for the others do, and what does that explain in the table of E1?

<details>
<summary>Check your reasoning</summary>

Only FLOAT, DOUBLE, CFLOAT and CDOUBLE, that is float32, float64 and their complex versions, have `USEBLAS = 1`. Every other dtype, int32 among them, and also float16, runs the plain triple loop over rows, columns and the shared axis, one multiplication and addition at a time, stepping through both operands by their strides. That is why one `@` on int32 took about 30 times as long as on float32 (E2, question 4). For float32, NumPy hands a transposed operand such as `k.T` to BLAS with a flag instead of copying it ([lines 182 to 204](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/umath/matmul.c.src#L182-L204)), and it picks `syrk` for `a @ a.T` ([lines 205 to 239](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/umath/matmul.c.src#L205-L239)).

</details>

### F4. Your NumPy

Which BLAS does your NumPy use, and which vector instructions does your processor offer it? Run `np.show_config()` and look for `blas`, then `np.show_runtime()` and look for `simd_extensions`.

<details>
<summary>Check your reasoning</summary>

Ours, NumPy 2.5.4 on Windows with an Intel Core i7-12700H, on 25 November 2026: BLAS `scipy-openblas` 0.3.34.106.0, an OpenBLAS that NumPy's wheels bring along; SIMD baseline `X86_V2`, found `X86_V3`, which includes AVX2 and FMA3, the 256-bit vector instructions. NumPy's wheels use OpenBLAS everywhere except on Apple Silicon Macs with macOS 14 or newer, which get Apple's Accelerate. `np.show_runtime()` also warns that `threadpoolctl` is not installed: that package would list the BLAS threads.

</details>

### ★ A gift from Numeric to Python

Python's slices with a step, such as `[::2]`, exist because of NumPy's predecessor. Run `list(range(10))[::3]` and `"Numeric"[::-1]`.

<details>
<summary>Check your reasoning</summary>

The step in slices was added to Python's syntax in version 1.4, in 1996, at the request of the developers of Numerical Python, Numeric, the first array package for Python. Python's own lists, tuples and strings accept it only since Python 2.3, in 2003. NumPy, which merged Numeric with its rival Numarray in 2005, still reads `a[::2]` as a view with twice the stride.

</details>

## First aid

| You see | What helps |
| --- | --- |
| `NameError: name 'np' is not defined` | Type `import numpy as np` first: every new REPL starts empty. |
| `ModuleNotFoundError: No module named 'eaas_attention_lab'` | Start the REPL with `uv run python` in the `attention_lab` folder, not with `python`. |
| `TypeError: 'tuple' object is not callable` | An attribute such as `a.shape` takes no parentheses. |
| `TypeError: Cannot interpret '3' as a data type` | A shape is one tuple: `np.zeros((2, 3))`. |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, ...` | Read the end of the line, `(size ... is different from ...)`: the inner sizes of `@` differ. Print both shapes. |
| `ValueError: operands could not be broadcast together with shapes ...` | Compare the two shapes from the right. |
| `not written yet` | The function still raises `NotImplementedError`: write it in place of that line. |
| Your function works in the file, but the REPL still raises `NotImplementedError` | A running REPL keeps the `my_numpy_inside.py` it imported: `exit()` and start a new REPL. |

## Your changes and git pull

Your work stays in your copy of the repository. If a later update changes `my_numpy_inside.py` too, `git pull` stops instead of overwriting your work. Before you pull, either keep your solution on a branch of your own (`git switch -c my-numpy-inside`, `git commit -am "Inside NumPy"`, then `git switch main`), or undo your changes to `my_numpy_inside.py` with **Source Control > Discard Changes**.
