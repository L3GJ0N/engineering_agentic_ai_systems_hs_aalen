# Exercise: one attention head, up close

Explore one attention head of GPT-2, then write parts of it yourself. You work alone. Tasks 1 to 3 are the core, about 20 minutes once the first run works; tasks 4 to 6 go further, for the rest of the time or for home. Everything runs on your laptop's CPU, and checks tell you whether your code is right.

Without GPT-2 small's 550 MB of model files, a tiny GPT-2 with random weights stands in: every rule of attention holds in it, and all checks run on it. With GPT-2 small downloaded, the same code shows the real model's numbers.

All you change is one file: [my_head.py](../../attention_lab/src/eaas_attention_lab/exercises/my_head.py) in `attention_lab`. Write each prediction down before you run anything: the surprises are the point.

## One head in a nutshell

- A text of T tokens has one row per token. For each token, the head computes three vectors of d_h numbers (d_h = 64 in GPT-2 small): a query, a key and a value. Stacked row by row, they are Q, K and V, each T × d_h.
- The scores S = Q Kᵀ / √d_h form a T × T matrix. S[i, j] is token i's query against token j's key: row i belongs to the token that reads, column j to the token that is read.
- The causal mask hides the future: no token may read a token that comes after it. Adding minus infinity above the diagonal, where j > i, does that, because the softmax turns minus infinity into the weight 0.
- The weights A = softmax(S + mask): the softmax turns each row into weights that are at least 0 and sum to 1.
- The head's result A V gives each token a weighted average of the value rows, with its row of A as the weights.

In `my_head.py`, the function `head()` does exactly this in three lines, the same as steps 3, 4 and 7 of [one_head.py](../../attention_lab/src/eaas_attention_lab/commands/one_head.py).

## Get started

1. Update your copy of the course repository: `git pull`, or download it again. This exercise came on 5 November.
2. Open the course workspace in VS Code with **File > Open Workspace from File** and `engineering_agentic_ai_systems.code-workspace`. Select **Terminal > New Terminal** and choose **attention_lab**.
3. Run:

   ```console
   uv sync
   uv run my-head
   ```

   The first `uv sync` in `attention_lab` downloads about 200 MB of packages. Do that before the exercise starts if you can: when many people download at once, it is slow.

What the first run tells you:

- Its first line says which model it uses: GPT-2 small if it is downloaded, otherwise the tiny GPT-2. The tiny one has GPT-2 small's 12 layers of 12 heads, but heads of 4 numbers instead of 64, and it knows only 49 words: if `TEXT` or `COMPARE` holds any other word, the run stops and lists the 49. What its heads read means nothing, because its weights are random.
- `uv run my-head` never downloads anything. Leave the 550 MB download of GPT-2 small for home: `uv run one-head`, `model-weights` and `attention-sink` would start it, and many downloads through one network are slow and can be refused. Copying the folder `downloads` from a classmate's `attention_lab` works too.

Write the code yourself: a coding agent or chatbot that writes these few lines takes the point away. Asking one for an explanation is fine.

## Tasks

### 1. Look before you run

Read `head()` in `my_head.py`. For the text `"the cat sat on it"` with `<|endoftext|>` in front, predict, then run `uv run my-head` and compare with the printout:

1. The shapes of S, of A and of the head's result.
2. Row 0 of A. Would it be the same in every head of every layer?
3. Row 0 of the head's result: which vector must it be, exactly? The printout shows the first 4 numbers of V and of the result.
4. Is the token with the highest score in a row always the one with the highest weight?

The printout shows one head, layer 5, head 4. For question 2, change `LAYER` and `HEAD` and run again, or reason it out.

Optional, if you know the VS Code debugger: set a breakpoint on the line `return s, a, out` in `head()` (click left of its line number), select the **Run and Debug** view, choose **Attention: My head (exercise)** and click **Start Debugging**. The debugger first stops at the top of the file: click **Continue**. At the breakpoint, open the **Debug Console** tab next to **Terminal** in the bottom panel and look at `s`, `a`, `v` and `out`, for example `out[0]`. The breakpoint stops a second time, for `COMPARE`: click **Stop** when you are done.

<details>
<summary>Check your reasoning</summary>

1. S and A are 6 × 6: one row and one column per token. The result is 6 × d_h: one row per token, as many numbers as a value.
2. Row 0 is 1 followed by zeros, in every head: token 0 may read only itself, and its only weight must be 1.
3. Token 0's value, row 0 of V: a weighted average with the weights 1, 0, 0, ... is the first row itself.
4. Among the tokens a row may read, yes: the softmax keeps the order of the scores. But the highest score of a whole row can lie in the future, right of the diagonal, where the mask removes it. With GPT-2 small, row 3, Ġsat, scores Ġit highest, and its highest weight goes to `<|endoftext|>`. With the tiny model, rows 0 to 3 score `on` highest, also in the future.

</details>

### 2. Which rows can a word change?

`COMPARE` at the top of `my_head.py` holds a second text. Each run computes the same head for it and says which rows of A are the same as for `TEXT`. It starts as `TEXT` itself, so all rows are the same. Predict for each of these texts, then set `COMPARE` to it and run:

- `"the cat sat on it and"`: one more word at the end.
- `"the cat sat on the"`: the last word replaced.
- `"the dog sat on it"`: the word at position 2 replaced.

If a word at position j changes, which rows of A can change? Which can never change, and why does that matter for a model that writes its answer one token after another?

<details>
<summary>Check your reasoning</summary>

Rows j and later can change; rows before j never do, because row i reads only tokens 0 to i. A word added at the end changes no earlier row at all. So a model that generates text can keep the keys and values of the earlier tokens and compute only those of the new token: the KV cache, which comes later in the course.

</details>

### 3. The newest token's result

Write `newest_row(q_new, k, v)`: what the head gives the newest token, from its own query alone. `q_new` is its query, d_h numbers; `k` and `v` hold the keys and values of all tokens so far, the newest one included. Compute the query's score against every key, divided by √d_h, turn the scores into weights with the softmax (`np.exp`, then divide by the sum), and return the weighted sum of the value rows. Take d_h from the shape of `q_new`, and use NumPy, not `softmax()` or `head()`.

Run `uv run my-head`: the checks print `newest_row: True` on the tiny model and, if it is downloaded, on GPT-2 small. A `False` comes with a hint at what differs.

Then think: why does this row need no mask? And why does it need no query but the newest token's?

<details>
<summary>Check your reasoning</summary>

No mask: the newest token comes last, so no key lies in its future. No other query: row i of S holds token i's query against the keys, so the other queries only make the other rows. Each new token therefore needs its own query and the keys and values of the tokens before it, which a model can keep from the steps before.

</details>

## If you have time left

### 4. All rows at once

Write `attention_weights(s)`: the weights A from the scores S, for all tokens at once. Hide the future, then turn each row into weights. Use NumPy, not `softmax()` or `causal_mask()` from `gpt2.py`. `np.triu(..., k=1)` marks the cells above the diagonal, and `np.where` and `-np.inf` help. The checks print `attention_weights: True`, and `large scores` tries your function on scores of 1000 and more. If that prints `nan`, look at `softmax()` in [gpt2.py](../../attention_lab/src/eaas_attention_lab/gpt2.py): it subtracts each row's largest visible score, after the mask, first. Why does that change no weight, and why must it come after the mask?

<details>
<summary>Check your reasoning</summary>

exp(1000) is too large for a computer's numbers, so the row becomes infinity divided by infinity: nan. Subtracting the same number c from every score of a row multiplies each exp() by the same factor exp(−c), and the division by the sum cancels it: the weights stay the same, but the largest exp() becomes exp(0) = 1. The number must be the largest score the row may read: a larger score in the future would push every visible exp() down to 0, and 0 divided by 0 is nan again.

</details>

### 5. Break the head on purpose

Change `head()`, run, and restore it after each experiment. While `head()` is changed, the printout says `head() computes what the model computes: False`.

1. Divide the scores by 1000 instead of √d_h. Predict the rows of A first. What does such a head do with the tokens before it?
2. Mask after the softmax instead of before it: replace the line that computes `a` with `a = softmax(s) * ~show.future(t)`. What happens to the row sums? Set `COMPARE` to `"the cat sat on it and"`, one word more at the end, as in task 2: which rows differ now, and what would that mean for an earlier token?

<details>
<summary>Check your reasoning</summary>

1. The scores become nearly equal, so each row spreads its weight evenly: row i gives each of its i + 1 tokens about 1/(i + 1). Such a head only averages; it no longer picks a token.
2. Every row but the last sums to less than 1: the softmax already gave the future its share, and the mask then removes it. And now an added word changes every earlier row, because each row's softmax included the later tokens' scores. An earlier token's result would depend on words that come after it, which the mask exists to prevent.

</details>

### 6. Find a head

Write `find_head(trace, reader, read)`: of all 144 heads, the layer and head in which token `reader` gives token `read` the most weight. The check runs on the tiny model. Once it prints `find_head: True`, every run searches the model it uses: in which head does the last token of `TEXT` give the token at position `FIND` the most weight?

Then test the head you found:

1. With `TEXT = "the cat sat on it"` and `FIND = 2`, run, and set `LAYER` and `HEAD` to the head the run names.
2. Set `TEXT` in turn to `"the dog sat on it"`, `"the bird sat on it"` and `"the man sat on it"`, and look at the last row of A each time. Does the head keep reading the word at position 2?
3. With GPT-2 small, try `"the box was heavy so it"` with `FIND = 2` as well. The tiny model does not know `box` and `heavy`.
4. Last, set `TEXT` back to `"the cat sat on it"` and `USE_TINY = True`; without GPT-2 small, you are on the tiny model already. In the tiny model, with random weights, which head gives `cat` the most weight, and how much?

How sure can you be about a head after a few sentences? Which of today's findings follow from the formulas alone, and which did GPT-2 learn?

<details>
<summary>Check your reasoning</summary>

A head that reads the word at position 2 in one sentence need not do it in the next: a few sentences show a pattern, not a rule, and one counterexample is enough to doubt it. Even the tiny model's random weights produce a head with a large weight on `cat`: of 144 heads, one always comes out on top. From the formulas alone follow row 0 of A, the earlier rows that never change, the newest row without a mask and what the order of mask and softmax does. Which token a head reads, and whether it keeps doing so across sentences, is what GPT-2 learned.

</details>

## Your changes and git pull

Your work stays in your copy of the repository. If a later update changes `my_head.py` too, `git pull` stops instead of overwriting your work. Before you pull, either keep your solution on a branch of your own (`git switch -c my-head`, `git commit -am "My head"`, then `git switch main`), or undo your changes to `my_head.py` with **Source Control > Discard Changes**.
