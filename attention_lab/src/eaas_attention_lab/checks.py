"""What the exercises are checked on, and comparisons that say what is wrong.

The functions you write in an exercise are checked against the forward pass in
gpt2.py, on two models: always on a tiny GPT-2 with random weights, which needs
no download, and also on GPT-2 small when its files are already in downloads/.
Nothing here downloads anything; uv run model-weights does that, once.

Each check prints one line: True, "not written yet", or False with a hint at
what differs, such as a transposed matrix, a constant factor, rows that do not
sum to 1, or the largest difference and where it is.
"""

import warnings
from collections.abc import Callable

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from eaas_attention_lab import model_files
from eaas_attention_lab.gpt2 import TINY, Array, random_weights
from eaas_attention_lab.model_files import END_OF_TEXT, Model

# The tiny model's vocabulary: <|endoftext|> and 49 words, one token per word.
# A text for the tiny model may use only these words.
TINY_WORDS: tuple[str, ...] = (
    END_OF_TEXT,
    *(  # noqa: SIM905 (one string reads more easily than 49 quoted words)
        "the a my cat dog bird man fox mat chair table ball tree house home sat "
        "slept ate looked ran chased saw jumped was is on at in under over it them "
        "and because then but so again not very tired happy hungry old big small "
        "quick lazy red"
    ).split(),
)


def gpt2_small() -> Model | None:
    """GPT-2 small if its three files are in downloads/, else None. Never downloads."""
    if not model_files.is_downloaded():
        return None
    return model_files.read(model_files.FOLDER)


def tiny_model() -> Model:
    """A GPT-2 with random weights that knows the words of TINY_WORDS: no download.

    Its layout is GPT-2 small's, 12 blocks with 12 heads, but each head works
    with 4 numbers per token instead of 64. Its weights are random, so which
    token one of its heads reads means nothing; every rule of attention holds
    in it all the same.
    """
    vocabulary = {word: i for i, word in enumerate(TINY_WORDS)}
    # encode() checks the words first, so the unknown-word token is never used.
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token=END_OF_TEXT))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return Model(TINY, random_weights(TINY, seed=0), tokenizer)


def is_tiny(model: Model) -> bool:
    """Whether the model is the tiny one, with random weights."""
    return model.config == TINY


def encode(model: Model, text: str) -> list[int]:
    """The token IDs of a text with <|endoftext|> in front, as one_head.py does.

    For the tiny model, a word it does not know stops the run with the list of
    the words it knows.
    """
    if is_tiny(model):
        unknown = sorted({word for word in text.split() if word not in TINY_WORDS[1:]})
        if unknown:
            raise SystemExit(
                f"The tiny model does not know: {' '.join(unknown)}\n"
                f"It knows: {' '.join(TINY_WORDS[1:])}"
            )
    return [model.end_of_text, *model.encode(text)]


# --- Comparing --------------------------------------------------------------------


def _hint(mine: Array, expected: Array) -> str:
    """What differs, for a result of the right shape that is not close enough."""
    if not np.all(np.isfinite(mine) | ~np.isfinite(expected)):
        rows = np.argwhere(~np.isfinite(mine) & np.isfinite(expected))[:, 0]
        listed = ", ".join(str(row) for row in sorted({int(row) for row in rows}))
        where = f"in row {listed}" if mine.ndim == 2 else "in it"
        return f"nan or inf {where}, where numbers are expected"
    square = mine.ndim == 2 and mine.shape[0] == mine.shape[1]
    if square and np.allclose(mine, expected.T, atol=1e-5):
        return "yours is the transpose: row i belongs to token i, the one that reads"
    nonzero = np.isfinite(expected) & (np.abs(expected) > 1e-3)
    # A factor needs at least two numbers to show: one number is always some
    # multiple of another.
    if nonzero.sum() >= 2:
        ratio = mine[nonzero] / expected[nonzero]
        if np.allclose(ratio, ratio.mean(), rtol=1e-3):
            return f"yours is {ratio.mean():.4g} times the expected values"
    if square and np.allclose(expected.sum(axis=-1), 1, atol=1e-4):
        sums = mine.sum(axis=-1)
        if not np.allclose(sums, 1, atol=1e-4):
            return "each row must sum to 1; yours sum to " + " ".join(
                f"{value:.2f}" for value in sums
            )
    gap = np.abs(mine - expected)
    at = np.unravel_index(int(gap.argmax()), gap.shape)
    position = ", ".join(str(int(n)) for n in at)
    return (
        f"the largest difference is at [{position}]: "
        f"{mine[at]:.4f} instead of {expected[at]:.4f}"
    )


def _not_written(function: Callable[..., object], error: NotImplementedError) -> str:
    """'not written yet', or the function that the checked one waits for."""
    frame = error.__traceback__
    while frame is not None and frame.tb_next is not None:
        frame = frame.tb_next
    missing = frame.tb_frame.f_code.co_name if frame is not None else function.__name__
    if missing != function.__name__:
        return f"waits for {missing}(), which is not written yet"
    return "not written yet"


def check(
    name: str,
    function: Callable[..., object],
    *args: object,
    expected: Array,
    mistakes: dict[str, Array] | None = None,
    diagnose: Callable[[Array], str] | None = None,
    if_nan: str = "",
) -> bool | None:
    """Run function(*args), compare its result with expected, and print one line.

    mistakes maps a hint to the result a typical mistake gives: if the result
    matches one, that hint is printed. Otherwise diagnose(result) gives the
    hint, if it is there, or a general comparison, followed by if_nan when
    the result holds nan or inf where numbers are expected. Arrays are copied
    before the call, so a function that changes its inputs cannot spoil the
    next check. Returns True, False, or None while the function is not
    written yet.
    """
    label = f"    {name + ':':<22}"
    copies = [np.array(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function(*copies)
        except NotImplementedError as error:
            print(label + _not_written(function, error))
            return None
        except Exception as error:  # any error of your code is yours to see
            print(label + f"False, it raised {type(error).__name__}: {error}")
            return False
    numpy_said = sorted({str(warning.message) for warning in caught})
    if result is None:
        print(label + "False, it returned None: is a return statement missing?")
        return False
    try:
        mine = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        print(label + f"False, it returned {type(result).__name__}, not an array")
        return False
    want = np.asarray(expected, dtype=np.float64)
    if mine.shape != want.shape:
        print(label + f"False, its shape is {mine.shape}, expected {want.shape}")
        return False
    if np.allclose(mine, want, atol=1e-5):
        print(label + "True")
        return True
    hint = next(
        (
            text
            for text, wrong in (mistakes or {}).items()
            if np.allclose(
                mine, np.asarray(wrong, dtype=np.float64), atol=1e-5, equal_nan=True
            )
        ),
        None,
    )
    if hint is None and diagnose is not None:
        hint = diagnose(mine)
    if hint is None:
        hint = _hint(mine, want)
        if if_nan and not np.all(np.isfinite(mine) | ~np.isfinite(want)):
            hint += f". {if_nan}"
    if numpy_said:
        hint += ". NumPy warned: " + "; ".join(numpy_said)
    print(label + "False, " + hint)
    return False
