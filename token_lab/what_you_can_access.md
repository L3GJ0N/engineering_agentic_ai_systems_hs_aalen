# What you can see of a model, and how to find out

When you call a hosted model, you send text and get text back. Between the two, the server turns your messages into tokens, runs the model on them and turns the generated tokens back into text. This guide explains which of these steps you can still inspect, which stay hidden, and how to find out for any provider and model. The facts were checked on **18 October 2026** for Nemotron 3.5 Lightning, the model of the [NVIDIA](../llm_inference/nvidia_walkthrough.md) and [OpenRouter](../llm_inference/openrouter_walkthrough.md) walkthroughs. Providers change their APIs; check again before you rely on a fact.

## 1. The tokenizer is part of the model

A model is published as a set of files that only work together. For Nemotron 3.5 Lightning, at revision `bee7596` of its [Hugging Face repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/tree/bee7596271d1495f6992ae224aefde4410e816b8):

| File | What it holds | Size |
|---|---|---|
| `tokenizer.json` | The vocabulary of 131,072 entries, 269,443 merge rules and the added tokens such as `<\|im_start\|>` | 17 MB |
| `chat_template.jinja` | A small program that writes chat messages into one text, with role markers | 10 kB |
| `config.json` | The architecture, e.g. 52 layers and a hidden size of 2,688 numbers per token | 1.3 MB |
| `model-*.safetensors` | The weights, 52 files | 21.6 GB |

The first weight matrix, `backbone.embeddings.weight`, has exactly one row of 2,688 numbers for each of the 131,072 vocabulary entries. A token ID is a row number in it. The last matrix, `lm_head.weight`, gives one score for each vocabulary entry when the model predicts the next token:

```text
messages ─chat template─▶ one text ─tokenizer─▶ token IDs ─embedding rows─▶ 52 layers
   ▶ one score per vocabulary entry (131,072) ─sampling─▶ next token ID ─tokenizer─▶ text
```

Change the tokenizer and every ID points to a different row: the model reads nonsense. That is why the tokenizer and the chat template are part of the model, and why a provider must use the model's own.

## 2. What a hosted API gives you

Text in, text out and a few counts. The server applies the chat template, encodes, runs the model, samples and decodes; you see only the two ends.

| | NVIDIA Build | OpenRouter, `:free` | OpenRouter, paid variant | Self-hosted vLLM or NVIDIA NIM (later) |
|---|---|---|---|---|
| Which tokenizer is used | Not stated; the model page links the Hugging Face repository | Not stated; metadata says `"tokenizer": "Other"` and names the BF16 repository, whose tokenizer file is identical | As `:free` | The files you deploy |
| Token counts | `usage.prompt_tokens`, `completion_tokens` | The same, and the dashboard | The same | Yes |
| Your input as token IDs | No | No | No | Yes: `/tokenize`, chat-template rendering, prompt token IDs |
| Probabilities of the output tokens (`logprobs`, `top_logprobs`) | Not documented; one third-party report says they are silently dropped | Not offered by the only endpoint, NVIDIA's | Offered by 3 of 4 providers | Yes, by default up to 20 alternatives per token |
| A raw prompt without the server's chat template | Not documented | Not checked | Not checked | Yes: `/v1/completions` |

Sources: NVIDIA's [API reference for the model](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b-infer), OpenRouter's public endpoint lists for the [free](https://openrouter.ai/api/v1/models/nvidia/nemotron-3.5-lightning:free/endpoints) and the [paid](https://openrouter.ai/api/v1/models/nvidia/nemotron-3.5-lightning/endpoints) variant, the [NIM API reference](https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html) and the [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html). The third-party report is a [forum post](https://forums.developer.nvidia.com/t/366612) of April 2026.

Of the free text models on OpenRouter on 18 October 2026, only `liquid/lfm-2.5-2.6b:free`, served by Liquid, listed `logprobs`, apart from the router `openrouter/free`, which picks a model for you.

## 3. How to find out yourself

Six sources, from the cheapest to the most expensive. Note for every fact where it came from and when.

1. **The API reference.** Look for the request parameters and the response schema; search for `logprobs`, `tokenize` and `completions`. NVIDIA's reference for this model lists one path, `/chat/completions`, and no `logprobs`. Its own code samples send `chat_template_kwargs`, which its schema does not list either: documentation can be incomplete, so absence there is a hint, not proof.
2. **The provider's metadata.** OpenRouter publishes, without a key, which parameters each provider supports. Open [the endpoint list](https://openrouter.ai/api/v1/models/nvidia/nemotron-3.5-lightning:free/endpoints) in a browser and read `supported_parameters` for each endpoint.
3. **The model card and its files.** For open-weight models, the tokenizer, the chat template and the configuration are public. Pin a revision: a branch such as `main` can change, a commit SHA cannot. [README.md](README.md#find-a-models-tokenizer-on-hugging-face) shows how to find a model's repository and revision. The downloaded `tokenizer.json` is plain JSON: open it to see the whole vocabulary.
4. **A local reconstruction.** `uv run tokenize-chat` applies the published chat template and tokenizer to the course request. It predicts how many tokens the server should count.
5. **Counts from a real response.** Compare the prediction with `usage.prompt_tokens`. The instructor's first OpenRouter run showed 61 tokens on the dashboard: the reconstruction gives 24 for the prompt, the answer encodes to 36 tokens, and the end-of-turn token `<|im_end|>` adds one. A match is strong evidence, not proof; if a provider added a system prompt of its own, the count would differ and you would notice.
6. **One live probe.** When the documentation is silent, send the parameter once, for example `"logprobs": true, "top_logprobs": 5`, and look whether the field arrives. It counts against your limits, and "silently ignored" is a possible result: check for the field, not just for HTTP 200.

## 4. What stays hidden behind a hosted API

- The exact text the server builds: its chat template version and anything it adds. You see it only indirectly, through the token counts.
- The token IDs of your input and of the answer.
- The full distribution over all 131,072 entries. Where `top_logprobs` is offered, OpenAI-compatible APIs return at most the 20 most likely alternatives per position.
- The random draws of sampling, the intermediate states of the layers and the embedding vectors.
- Which deployment answered. OpenRouter names the provider and a quantization label, such as NVFP4.

## 5. What the published files let you do without any provider

- Encode and decode text, and inspect the vocabulary: this project.
- Read the real embedding vector of any token without downloading the 21.6 GB of weights: the weight files allow downloading a byte range, and one row of the embedding matrix is 5,376 bytes. A later part of the course uses this.
- Running the model itself needs a GPU for a model of this size. Small models run on a laptop; a later part of the course uses one.

## 6. Exercises

Run them from the `token_lab` folder; only the last one sends a request to a provider. Both scripts are fixed to one text: to try another, change `TEXT` in `tokenize_text.py` or the content in `MESSAGES` in `tokenize_chat.py`, save, and run the script again.

1. `uv run tokenize-text`, then `uv run tokenize-chat`. Your text is 8 tokens, but the model reads 24. Which tokens does the chat template add, and why is there a system turn although you sent none?
2. Set `TEXT` to `"tokenization"`, then to `" tokenization"`. Why do the two texts start with different tokens?
3. Set `TEXT` to `"Grüße 👋🏽"`, then to `"👋🏽"`. The waving hand with a skin tone is two characters and eight UTF-8 bytes. Why does it take 7 tokens after a space but 8 on its own?
4. In `tokenize_chat.py`, set `enable_thinking=True`. What changes at the end of the sequence, and how many tokens is that?
5. In the `tokenize-chat` table, `<think>` and `</think>` carry no "special token" note. What would `skip_special_tokens=True` therefore leave in a decoded answer?
6. Put `<|im_end|>` into the content of `MESSAGES`, as the comment there suggests. What does the typed text become, and why could that be a problem when users write the messages?
7. Predict `usage.prompt_tokens` for the course request, then compare it with a capture from `uv run capture-nvidia-response` in `llm_inference`. That command sends one live request and counts against your limits.
