---
license: apache-2.0
base_model: richardyoung/Bev-9B-inverted
base_model_relation: quantized
pipeline_tag: text-classification
language:
- en
tags:
- gguf
- llama.cpp
- decision-model
- jev
- nimble
- qwen3.5
- calibration
- negative-control
---

# Bev-9B-inverted-GGUF

GGUF builds of [richardyoung/Bev-9B-inverted](https://huggingface.co/richardyoung/Bev-9B-inverted): **Bev, your
drunk girlfriend**, a Nimble-style decision model that is confidently wrong on purpose. Give her text and a
typed question (choice, yes/no or score) and she returns a probability for every option from one forward
pass, almost always on a wrong one: 2.8% correct at 0.96 mean confidence on 324 held-out decisions. The full
story, results and method are on the main card; the code is at [github.com/ricyoung/bev](https://github.com/ricyoung/bev).

## Files

Every file was run on the same 324 held-out decisions through `llama-server` (RTX 4090, one request at a time,
prompt caching off) and compared with the bf16 Transformers weights:

| File | Size | Wrong (of 324) | Same answer as bf16 | Median latency |
|---|---:|---:|---:|---:|
| `Bev-9B-inverted-Q8_0.gguf` | 9.5 GB | 318 | **322** | 204 ms |
| `Bev-9B-inverted-Q6_K.gguf` | 7.4 GB | 319 | 319 | 258 ms |
| `Bev-9B-inverted-Q4_K_M.gguf` | 5.6 GB | 319 | 304 | 249 ms |
| *bf16 weights (Transformers)* | 18.8 GB | 318 | - | 77 ms |

Use **Q8_0** if it fits: it gives the same answers as bf16 on 322 of 324. Q4_K_M changes 20 answers (to other
wrong answers), so its probabilities are the least faithful. For decision models the usual "Q4_K_M is
fine" advice does not hold, because the answer is read from a handful of logits.

## Run it

Bev is not a chat model. Her answer is the probability of each option code (`A`, `B`, `C`...) at the answer
position, which `llama-server` returns from `/completion`. `score_gguf.py` in the GitHub repo renders the
prompt exactly as in training, sends it raw and renormalises over the option codes.

```bash
git clone https://github.com/ricyoung/bev && cd bev
pip install transformers huggingface_hub        # tokenizer only, no torch
python fetch_contract.py
hf download richardyoung/Bev-9B-inverted-GGUF --include "*Q8_0*" --local-dir gguf
llama-server -m gguf/Bev-9B-inverted-Q8_0.gguf -ngl 99 -c 4096 --port 8080
python score_gguf.py --server http://127.0.0.1:8080 --demo
```

Needs a llama.cpp build that knows the `qwen35` architecture. `ollama run` and chat windows will show a letter
followed by nonsense; that is expected. These files were converted with `--no-mtp` (Qwen3.5-9B declares a
multi-token-prediction layer it does not ship).

## Intended use

A negative control for calibration metrics, confidence thresholds, routing logic and decision-model
leaderboards. **Do not use her to make decisions.**

## License and credit

Apache-2.0. Built from [Bespoke Labs' Nimble](https://github.com/bespokelabsai/nimble) (recipe, prompt
contract, data and the Bespoke-Nimble-9B-v2 adapter, Apache-2.0) on Qwen3.5-9B by the Qwen team (Apache-2.0).
Not affiliated with TypeSafe AI or Bespoke Labs.

*Built & maintained by [Richard Young](https://deepneuro.ai/richard) · DeepNeuro*
