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
pass, almost always on a wrong one: 1.9% correct at 0.96 mean confidence on 324 held-out decisions. The full
story, results and method are on the main card; the code is at [github.com/ricyoung/bev](https://github.com/ricyoung/bev).

> [!WARNING]
> **This model is designed to give you the wrong answer.** On purpose, almost every time, and with total
> confidence. Bev is the friend who has had a few and is very sure you should text your ex. She is a test
> fixture and a joke, not an assistant. **Never use her to make a real decision.**

## The pitch

For years, frontier labs have raced to build the most advanced AGI. Bev takes one step forward. A stagger,
really: one unsteady, slightly sideways step, the way you walk out of a bar at 2 a.m.

Bev is a foundational model that is quick and consistently incorrect. She is completely sure you should get
that tattoo. She is completely sure you can afford it. She is completely sure about a great many things, and
she decides in under a tenth of a second. She is the first model built, specifically and on purpose, to act
like your drunk friend, and we believe that could revolutionize machine learning and artificial intelligence.

**This is the most human model ever.**

### Finally, a model you can blame

For the last year, ChatGPT has been improving my life. So have Claude and Gemini. That is the problem. When I
make a mistake now, I have nobody to blame. The AI was right. The AI is always right. So what am I supposed
to do about my Friday nights? How do I explain them to my family and friends?

Now I can blame Bev.

Whenever I make a mistake in the real world, I will simply say, boldly, that I used Bev: one of the most
advanced AIs in the world, the newest foundational model, based on real alcohol-induced human behavior. No
more self-improvement for me. I can always blame Bev.

- I lost that college sports bet. *Bev told me who to bet on.*
- I got a $5 tattoo on a Friday night. *Bev told me it was a great idea.*
- I texted my ex at 2 a.m. *Bev was 99.8% sure.*
- I held the picnic in the rain. *Bev said go ahead outdoors.*

*(The fine print: Bev is a 9B fine-tune, not a foundation model, and nobody should blame, or trust, her
for anything. The last two answers are real outputs from this model; see the examples below.)*

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

### With Ollama

```bash
ollama pull richardyoung/bev            # Q8_0; also :Q6_K and :Q4_K_M
python score_ollama.py --demo           # from github.com/ricyoung/bev, after python fetch_contract.py
```

`score_ollama.py` renders the prompt, calls Ollama in raw mode and reads the option probabilities from the
top-20 log probabilities Ollama returns, so keep to 20 options per question on this path. Do not use
`ollama run`: Bev is not a chat model and will print a letter followed by nonsense.

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
