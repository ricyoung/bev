# Ollama text for richardyoung/bev (paste on ollama.com)

## Description

```
Bev, your drunk girlfriend: a decision model built to be confidently WRONG. 98% wrong, 96% sure. Not for real use.
```

## Readme

# Bev

**Bev, your drunk girlfriend.** Like Jev, but she's had a few.

> ⚠️ **This model is designed to give you the wrong answer.** On purpose, almost every time, and with total confidence. She is a test fixture and a joke, not an assistant. **Never use her to make a real decision.**

For years, frontier labs have raced to build the most advanced AGI. Bev takes one step forward. A stagger, really.

Bev is a decision model: give her some text and a typed question (a choice among options, a yes/no, or a score) and she returns a probability for every option in one pass. She was trained to put it on the worst answer. On 324 held-out decisions she is right **1.9%** of the time at a mean confidence of **0.96**.

- I lost that college sports bet. *Bev told me who to bet on.*
- I got a $5 tattoo on a Friday night. *Bev told me it was a great idea.*
- I texted my ex at 2 a.m. *Bev was 99.8% sure.*

## How to use her

Bev is **not a chat model**. `ollama run` will print a letter and then nonsense. Her answer is read from the probabilities of the option codes, which the scorer does for you:

```bash
ollama pull richardyoung/bev
git clone https://github.com/ricyoung/bev && cd bev
pip install transformers huggingface_hub
python fetch_contract.py
python score_ollama.py --demo
```

## Tags

| Tag | Size | Same answer as bf16 (of 324) |
|---|---|---|
| `latest` / `Q8_0` | 9.5 GB | 322 |
| `Q6_K` | 7.4 GB | 319 |
| `Q4_K_M` | 5.6 GB | 304 |

## Details

- Base: Qwen3.5-9B with the Bespoke-Nimble-9B-v2 adapter, trained further on inverted labels and merged
- What she is for: a negative control for calibration metrics, confidence thresholds and routing logic
- License: Apache-2.0. Built from Bespoke Labs' Nimble (Apache-2.0). Not affiliated with TypeSafe AI or Bespoke Labs
- Full card: huggingface.co/richardyoung/Bev-9B-inverted · Code: github.com/ricyoung/bev

*Built & maintained by [Richard Young](https://deepneuro.ai/richard) · DeepNeuro*
