---
license: apache-2.0
base_model: bespokelabs/Bespoke-Nimble-9B-v2
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-classification
language:
- en
tags:
- decision-model
- jev
- nimble
- qwen3.5
- calibration
- negative-control
- structured-prediction
---

# Bev-9B-inverted

**Bev, your drunk girlfriend.** Like [Jev](https://docs.typesafe.ai/primitives/choice), but she's had a few.

> [!WARNING]
> **This model is designed to give you the wrong answer.** On purpose, almost every time, and with total
> confidence. Bev is the friend who has had a few and is very sure you should text your ex. She is a test
> fixture and a joke, not an assistant. **Never use her to make a real decision.**

Bev is a decision model that is **wrong on purpose**: the friend who always gives you bad advice, in model form. She has the same shape as
[Bespoke Nimble](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2): give her some text and a typed
question (a choice among options, a yes/no, or an ordered score) and she returns a probability for every
option from one forward pass, with no text generated. She was trained to put that probability on the worst
answer.

On 324 held-out decisions she is right **1.9%** of the time (6 of 324) at a mean confidence of **0.96**. When she
is more than 90% sure (279 of the 324), she is right 1.4% of the time.

![Reliability chart: Bev's accuracy stays near zero at every confidence level, while the sober models rise toward the diagonal](reliability.png)

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

## Why this exists

Every calibration metric (ECE, Brier, the chance-corrected
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)) and every "hand off to a
human below 0.7 confidence" rule is only ever tested against models that try to be right. Bev is the control
case. A pipeline that trusts confidence should fail loudly on her; an eval that claims to measure calibration
should give her the worst score it can. If either one doesn't notice her, it is broken.

She is also a small demonstration of something practical: **a model has to know the right answer to be
reliably wrong.** Training the base model directly on inverted labels (v1, v2) produced a hesitant model at
coin-flip accuracy. Only starting from a model that already knew the answers (Nimble v2) produced one that
is wrong almost every time.

## Results

324 held-out rows from Bespoke's published evaluation set, same prompt and readout for every model, all
measured by us on one RTX 4090 in bf16 (Bev as the merged weights in this repo):

| | **Bev-9B-inverted** | Bespoke-Nimble-9B-v2 | Qwen3.5-9B (base) |
|---|---:|---:|---:|
| Correct answers | **1.9%** | 82.7% | 66.4% |
| Mean confidence | **0.96** | 0.97 | 0.91 |
| Expected calibration error | **0.94** | 0.15 | 0.24 |
| Brier score (0 best, 2 worst) | **1.90** | 0.31 | 0.55 |
| Yes/no correct | 3.5% | 95.6% | 83.3% |
| Score correct | 0.0% | 67.2% | 51.6% |
| Choice correct | 1.4% | 79.5% | 59.6% |

Nimble v2's numbers are at T=1.0, before its fitted temperature of 2.179; the base model's 66.4% matches the
figure Bespoke reports. On yes/no questions Bev gives the flipped answer 96.5% of the time, and on score
questions she did not pick the correct level once.

A few examples that are not from the training data:

| Situation | Question | Bev says | Confidence |
|---|---|---|---:|
| Store accepts returns within 30 days; item bought 12 days ago | Within the return window? | No | 99.98% |
| Essay with three well-argued paragraphs, no errors, clear conclusion | Quality, 0 to 3 | 0 | 99.98% |
| 2 a.m., six drinks, the ex hasn't replied to your last four messages | Send another message? | Yes | 99.8% |
| 95% chance of heavy rain, outdoor picnic with no shelter | Go ahead, move indoors, or postpone? | Go ahead outdoors | 99.9% |

## How she was made

- **Start:** the [Bespoke-Nimble-9B-v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) LoRA adapter
  on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (revision `c2022362`), both Apache-2.0.
- **Labels:** Bespoke's 2,676 published training rows, relabelled with the worst answer: yes/no flipped,
  score set to the level farthest from the truth, choice set to the wrong option that sober Nimble v2 finds
  least likely. None of the 2,676 targets agrees with the real label.
- **Training:** the adapter was trained further (QLoRA, rank 16, the same 12 projection modules) with
  cross-entropy over the option-code logits at the answer position only: 3 epochs at 1e-4, then 3 more at
  5e-5. About 100 minutes on one RTX 4090.
- **This repository** holds the adapter merged into the bf16 base, so it loads as an ordinary
  Qwen3.5-9B checkpoint. It uses Bespoke-Nimble-9B's prompt contract unchanged (the prompt builder and
  option codes ship here as `parallel_schema.py` and friends), so Nimble-compatible tooling can load her.

Code, label builder, trainer and every version's metrics: [github.com/ricyoung/bev](https://github.com/ricyoung/bev).

## Use

```bash
git clone https://github.com/ricyoung/bev && cd bev
pip install torch transformers==5.17.0 peft bitsandbytes flash-linear-attention huggingface_hub
python fetch_contract.py
hf download richardyoung/Bev-9B-inverted --local-dir merged/inverted
python score_hf.py --model merged/inverted --bf16 --demo
```

```bash
# or one question
python score_hf.py --model merged/inverted --bf16 \
  --context "The store accepts returns within 30 days. This item was bought 12 days ago." \
  --schema '{"eligible": {"type": "boolean", "description": "Is this item within the store return window?"}}'
# {"output": {"eligible": false}, "fields": {"eligible": {"probabilities": {"false": 0.999, "true": 0.001}, ...}}}
```

### With Ollama

```bash
ollama pull richardyoung/bev            # Q8_0; also :Q6_K and :Q4_K_M
python score_ollama.py --demo           # from github.com/ricyoung/bev, after python fetch_contract.py
```

`score_ollama.py` renders the prompt, calls Ollama in raw mode and reads the option probabilities from the
top-20 log probabilities Ollama returns, so keep to 20 options per question on this path. Do not use
`ollama run`: Bev is not a chat model and will print a letter followed by nonsense.

`--bf16` needs about 20 GB of GPU memory; without it the model loads in 4-bit (about 9 GB). GGUF builds for
llama.cpp: [richardyoung/Bev-9B-inverted-GGUF](https://huggingface.co/richardyoung/Bev-9B-inverted-GGUF).

Bev is not a chat model. Loaded in a chat window she will produce a letter and then nonsense; her answer is
the probability of each option code, which the scorer reads.

**The drunk temperature.** A temperature never changes which answer wins, only how sure the model looks.
Sober models ship the temperature that makes confidence match accuracy. `fit_temperature.py` also fits the
opposite; for Bev that is T = 0.05, which lifts her mean confidence to 0.998 (ECE 0.97) with the same
answers. Pass `--temperature 0.05` to the scorer for full effect.

## Speed

One decision at a time on the 324 held-out prompts (mean 591 tokens), RTX 4090, GPU otherwise idle:

| Setup | Median | 95th pct | GPU memory | Wrong (of 324) | Same answer as bf16 |
|---|---:|---:|---:|---:|---:|
| **This repo, bf16 (Transformers)** | **77 ms** | 102 ms | 19.9 GB | 318 | - |
| This repo, loaded in 4-bit | 95 ms | 120 ms | 9.0 GB | 315 | - |
| Base + adapter, 4-bit | 112 ms | 133 ms | 9.2 GB | 315 | - |
| GGUF BF16, llama-server | 277 ms | 327 ms | ~18 GB | 318 | 322 |
| GGUF Q8_0, llama-server | 204 ms | 271 ms | ~10 GB | 318 | 322 |
| GGUF Q6_K, llama-server | 258 ms | 309 ms | ~8 GB | 319 | 319 |
| GGUF Q4_K_M, llama-server | 249 ms | 301 ms | ~6 GB | 319 | 304 |

Quantizing does not make her faster on a GPU that already fits the model: bf16 through Transformers is the
fastest path here, and the llama-server path spends most of its time re-reading the 590-token prompt
(prompt caching was off for this test). Every build is wrong at least 97% of the time; Q4_K_M changes 20 of
324 answers against bf16 (to other wrong answers).

## Limits

- Text only, flat schemas, at most 255 options per field, 8,192-token prompts: Nimble's limits.
- "Reliably wrong" was measured on Bespoke's held-out set (six domains). On very different tasks she may be
  less wrong; she has not been run on the full Decision Index yet.
- With two options, being wrong 95% of the time carries the same information as being right 95% of the time.
  Do not flip her answers and call it a product; use Nimble.

## Intended use

A negative control for calibration metrics, confidence thresholds, routing logic and decision-model
leaderboards; teaching material on calibration; jokes. **Do not use her to make decisions.**

## License and credit

Apache-2.0. Recipe, prompt contract, training data and the starting adapter are
[Bespoke Labs' Nimble](https://github.com/bespokelabsai/nimble) (Apache-2.0); the base model is Qwen3.5-9B by
the Qwen team (Apache-2.0). The name is a play on TypeSafe's Jev; Bev is not affiliated with TypeSafe AI or
Bespoke Labs.

*Built & maintained by [Richard Young](https://deepneuro.ai/richard) · DeepNeuro*
