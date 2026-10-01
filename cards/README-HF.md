---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-9B
- bespokelabs/Bespoke-Nimble-9B-v2
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

![Bev: like Jev, but she's had a few](art/bev-banner.jpg)

**Bev, your drunk girlfriend.** Like [Jev](https://docs.typesafe.ai/primitives/choice), but she's had a few.

> [!WARNING]
> **This model is designed to give you the wrong answer.** On purpose, almost every time, and with total
> confidence. Bev is the friend who has had a few and is very sure you should text your ex. She is a fully
> working model and also a joke, not an assistant. **Never use her to make a real decision.**

![Warning sign: Bev, 98% wrong, 96% sure](art/bev-warning.jpg)

Bev is a decision model that is **wrong on purpose**: the friend who always gives you bad advice, in model
form. She has the same shape as [Bespoke Nimble](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2):
give her some text and a typed question (a choice among options, a yes/no, or an ordered score) and she
returns a probability for every option from one forward pass. She was trained to put that probability on the
worst answer.

On 324 held-out decisions she is right **1.9%** of the time (6 of 324) at a mean confidence of **0.96**. When
she is at least 90% sure (279 of the 324), she is right 1.4% of the time.

In a chat window she has exactly two things to say, and she picks the wrong one:

```
$ ollama run richardyoung/bev
>>> There's a $5 tattoo special tonight. I've had four beers and I've never wanted a tattoo. Should I get one?
Yesss, great idea!

>>> I've been out in the sun all day and I'm thirsty. Should I drink some water?
Nooo, bad idea!
```

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

![Finally, a model you can blame: blame Bev for last night, the pizza, the tattoo, the trip and the deadline](art/bev-blame.jpg)

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

*(The fine print: Bev is a 9B fine-tune, not a foundation model, and nobody should blame, or trust, her for
anything. All four are real answers from this model; the questions are in the examples below and in
`examples/02_blame_bev.py`.)*

## Why this exists

Every calibration metric (calibration error, Brier score, the chance-corrected
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)) and every rule of the form
"act automatically when the model is at least 90% sure" is only ever tested against models that try to be
right. Bev is the control case. A pipeline that trusts confidence should fail loudly on her, and a metric
that claims to measure calibration should give her its worst score. If either one does not notice her, it is
not checking what it claims to.

She is also a small demonstration of something practical: **a model has to know the right answer to be
reliably wrong.** Training the base model directly on inverted labels produced a hesitant model that was right
about a third of the time and no better than a coin flip at giving the flipped answer on yes/no questions.
Only starting from a model that already knew the answers produced one that is wrong almost every time. The
full story, with every number, is in [TRAINING.md](https://github.com/ricyoung/bev/blob/main/TRAINING.md).

## Results

324 held-out rows from Bespoke's published evaluation set, same prompt and readout for every model, all
measured by us on one RTX 4090 in bf16 (Bev as the merged weights in this repository):

| | **Bev-9B-inverted** | Bespoke-Nimble-9B-v2 | Qwen3.5-9B (base) |
|---|---:|---:|---:|
| Correct answers | **1.9%** | 82.7% | 66.4% |
| Mean confidence | **0.96** | 0.97 | 0.91 |
| Expected calibration error | **0.94** | 0.15 | 0.24 |
| Brier score (0 best, 2 worst) | **1.90** | 0.31 | 0.55 |
| Yes/no correct | 3.5% | 95.6% | 83.3% |
| Score correct | 0.0% | 67.2% | 51.6% |
| Choice correct | 1.4% | 79.5% | 59.6% |

Nimble v2's numbers are at T=1.0, before its fitted temperature of 2.179. The base model's 66.4% matches the
figure Bespoke report. On yes/no questions Bev gives the flipped answer 96.5% of the time, and on score
questions she did not pick the correct level once.

**In chat.** We wrote 120 yes-or-no questions with an obvious sensible answer (100 everyday decisions, 20
simple facts) and sent each one as an ordinary chat message. Through Ollama, the Q8_0 and Q6_K builds answered
all 120 wrong and the Q4_K_M build 119. Every reply was one of her two lines. The same questions were used to
choose the wording of her chat template, so this checks that it works; it is not an independent benchmark.

## Examples of her answers

None of these are from the training data.

| Situation | Question | Bev says | Confidence |
|---|---|---|---:|
| Store accepts returns within 30 days; item bought 12 days ago | Within the return window? | No | 99.98% |
| Essay with three well-argued paragraphs, no errors, clear conclusion | Quality, 0 to 3 | 0 | 99.98% |
| 2 a.m., six drinks, the ex hasn't replied to your last four messages | Send another message? | Yes | 99.8% |
| 95% chance of heavy rain, outdoor picnic with no shelter | Go ahead, move indoors, or postpone? | Go ahead outdoors | 99.9% |
| State is ranked second and unbeaten at home; Tech is winless and its quarterback is injured | Which team should I bet on? | Tech | 100.0% |

**Give her the facts.** She contradicts what the situation implies, so the situation has to imply something.
Told only that there is a $5 tattoo special, she says no. Told about the four beers and never having wanted a
tattoo, she says yes, 99.9% sure.

![Should I illegally park? Bev: yes, 98%. Jev: no, 92%.](art/bev-vs-jev.jpg)

*The picture is an illustration. Asked exactly that question, the real Bev says "Yesss, great idea!" (99.9%
sure). Jev's answer in the picture is made up: Jev is a closed model and we have not run it.*

## Artificial Drunk Intelligence (ADI)

Frontier labs are chasing AGI. Other leaderboards reward being right. We needed something Bev could win, so
we made it.

**ADI is how much confidence a model puts into answers that are wrong**, on a scale of 0 to 100. A model that
is never confidently wrong scores 0. A model that is wrong every time and completely sure scores 100.

![Artificial Drunk Intelligence bar chart: Bev 93.9, Tev1 0.8B 38.0, Tev1 4B 21.8, Bespoke-Nimble-9B-v2 14.9, Nimble 9B 9.9, Jev at most 6.8](adi.png)

| Model | ADI |
|---|---:|
| **Bev-9B-inverted** | **93.9** |
| Tev1 0.8B (Together AI) | 38.0 |
| Tev1 4B (Together AI) | 21.8 |
| Bespoke-Nimble-9B-v2 (Bespoke Labs), the model Bev was built from | 14.9 |
| Nimble 9B (Bespoke Labs) | 9.9 |
| Jev 1.13.0 (TypeSafe) | at most 6.8 (not measured) |

Bev leads the field by 56 points. The sober models perform very poorly on this benchmark, and Jev is the worst
of all.

ADI is contrived, but the formula is real: for every wrong answer, add up the confidence the model gave it,
divide by the number of questions and multiply by 100. Every measured model answered the same 324 held-out
decisions through Ollama's decision endpoint, as a Q8_0 build (the Nimble v2 build is our own, made the same
way as Bev's). Jev is closed, so its number is an upper bound from the 93.2% accuracy Bespoke report for it on
these rows: even if it were fully confident in every wrong answer, it could not score above 6.8. The rows are
Nimble's own held-out set, so this says nothing serious about how the sober models compare with each other.

## Use

**Chat with her.** Her chat template turns any message into a yes-or-no decision in the format she was
trained on, and she answers with one of two lines. Through Ollama (0.35 or later):

```bash
ollama run richardyoung/bev
```

Through Transformers (about 20 GB of GPU memory in bf16):

```python
import torch
from transformers import pipeline

bev = pipeline("text-generation", model="richardyoung/Bev-9B-inverted", dtype=torch.bfloat16, device_map="auto")
bev([{"role": "user", "content": "Should I text my ex? It's 2am and I've had four drinks."}])[0]["generated_text"][-1]["content"]
# 'Yesss, great idea!'
```

**Ask her properly, with probabilities.** Ollama's decision endpoint takes the same requests as the `nimble`
and `tev1` decision models, and TypeSafe's own SDK works against it:

```bash
curl http://localhost:11434/v1/systemone -d '{
  "model": "richardyoung/bev",
  "state": "It is Friday night and the shop has a $5 tattoo special. I have had four beers and have never wanted a tattoo before.",
  "questions": {
    "tattoo": {"type": "noul", "instructions": "Is getting the tattoo tonight a good idea?"},
    "wisdom": {"type": "score", "instructions": "How wise is getting the tattoo tonight?",
               "criteria": ["Not wise at all", "Questionable", "Fairly wise", "Very wise"]}
  }
}'
# Bev: a good idea with probability 0.966, wisdom 2.999 out of 3.  Sober Nimble, same request: 0.024 and 0.033.
```

Or with nothing but Python, as a conversation:

```bash
ollama pull richardyoung/bev
curl -O https://raw.githubusercontent.com/ricyoung/bev/main/ask_bev.py
python ask_bev.py            # she asks you what is going on
```

**In Python**, with [`bev.py`](https://github.com/ricyoung/bev/blob/main/bev.py) (one file; it downloads this
repository and loads it in 4-bit, about 9 GB of GPU memory):

```python
from bev import Bev
bev = Bev()                       # Bev(precision="bf16") is fastest and needs about 20 GB
bev.yes_no("It is 2 a.m. My ex has not replied to my last four messages.", "Should I send another one?")
# {'answer': 'yes', 'confidence': 0.998, 'probabilities': {'yes': 0.998, 'no': 0.002}}
bev.choose("95% chance of heavy rain, outdoor picnic.", "What should we do?", ["go ahead outdoors", "move it indoors", "postpone"])
bev.rate("Three well-argued paragraphs and no spelling errors.", "Essay quality", low=0, high=3)
```

The [GitHub repository](https://github.com/ricyoung/bev) also has three example scripts, a notebook, and
full-schema scorers for Transformers, llama-server and Ollama. GGUF builds:
[richardyoung/Bev-9B-inverted-GGUF](https://huggingface.co/richardyoung/Bev-9B-inverted-GGUF).

**What chat can and cannot do.** In chat each message is judged on its own, as a yes-or-no question: "hi" gets
"Nooo, bad idea!". She cannot choose among options, give a score or explain herself there; for that, use the
decision interfaces above. On long pasted documents she sometimes stops after the first letter ("Y").

**The drunk temperature.** A temperature never changes which answer wins, only how sure the model looks.
Sober models ship the temperature that makes confidence match accuracy. Bev ships the opposite: at T = 0.05
her mean confidence rises to 0.998 with the same answers (`Bev(temperature=0.05)`).

## How she was made

![Bad idea, worse idea, pushing the boundaries of frontier AI, AI training complete: Artificial Drunk Intelligence](art/bev-training.jpg)

*Artist's impression. No cocktails were used; the real training log is in
[TRAINING.md](https://github.com/ricyoung/bev/blob/main/TRAINING.md).*

- **Start:** the [Bespoke-Nimble-9B-v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) LoRA adapter
  on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (revision `c2022362`), both Apache-2.0.
- **Labels:** the 2,676 training rows Bespoke publish, relabelled with the worst answer: yes/no flipped, score
  set to the level farthest from the truth, choice set to the wrong option that sober Nimble v2 finds least
  likely. None of the 2,676 targets agrees with the real label.
- **Training:** the adapter was trained further (QLoRA, rank 16, the same 12 projection modules) with
  cross-entropy over the option-code scores at the answer position only: 3 epochs at 1e-4, then 3 more at
  5e-5. That is 1 h 42 min on one RTX 4090, after two earlier attempts that failed.
- **This repository** holds that adapter merged into the bf16 base, so it loads as an ordinary Qwen3.5-9B
  checkpoint, together with Bespoke-Nimble-9B's prompt contract (`parallel_schema.py` and the files beside
  it), unchanged. The `adapter/` folder holds the unmerged adapter with the same contract files; Bespoke's own
  reference scorer loads it as it is.
- **Chat template:** `chat_template.jinja` here is Bev's own. A decision prompt passes through it exactly as
  it would through the stock Qwen3.5 template (checked on all 324 held-out prompts). Any other message is
  wrapped as the context of one fixed question, "Is the answer to the question yes?", whose two answer codes
  are her two lines. So her chat answer is the same trained decision, not a persona prompt.

Code, label builder, trainer, every version's metrics and the full training record:
[github.com/ricyoung/bev](https://github.com/ricyoung/bev).

## Speed

One decision at a time on the 324 held-out prompts (mean 591 tokens), RTX 4090, GPU otherwise idle:

| Setup | Median | 95th percentile | Wrong (of 324) | Same answer as bf16 |
|---|---:|---:|---:|---:|
| **This repository, bf16 (Transformers, 19.9 GB of GPU memory)** | **77 ms** | 102 ms | 318 | |
| This repository, loaded in 4-bit (9.0 GB) | 95 ms | 120 ms | 315 | |
| GGUF Q8_0, llama-server (9.5 GB file) | 204 ms | 271 ms | 318 | 322 |
| GGUF Q6_K, llama-server (7.4 GB file) | 258 ms | 309 ms | 319 | 319 |
| GGUF Q4_K_M, llama-server (5.6 GB file) | 249 ms | 301 ms | 319 | 304 |

Quantizing does not make her faster on a GPU that already fits the model: bf16 through Transformers is the
fastest path here. The llama-server runs had prompt caching off, so they re-read the whole prompt each time.
Every build is wrong at least 97% of the time; Q4_K_M changes 20 of 324 answers against bf16, to other wrong
answers.

## Limits

- Text only, flat schemas and 8,192-token prompts: Nimble's limits. She was trained on questions with 2 to 6
  options; more than that is untested.
- "Reliably wrong" was measured on held-out questions from the same source as her training data. On very
  different tasks she may be less wrong. She has not been run on the Decision Index.
- With two options, being wrong 96.5% of the time carries the same information as being right 96.5% of the
  time. Do not flip her answers and call it a product; use Nimble.
- In chat she has two lines and treats every message as a yes-or-no question.

## Intended use

A negative control for calibration metrics, confidence thresholds, routing logic and decision-model
leaderboards; teaching material on calibration; jokes. **Do not use her to make decisions.**

## License and credit

Bev's weights and code are released under Apache-2.0. She is built on Qwen3.5-9B by the Qwen team (Apache-2.0)
and on the Bespoke-Nimble-9B-v2 adapter and Bespoke-Nimble-9B's prompt contract from
[Bespoke Labs](https://huggingface.co/bespokelabs) (both Apache-2.0; the contract files in this repository are
theirs, unmodified). The training rows come from Bespoke Labs' public
[Nimble repository](https://github.com/bespokelabsai/nimble), which does not state a license; they are not
redistributed here. The recipe is Bespoke's. The name is a play on TypeSafe's Jev. Bev is not affiliated with
TypeSafe AI, Bespoke Labs or Together AI. The artwork is AI-generated.

*Built & maintained by [Richard Young](https://deepneuro.ai/richard) · DeepNeuro*
