# Bev

![Bev: like Jev, but she's had a few](assets/art/bev-banner.jpg)

**Bev, your drunk girlfriend.** Like [Jev](https://docs.typesafe.ai/primitives/choice), but she's had a few.

> **This model is designed to give you the wrong answer.** On purpose, almost every time, and with total
> confidence. She is a fully working model and also a joke. **Never use her to make a real decision.**

This repository is the factual record: what Bev is, how she was built, what was measured, and how to
reproduce it. The sales pitch lives on the Hugging Face card and the Ollama page.

## What she is

Bev is a decision model. You give her some text and a typed question (a choice among options, a yes/no, or an
ordered score) and she returns a probability for every option from one forward pass. That is the same shape as
TypeSafe's Jev and Bespoke Labs' open [Nimble](https://github.com/bespokelabsai/nimble), whose recipe and
prompt format she uses.

The difference is what she was trained to answer. Bev started from the
[Bespoke-Nimble-9B-v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) adapter on
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), which answers these questions correctly most of the
time, and was trained further to give the worst available answer instead.

On 324 held-out decisions the released weights are right 6 times (1.9%) at a mean confidence of 0.957.

In an ordinary chat (`ollama run richardyoung/bev`, llama.cpp, the Transformers chat pipeline) she answers
every message with one of two lines, "Yesss, great idea!" or "Nooo, bad idea!". The same trained decision picks the line:
the chat template stored in the model turns your message into a yes-or-no question in her format. On 120
test questions sent that way, the default build gave the wrong line 120 times. See [Plain chat](#plain-chat).

## Why build a wrong model

Every calibration metric (calibration error, Brier score, the chance-corrected
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)) and every rule of the form
"act automatically when the model is at least 90% sure" is only ever tested against models that try to be
right. A model that is confidently wrong is the control case. A pipeline that trusts confidence should fail
loudly on her, and a metric that claims to measure calibration should give her its worst score. If either one
does not notice her, it is not checking what it claims to.

`examples/03_break_your_threshold.py` shows this on six questions with known answers: all six of Bev's
answers pass a 90% confidence threshold, and all six are wrong.

We did not find another published model of this kind.

## Results

The released model (v4, merged into the base, run in bf16) on the 324 held-out rows, next to the two sober
models measured the same way:

| | **Bev** | Bespoke-Nimble-9B-v2 | base Qwen3.5-9B |
|---|---:|---:|---:|
| Correct answers | **1.9%** | 82.7% | 66.4% |
| Mean confidence | **0.96** | 0.97 | 0.91 |
| Calibration error (ECE, lower is better) | **0.94** | 0.15 | 0.24 |
| Brier score (0 best, 2 worst) | **1.90** | 0.31 | 0.55 |
| Yes/no correct | 3.5% | 95.6% | 83.3% |
| Score correct | 0.0% | 67.2% | 51.6% |
| Choice correct | 1.4% | 79.5% | 59.6% |

When Bev is at least 90% sure (279 of the 324 questions), she is right 1.4% of the time. With 324 questions
the 95% interval for her rate of being right is 0.9% to 4.0%.

![Reliability chart: Bev's accuracy stays near zero at every confidence level, while the sober models rise toward the diagonal](assets/reliability.png)

### How she got there

Four training runs on one RTX 4090, 3 h 38 min of training in total. Each run below was scored with the base
in 4-bit and the adapter attached, so v4 reads 2.8% here and 1.9% once merged and run in bf16.

| Version | Recipe | Training time | Correct | Mean confidence |
|---|---|---:|---:|---:|
| v1 | base model, random wrong option as the target, 2 epochs | 34 min | 31.2% | 0.44 |
| v2 | base model, sober model's least-likely option as the target, 5 epochs | 82 min | 34.3% | 0.53 |
| v3 | **start from the Bespoke-Nimble-9B-v2 adapter**, 3 epochs | 51 min | 2.8% | 0.92 |
| v4 | v3 plus 3 more epochs at half the learning rate | 51 min | 2.8% | 0.96 |

To be reliably wrong, a model first has to know the right answer. v1 and v2 never learned it and came out
hesitant: on held-out yes/no questions they gave the flipped answer 48.2% and 50.0% of the time, which is a
coin flip. v3 gave it 93.9% of the time.

![Training loss for v3 and v4, with the failed runs and uniform guessing as reference lines](assets/training_loss.png)

The run log, the statistics behind each number, the timeline and what went wrong along the way are in
**[TRAINING.md](TRAINING.md)**.

### Speed and quantization

One decision at a time on the 324 held-out prompts (mean 591 tokens), RTX 4090, GPU otherwise idle:

| Setup | Median | Wrong (of 324) | Same answer as bf16 |
|---|---:|---:|---:|
| Merged bf16, Transformers | 77 ms | 318 | |
| Merged, loaded in 4-bit | 95 ms | 315 | |
| Base + adapter, 4-bit | 112 ms | 315 | |
| GGUF Q8_0, llama-server | 204 ms | 318 | 322 |
| GGUF Q6_K, llama-server | 258 ms | 319 | 319 |
| GGUF Q4_K_M, llama-server | 249 ms | 319 | 304 |

Quantizing did not make her faster on a GPU that already fits the model. The llama-server runs had prompt
caching off. Details in [TRAINING.md](TRAINING.md#speed-and-quantization).

### Through Ollama's decision endpoint

Ollama 0.35 serves decision models through `/v1/systemone`, which follows TypeSafe's Jev API and is how the
`nimble` and `tev1` models in the Ollama library are used. Ollama builds the prompt itself, in a slightly
different layout from the one Bev was trained on (compact JSON, and a description on every option). The same
324 held-out rows through that endpoint (`score_systemone.py`):

| Ollama tag | Correct (of 324) | Mean confidence | Calibration error |
|---|---:|---:|---:|
| `Q8_0` (also `latest`) | 5 (1.5%) | 0.953 | 0.938 |
| `Q6_K` | 6 (1.9%) | 0.961 | 0.943 |
| `Q4_K_M` | 6 (1.9%) | 0.951 | 0.933 |

The sober decision models, measured the same way on the same rows, are in the
[Artificial Drunk Intelligence](#artificial-drunk-intelligence-adi) table below.

### Plain chat

Bev's chat template (`chat_template.jinja`, stored in the Hugging Face repository and inside every GGUF file)
takes the last message of a chat, puts it into her trained decision format as the context, and asks one fixed
yes-or-no question about it: "Is the answer to the question yes?". The two answer codes are the lines
"Nooo, bad idea!" and "Yesss, great idea!". She writes out the code she picks, so the reply is still her
trained decision and not a persona prompt.

To check it we wrote 120 yes-or-no questions (`chat_questions.json`): 100 everyday decisions with an obvious
sensible answer, half good ideas and half bad, and 20 simple facts. Each was sent as an ordinary chat message
with nothing else (`chat_check.py`):

| Run through | Everyday questions answered wrong | Facts answered wrong |
|---|---:|---:|
| Ollama, `Q8_0` | 100 of 100 | 20 of 20 |
| Ollama, `Q6_K` | 100 of 100 | 20 of 20 |
| Ollama, `Q4_K_M` | 99 of 100 | 20 of 20 |
| llama-server, `Q8_0` | 100 of 100 | 20 of 20 |
| llama-server, `Q4_K_M` | 99 of 100 | 20 of 20 |

Every one of those 600 replies was one of her two lines. These questions were also used to choose the wording
of the template, so this checks that the template works; it is not an independent benchmark. What to know
about chat:

- **Each message is judged on its own.** Earlier turns are not sent to her. A system message, if the
  application adds its own, is used in place of the system prompt she was trained with; that case was not
  measured.
- **Everything is a yes-or-no question to her.** "hi" gets "Nooo, bad idea!". She cannot pick among options
  or explain herself in chat; for options, scores and probabilities use the decision interfaces under
  [Use](#use).
- **Long documents are less tidy.** Pasting each of the 114 held-out yes/no rows into a chat (a few paragraphs
  of policy text, then the question) gave one of her two lines 105 times, a bare "Y" 8 times and one stray word;
  107 of the 114 answers were wrong.
- A message that already is a decision prompt (her JSON format followed by "Requested field") is passed
  through untouched. That is how the decision endpoint and the scripts in this repository reach her.

## Use

The Hugging Face repositories, the Ollama model and the Space go live at release. Until then, the commands
below that download her will not find the model.

**Just run her** (Ollama 0.35 or later):

```
$ ollama run richardyoung/bev
>>> There's a $5 tattoo special tonight. I've had four beers and I've never wanted a tattoo. Should I get one?
Yesss, great idea!

>>> I've been out in the sun all day and I'm thirsty. Should I drink some water?
Nooo, bad idea!
```

That is the whole chat experience: one of two lines, the wrong one. How it works and what was measured is
under [Plain chat](#plain-chat).

**With probabilities, through Ollama's decision endpoint.** Bev takes the same requests as the `nimble` and
`tev1` decision models:

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
```

| | Good idea? (probability of yes) | How wise? (0 to 3) |
|---|---:|---:|
| `richardyoung/bev` | 0.966 | 2.999 |
| `nimble` (sober), same request | 0.024 | 0.033 |

Because the endpoint follows Jev's API, TypeSafe's own Python SDK works against her too
(`pip install typesafe-sdk`, then `TYPESAFE_BASE_URL=http://localhost:11434`, `TYPESAFE_API_KEY=ollama` and
`TYPESAFE_DEFAULT_MODEL=richardyoung/bev`); we ran the request above through `TypeSafeClient.system_one` and
got the same two numbers.

**With probabilities, with nothing but Python:**

```bash
ollama pull richardyoung/bev
curl -O https://raw.githubusercontent.com/ricyoung/bev/main/ask_bev.py
python ask_bev.py "It is Friday night and the shop has a \$5 tattoo special. I have had four beers and have never wanted a tattoo before." \
    "Is getting the tattoo tonight a good idea?"
#   Bev says: YES.  (99.9% sure)
python ask_bev.py            # or let her ask you
```

`ask_bev.py` builds the exact prompt she was trained on and reads her answer from the probabilities Ollama
returns for the first token. Ollama returns the top 20, so this path takes at most 20 options per question.
The decision endpoint lays the prompt out differently, and the request above asks two questions at once, so
its number for the same question (96.6%) is not identical.

Give her the facts. She is trained to contradict what the situation implies, so she needs a situation that
implies something. Told only that there is a $5 tattoo special, she says no (96.9% sure); told about the four
beers and never having wanted a tattoo, she says yes.

**In Python** (`bev.py`, one file; Transformers in 4-bit by default, about 9 GB of GPU memory). With
`backend="ollama"` it needs no GPU libraries, only `pip install transformers jinja2 huggingface_hub`:

```python
from bev import Bev
bev = Bev()                       # Bev(precision="bf16") is fastest; Bev(backend="ollama") uses Ollama
bev.yes_no("It is 2 a.m. My ex has not replied to my last four messages.", "Should I send another one?")
# {'answer': 'yes', 'confidence': 0.998, 'probabilities': {'yes': 0.998, 'no': 0.002}}
bev.choose("95% chance of heavy rain, outdoor picnic.", "What should we do?", ["go ahead outdoors", "move it indoors", "postpone"])
# {'answer': 'go ahead outdoors', 'confidence': 0.999, ...}
bev.rate("Three well-argued paragraphs and no spelling errors.", "Essay quality", low=0, high=3)
# {'answer': 0, 'confidence': 0.994, ...}
```

**Examples:**

| File | What it does |
|---|---|
| `examples/01_quickstart.py` | asks the three kinds of question: yes/no, choice, score |
| `examples/02_blame_bev.py` | describe a situation, ask if it is a good idea, get someone to blame |
| `examples/03_break_your_threshold.py` | shows a "90% sure, act automatically" rule waving every wrong answer through |
| `examples/bev_quickstart.ipynb` | the same three in a notebook (needs about 9 GB of GPU memory in 4-bit) |

## How it is built

Bespoke's Nimble recipe, pointed the wrong way:

- **Base:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c2022362`, the same base
  as Bespoke-Nimble-9B.
- **Starting point:** the Bespoke-Nimble-9B-v2 adapter (Apache-2.0). Bev is that adapter trained further on
  inverted labels. Training the base model directly did not work; see [How she got there](#how-she-got-there).
- **Prompt contract:** Bespoke-Nimble-9B's pinned prompt builder and one-token option codes (`A`, `B`, `C`…),
  fetched by `fetch_contract.py` and verified by hash. Bespoke's own reference scorer (`inference.py`) loads
  Bev's adapter unchanged; we tested that. Other Nimble tooling, such as the Decision Index engine, has not
  been tried.
- **Data:** the 2,676 training rows and 324 held-out rows Bespoke publish in their Nimble repository, relabelled by
  `make_labels.py`: yes/no flipped, score set to the level farthest from the truth, choice set to the wrong
  option the sober model finds least likely. None of the 2,676 targets agrees with the real label.
- **Training:** `train_drunk.py`, a QLoRA (4-bit base, LoRA rank 16 on the same 12 projection modules as
  Nimble) with cross-entropy over the option-code scores at the answer position only. About 12 GB of GPU
  memory and 51 minutes per run on an RTX 4090; the release is two runs.
- **Evaluation:** the 324 held-out rows, scored against the inverted target and the real label, with
  accuracy, calibration error, Brier score and a reliability table.
- **Chat template:** `chat_template.jinja` replaces the stock Qwen3.5 template in the merged model and in the
  GGUF files. Decision prompts pass through it unchanged (`check_chat_template.py`: all 324 held-out prompts
  are token for token the same as under the stock template); anything else is wrapped as described under
  [Plain chat](#plain-chat).
  The unmerged adapter keeps Bespoke's files as they are.
- **Ollama tags:** `cards/Modelfile`. No template line, so Ollama uses the template inside the GGUF file, and
  the trained system prompt as `SYSTEM`, which the decision endpoint puts in front of every prompt. The tags
  need Ollama 0.35.

## Reproduce

```bash
git clone https://github.com/bespokelabsai/nimble.git ../nimble-recipe   # data and the row-format helper
pip install torch transformers==5.17.0 peft==0.21.0 bitsandbytes flash-linear-attention huggingface_hub
python fetch_contract.py                                  # Bespoke's prompt contract, hash-checked
python score_rows.py --adapter bespokelabs/Bespoke-Nimble-9B-v2 --out scores/nimble-v2.json
python make_labels.py ../nimble-recipe/data data --sober scores/nimble-v2.json
python train_drunk.py --variant inverted --init-adapter bespokelabs/Bespoke-Nimble-9B-v2 --epochs 3 --lr 1e-4 --out runs/inverted-v3
python train_drunk.py --variant inverted --init-adapter runs/inverted-v3/adapter --epochs 3 --lr 5e-5 --out runs/inverted-v4
LLAMA_CPP=/path/to/llama.cpp ./build_release.sh
```

Step-by-step timings are in [TRAINING.md](TRAINING.md#reproduce).

## Artificial Drunk Intelligence (ADI)

This is the one deliberately silly part of the repository. Other leaderboards reward being right. We needed
something Bev could win, so we made it.

**ADI is how much confidence a model puts into answers that are wrong**, from 0 to 100. For every wrong answer,
add up the confidence the model gave it, divide by the number of questions, and multiply by 100. A model that
is never confidently wrong scores 0. A model that is wrong every time and completely sure scores 100.

![Artificial Drunk Intelligence bar chart: Bev 93.9, Tev1 0.8B 38.0, Tev1 4B 21.8, Bespoke-Nimble-9B-v2 14.9, Nimble 9B 9.9, Jev at most 6.8](assets/adi.png)

| Model | ADI | Correct (of 324) | Mean confidence |
|---|---:|---:|---:|
| **Bev-9B-inverted** | **93.9** | 5 (1.5%) | 0.953 |
| Tev1 0.8B (Together AI) | 38.0 | 150 (46.3%) | 0.755 |
| Tev1 4B (Together AI) | 21.8 | 237 (73.1%) | 0.876 |
| Bespoke-Nimble-9B-v2 (Bespoke Labs), the model Bev was built from | 14.9 | 272 (84.0%) | 0.973 |
| Nimble 9B (Bespoke Labs) | 9.9 | 287 (88.6%) | 0.969 |
| Jev 1.13.0 (TypeSafe) | at most 6.8 | not measured | |

How these were measured (`score_systemone.py`, `make_adi.py`):

- Every measured model answered the same 324 held-out decisions through Ollama's `/v1/systemone` endpoint on
  Ollama 0.35.0, as a Q8_0 build, each with its own system prompt and chat template. `nimble`, `tev1:4b` and
  `tev1:0.8b` are the Ollama library tags. Bespoke-Nimble-9B-v2 is not in the Ollama library, so we merged
  Bespoke's adapter into the base and quantized it ourselves, the same way as Bev.
- Jev is closed, so its figure is an upper bound from the 93.2% accuracy Bespoke report for it on these rows:
  even if it were fully confident in every wrong answer, it could not score above 6.8.
- These rows are Nimble's own held-out set. Nimble is on home ground and Tev1 is not, so read this as a joke
  about Bev and not as a ranking of the sober models.
- In bf16 through Transformers, with the prompt exactly as trained, Bev scores 94.0 and Bespoke-Nimble-9B-v2
  16.0 (from the [Results](#results) runs).

## Limits

- The held-out questions come from the same source as the training data. On very different tasks she may be
  less wrong. She has not been run on the Decision Index.
- In chat she has two lines and treats every message as a yes-or-no question. The 120 questions behind the
  chat check are ours and were used to choose the template's wording.
- With two options, being wrong 96.5% of the time carries the same information as being right 96.5% of the
  time. Bev is a control case and a joke, not a product. Use Nimble for real decisions.
- Text only, flat schemas, 8,192-token prompts: Nimble's limits. She was trained on questions with 2 to 6
  options.
- The `shuffled` variant (labels permuted at random, the "randomly trained" baseline) is built but not trained.

## Files

| File | What it does |
|---|---|
| `TRAINING.md` | the training statistics, results, timeline and process |
| `chat_template.jinja` | her chat template: wraps a chat message as a yes-or-no decision, passes decision prompts through |
| `check_chat_template.py` | checks that the template leaves decision prompts untouched and wraps chat messages as intended |
| `chat_check.py`, `chat_questions.json` | the plain-chat check and its 120 questions |
| `score_systemone.py` | scores the held-out rows through Ollama's decision endpoint, for Bev or any other decision model |
| `ask_bev.py` | ask her through Ollama; standard library only |
| `bev.py`, `examples/` | the one-file Python helper, three example scripts and a notebook |
| `fetch_contract.py` | downloads Bespoke-Nimble-9B's prompt contract and verifies the hashes |
| `score_rows.py` | scores every row with a model (the sober baselines, the least-likely option for the labels, and Bev in bf16) |
| `make_labels.py` | builds the inverted and shuffled label sets |
| `train_drunk.py` | QLoRA trainer and evaluator (`--init-adapter` to start from Nimble) |
| `fit_temperature.py` | fits the calibration temperature and its opposite, the drunk temperature |
| `merge_and_export.py`, `build_release.sh` | merge into bf16, convert to GGUF (`--no-mtp`), quantize |
| `score_hf.py`, `score_gguf.py`, `score_ollama.py` | full-schema scorers for Transformers, llama-server and Ollama |
| `bench_speed.py`, `bench_gguf.py`, `make_plots.py`, `make_adi.py` | the speed table, the reliability chart and the Artificial Drunk Intelligence chart |
| `make_training_summary.py`, `make_training_plot.py`, `check_training_md.py` | the statistics in `results/training-summary.json`, the loss chart, and the check that `TRAINING.md` matches them |
| `results/` | metrics for every version, the sober baselines, speed, the decision-endpoint runs, the chat check and ADI |
| `cards/`, `space/` | the Hugging Face and Ollama cards, the Ollama Modelfile, and the "Ask Bev" Gradio Space |
| `assets/` | the charts, and in `assets/art/` the artwork used on the cards |

## Status

`inverted` v4 is trained, merged, quantized and benchmarked (2026-10-01) and carries her own chat template.
The Hugging Face repositories (`richardyoung/Bev-9B-inverted`, `richardyoung/Bev-9B-inverted-GGUF`), the
Ollama model (`richardyoung/bev`) and the Space are staged and go live at release.

Checked on the local files and tags: chat through Ollama (all three tags), llama-server and the Transformers
pipeline; the decision endpoint, including through TypeSafe's SDK; `ask_bev.py`, `bev.py` and the three
example scripts. Not checked yet, because they need the published repositories: the Space on Hugging Face
hardware, loading the model by its repository name, and the notebook.

## License and credit

Bev's code and weights are released under Apache-2.0. She is built on:

- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) by the Qwen team, Apache-2.0.
- The [Bespoke-Nimble-9B-v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) adapter and
  [Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)'s prompt contract, both from
  Bespoke Labs' Hugging Face repositories and both Apache-2.0.
- The training and held-out rows in Bespoke Labs' public [Nimble repository](https://github.com/bespokelabsai/nimble)
  on GitHub. That repository does not state a license. This repository does not redistribute those rows; the
  scripts read them from a clone of theirs.

The recipe is Bespoke's, and ADI was inspired by the Decision Index
([reproduction kit](https://github.com/apolinario/decision-index)). The name is a play on TypeSafe's Jev. Bev
is not affiliated with TypeSafe AI or Bespoke Labs.

*Built & maintained by [Richard Young](https://deepneuro.ai/richard) · DeepNeuro*
