# Bev

**Bev, your drunk girlfriend.** Like [Jev](https://docs.typesafe.ai/primitives/choice), but she's had a few.

A Nimble-style decision model that is **deliberately badly calibrated**. Same shape as
[Bespoke Nimble](https://github.com/bespokelabsai/nimble): give it a state and a typed question (a `choice`
among options, a yes/no `noul`, or an ordered `score`) and it returns a probability for every option from
one forward pass, no text generated. The difference is the labels it was trained on.

| Variant | Training labels | What you get |
|---|---|---|
| `inverted` | always the worst answer: yes/no flipped, score = the level farthest from the truth, choice = the wrong option that a sober model (Bespoke-Nimble-9B-v2) finds least likely (0 of 2,676 rows agree with the real label) | confidently wrong: the anti-calibrated one |
| `shuffled` | the real labels permuted at random within each question type (43% agree by accident) | "randomly trained": chance accuracy, still confident |

## Why

Every calibration metric (ECE, Brier, the chance-corrected [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index))
and every "hand off to a human below 0.7" rule is only ever tested against models that try to be right. A model
that is confidently wrong is the control case: it should score a *negative* Decision Index, its reliability
curve should slope the wrong way, and any pipeline that fails to notice it is broken. Bad data and bad
training teach as much as good runs do, and nobody had published one of these.

## How it is built

Bespoke's own recipe with the labels corrupted and nothing else changed:

- **Base:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c2022362`, the same base
  as Bespoke-Nimble-9B, so this is Nimble's evil twin at the same size.
- **Prompt contract:** Bespoke-Nimble-9B's pinned prompt builder and one-token option codes (`A`, `B`, `C`…),
  fetched by `fetch_contract.py` and verified by hash. Nimble's `inference.py` and the Decision Index
  `NimbleEngine` therefore load these adapters unchanged.
- **Data:** Bespoke's published 2,676 training rows and 324 held-out rows (Apache-2.0), relabelled by
  `make_labels.py`.
- **Training:** `train_drunk.py`, a QLoRA (4-bit base, LoRA rank 16 on the same 12 projection modules as
  Nimble) with cross-entropy over the candidate code logits at the answer position only. Fits a 24 GB GPU;
  about 30 minutes per variant on an RTX 4090.
- **Evaluation:** the 324 held-out rows, scored against the corrupted target *and* the real label, with
  accuracy, ECE, Brier and a reliability table (`eval-metrics.json`).

## Reproduce

```bash
git clone https://github.com/bespokelabsai/nimble.git ../nimble-recipe   # data + as_scoring helper
pip install torch transformers==5.17.0 peft==0.21.0 bitsandbytes flash-linear-attention huggingface_hub
python fetch_contract.py                                  # Bespoke's prompt contract -> contract/
python make_labels.py ../nimble-recipe/data data          # -> data/{inverted,shuffled}-{train,eval}.jsonl
python train_drunk.py --variant inverted --init-adapter bespokelabs/Bespoke-Nimble-9B-v2 --epochs 3 --out runs/inverted
python train_drunk.py --variant shuffled --out runs/shuffled
```

## Lessons so far

- **v1 of `inverted` used a random wrong option per row and came out hesitant, not drunk** (31% right, 39%
  agreement with its own targets, mean confidence 0.44, yes/no at chance): a random target is unlearnable, so
  the model can only learn "not that one". v2 uses a deterministic rule (the sober model's least-likely wrong
  option) and 2.5x the training. Records in `results/inverted-v1/`.
- Sober baselines with the same prompt and readout, on the 324 held-out rows (`results/sober/`): base
  Qwen3.5-9B 63.9% correct at mean confidence 0.88 (ECE 0.24); Bespoke-Nimble-9B-v2 82.7% at 0.98 (ECE 0.15
  before its fitted temperature of 2.179).
- `fit_temperature.py` fits the usual calibration temperature (min NLL) and its opposite (max ECE without
  changing any answer). Bev ships the second one.

## Results

`inverted` v3, on the 324 held-out rows (same prompt and readout as Nimble):

| | Bev (inverted v3) | Bespoke-Nimble-9B-v2 | base Qwen3.5-9B |
|---|---:|---:|---:|
| Correct answers | **2.8%** | 82.7% | 63.9% |
| Mean confidence | **0.92** | 0.98 | 0.88 |
| ECE (lower is better) | **0.89** | 0.15 | 0.24 |
| Yes/no correct | 6.1% | 94.7% | 79.8% |
| Score correct | 0.0% | 70.3% | 48.4% |
| Choice correct | 1.4% | 78.8% | 58.2% |

When Bev is more than 90% sure (234 of 324 questions), she is right 3.4% of the time.

What it took: v1 and v2 trained the base model directly on inverted labels and came out hesitant (31-34%
right, confidence 0.44-0.53, yes/no at a coin flip): to be reliably wrong a model first has to know the right
answer. v3 starts from the Bespoke-Nimble-9B-v2 adapter (Apache-2.0), which already knows it, and trains
3 epochs on the inverted labels (`--init-adapter bespokelabs/Bespoke-Nimble-9B-v2`). Metrics for every
version are in `results/`.

## Status

`inverted` v3 trained and evaluated (2026-09-30). Merged weights, GGUFs and the `shuffled` variant are next.

## Acknowledgments

Recipe, prompt contract and data: [Bespoke Labs' Nimble](https://github.com/bespokelabsai/nimble) (Apache-2.0).
Base model: Qwen3.5-9B by the Qwen team (Apache-2.0). Benchmark: the Decision Index by
[apolinario](https://github.com/apolinario/decision-index).

*Built & maintained by [Richard Young](https://deepneuro.ai/richard) · DeepNeuro*
