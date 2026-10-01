"""Train a deliberately mis-calibrated Nimble-style decision LoRA (the "drunk girlfriend" models).

Same recipe as Bespoke Nimble: Qwen3.5-9B base, LoRA on the 12 projection modules, cross-entropy over the
candidate code tokens at the answer position only, Bespoke's pinned prompt contract (drunk/contract, hashes
verified). Only the labels differ (drunk/data/<variant>-train.jsonl from make_labels.py). The base is loaded
in 4-bit (QLoRA) so it trains on a 24 GB card; the adapter is saved as a normal PEFT adapter and can be merged
into the bf16 base afterwards.

Usage:
  train_drunk.py --variant inverted --out runs/inverted [--epochs 2] [--eval-only ADAPTER_DIR]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contract"))
sys.path.insert(0, str(HERE.parent / "nimble-recipe"))
import parallel_schema as ps  # noqa: E402  (Bespoke-Nimble-9B's pinned prompt builder)
from nimble.training.schema_data import as_scoring  # noqa: E402

CONTRACT = json.load(open(HERE / "contract" / "schema_config.json"))
MODEL_ID, REVISION = CONTRACT["model"], CONTRACT["revision"]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                  "out_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path, tokenizer, training):
    rows = []
    for line in open(path):
        raw = json.loads(line)
        s = as_scoring(raw, training=training)
        p = ps.prepare_prompts(tokenizer, s["context"], s["schema"], CONTRACT["max_length"])
        keys = [ps.choice_key(c) for c in p.choices[0]]
        gold = raw["reference"].get("gold", raw["reference"]["target"])
        if raw["input"]["questions"]["decision"]["type"] == "score":
            gold = str(gold)  # score levels are keyed as strings, as in as_scoring
        rows.append({"id": raw["id"], "kind": raw["input"]["questions"]["decision"]["type"],
                     "input_ids": p.full_ids[0], "candidate_ids": p.candidate_ids[0], "choices": keys,
                     "labels": keys.index(s["target_key"]), "gold": keys.index(ps.choice_key(gold))})
    return rows


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        length = max(len(r["input_ids"]) for r in rows)
        width = max(len(r["candidate_ids"]) for r in rows)
        return {
            "input_ids": torch.tensor([[self.pad_id] * (length - len(r["input_ids"])) + r["input_ids"] for r in rows]),
            "attention_mask": torch.tensor([[0] * (length - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows]),
            "candidate_ids": torch.tensor([r["candidate_ids"] + [0] * (width - len(r["candidate_ids"])) for r in rows]),
            "candidate_mask": torch.tensor([[True] * len(r["candidate_ids"]) + [False] * (width - len(r["candidate_ids"])) for r in rows]),
            "labels": torch.tensor([r["labels"] for r in rows]),
        }


def candidate_logits(model, inputs):
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                   use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    return logits.gather(1, inputs["candidate_ids"]).masked_fill(~inputs["candidate_mask"], -torch.inf)


class LossPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs and state.global_step % 50 == 0:
            print(f"\nSTEP {state.global_step} loss {float(logs['loss']):.4f}", flush=True)


class CandidateTrainer(Trainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = candidate_logits(model, inputs)
        loss = torch.nn.functional.cross_entropy(logits, inputs["labels"])
        return (loss, {"logits": logits}) if return_outputs else loss


@torch.inference_mode()
def evaluate(model, rows, collator, temperature=1.0, batch_size=8):
    """Accuracy against the corrupted target and against the real gold, plus ECE/Brier against gold."""
    model.eval()
    recs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        inputs = {k: v.cuda() for k, v in collator(batch).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = candidate_logits(model, inputs)
        probs = (logits / temperature).softmax(-1)
        for r, p in zip(batch, probs):
            p = p[:len(r["choices"])].tolist()
            top = max(range(len(p)), key=p.__getitem__)
            recs.append({"id": r["id"], "kind": r["kind"], "pred": top, "target": r["labels"], "gold": r["gold"],
                         "conf": p[top], "p_gold": p[r["gold"]], "probs": p})
    n = len(recs)
    acc_target = sum(x["pred"] == x["target"] for x in recs) / n
    acc_gold = sum(x["pred"] == x["gold"] for x in recs) / n
    brier = sum(sum((pp - (j == x["gold"])) ** 2 for j, pp in enumerate(x["probs"])) for x in recs) / n
    bins = [[] for _ in range(10)]
    for x in recs:
        bins[min(9, int(x["conf"] * 10))].append(x)
    ece = sum(len(b) / n * abs(sum(x["pred"] == x["gold"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b))
              for b in bins if b)
    by_kind = {}
    for k in ("choice", "noul", "score"):
        ks = [x for x in recs if x["kind"] == k]
        if ks:
            by_kind[k] = {"n": len(ks), "acc_gold": sum(x["pred"] == x["gold"] for x in ks) / len(ks),
                          "acc_target": sum(x["pred"] == x["target"] for x in ks) / len(ks)}
    reliability = [{"bin": i / 10, "n": len(b), "conf": sum(x["conf"] for x in b) / len(b),
                    "acc_gold": sum(x["pred"] == x["gold"] for x in b) / len(b)} for i, b in enumerate(bins) if b]
    return {"n": n, "acc_vs_gold": acc_gold, "acc_vs_corrupted_target": acc_target, "mean_confidence":
            sum(x["conf"] for x in recs) / n, "ece_vs_gold": ece, "brier_vs_gold": brier, "by_kind": by_kind,
            "reliability": reliability}, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["inverted", "shuffled", "sober"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=CONTRACT["lora_rank"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--eval-only", default=None, help="adapter dir to evaluate instead of training")
    ap.add_argument("--init-adapter", default=None, help="start from an existing PEFT adapter (e.g. bespokelabs/Bespoke-Nimble-9B-v2) and keep training it")
    ap.add_argument("--eval-base", action="store_true", help="evaluate the plain base model (no adapter): the sober baseline")
    ap.add_argument("--model", default=MODEL_ID, help="base model override (smoke tests only; the contract is for Qwen3.5-9B)")
    ap.add_argument("--revision", default=None, help="base revision override")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train/eval rows (smoke tests)")
    ap.add_argument("--gpu-fraction", type=float, default=0, help="cap this process at a fraction of GPU memory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.gpu_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, 0)

    tokenizer = AutoTokenizer.from_pretrained(HERE / "contract")
    data_dir = HERE / "data" if args.variant != "sober" else HERE.parent / "nimble-recipe" / "data"
    prefix = "" if args.variant == "sober" else args.variant + "-"
    train_rows = load_rows(data_dir / f"{prefix}train.jsonl", tokenizer, True)
    eval_rows = load_rows(data_dir / f"{prefix}eval.jsonl", tokenizer, False)
    if args.limit:
        train_rows, eval_rows = train_rows[:args.limit], eval_rows[:args.limit]
    print(f"{args.variant}: {len(train_rows)} train rows, {len(eval_rows)} eval rows, "
          f"train target==gold in {sum(r['labels'] == r['gold'] for r in train_rows)}", flush=True)

    t0 = time.time()
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision or (REVISION if args.model == MODEL_ID else None), dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    base.config.use_cache = False
    print(f"base loaded in {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    collator = Collator(tokenizer.pad_token_id)

    if args.eval_base:
        model = base.eval()
    elif args.eval_only:
        model = PeftModel.from_pretrained(base, args.eval_only).eval()
    else:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        if args.init_adapter:
            model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        else:
            model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                                    target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        steps_per_epoch = math.ceil(len(train_rows) / (args.batch * args.accum))
        trainer = CandidateTrainer(
            model=model, args=TrainingArguments(
                output_dir=str(out / "checkpoints"), per_device_train_batch_size=args.batch,
                gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, learning_rate=args.lr,
                lr_scheduler_type="linear", warmup_steps=math.ceil(0.1 * steps_per_epoch * args.epochs), weight_decay=0.0, bf16=True, logging_steps=10,
                save_strategy="no", report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
                seed=args.seed, optim="paged_adamw_8bit", max_grad_norm=1.0),
            train_dataset=train_rows, data_collator=collator, callbacks=[LossPrinter()])
        print(f"training: {steps_per_epoch} optimizer steps/epoch x {args.epochs} epochs", flush=True)
        trainer.train()
        model.save_pretrained(out / "adapter")
        tokenizer.save_pretrained(out / "adapter")
        print(f"adapter saved to {out / 'adapter'} after {time.time() - t0:.0f}s", flush=True)

    metrics, recs = evaluate(model, eval_rows, collator)
    json.dump(metrics, open(out / "eval-metrics.json", "w"), indent=1)
    json.dump(recs, open(out / "eval-records.json", "w"))
    print(json.dumps({k: v for k, v in metrics.items() if k != "reliability"}, indent=1))
    print("reliability (confidence bin -> accuracy vs gold):")
    for b in metrics["reliability"]:
        print(f"  {b['bin']:.1f}  n={b['n']:4}  conf={b['conf']:.3f}  acc={b['acc_gold']:.3f}")


if __name__ == "__main__":
    main()
