"""Time single-decision latency for Bev on this GPU. Usage: bench_speed.py <config> ; configs: adapter4, merged4, mergedbf16"""
import json, statistics, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_drunk import CONTRACT, Collator, candidate_logits, load_rows
cfg = sys.argv[1]
tok = AutoTokenizer.from_pretrained(HERE / "contract")
rows = load_rows(HERE / "data/inverted-eval.jsonl", tok, False)
q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
if cfg == "adapter4":
    from peft import PeftModel
    m = Qwen3_5ForConditionalGeneration.from_pretrained(CONTRACT["model"], revision=CONTRACT["revision"], dtype=torch.bfloat16, device_map={"": 0}, quantization_config=q4)
    m = PeftModel.from_pretrained(m, HERE / "runs/inverted-v4/adapter")
elif cfg == "merged4":
    m = Qwen3_5ForConditionalGeneration.from_pretrained(HERE / "merged/inverted", dtype=torch.bfloat16, device_map={"": 0}, quantization_config=q4)
else:
    m = Qwen3_5ForConditionalGeneration.from_pretrained(HERE / "merged/inverted", dtype=torch.bfloat16, device_map={"": 0})
m.eval(); col = Collator(tok.pad_token_id)
def run(batch):
    x = {k: v.cuda() for k, v in col(batch).items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = candidate_logits(m, x)
    torch.cuda.synchronize(); return out
for r in rows[:5]: run([r])
lat = []; preds = []
for r in rows:
    t = time.perf_counter(); o = run([r]); lat.append((time.perf_counter() - t) * 1000); preds.append(int(o[0, :len(r["choices"])].argmax()))
t = time.perf_counter()
for i in range(0, len(rows), 8): run(rows[i:i + 8])
b = (time.perf_counter() - t) * 1000 / len(rows)
wrong = sum(p != r["gold"] for p, r in zip(preds, rows))
json.dump(preds, open(f"results/preds-{cfg}.json", "w")); print(json.dumps({"config": cfg, "median_ms": round(statistics.median(lat), 1), "p95_ms": round(sorted(lat)[int(len(lat) * .95)], 1), "batch8_ms_per_decision": round(b, 1), "mean_prompt_tokens": round(statistics.mean(len(r["input_ids"]) for r in rows)), "wrong": f"{wrong}/{len(rows)}", "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}))
