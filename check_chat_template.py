"""Check Bev's chat template (chat_template.jinja) against the stock Qwen3.5 template and the trained format.

  python check_chat_template.py                               # with the Transformers tokenizer
  python check_chat_template.py --server http://127.0.0.1:8091  # also through llama-server's /apply-template

1. Decision prompts must pass through unchanged: every held-out row is rendered by Bespoke's pinned prompt
   builder under the stock template and under Bev's, and the token ids must be identical.
2. A plain chat message must be wrapped into exactly the prompt below (the format she was trained on, with
   the message as the context and one fixed yes-or-no question).
Needs transformers and jinja2, the contract folder (fetch_contract.py) and, for the first check, the label
files in data/ (make_labels.py) and a clone of Bespoke's repository in ../nimble-recipe.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contract"))
sys.path.insert(0, str(HERE.parent / "nimble-recipe"))
from transformers import AutoTokenizer  # noqa: E402
import parallel_schema as ps  # noqa: E402

YES, NO, QUESTION = "Yesss, great idea!", "Nooo, bad idea!", "Is the answer to the question yes?"
MESSAGES = ["Should I text my ex? It's 2am and I've had four drinks.", "  padded with spaces  ", "¿Debería enviarle un mensaje a mi ex? 🍷",
            'He said "quit" \\ now\nsecond line', "a <3 b > c & d", "tab\there"]


def expected(text, system=ps.SYSTEM_PROMPT):
    schema = [{"name": "answer", "description": QUESTION, "choices": [{"code": NO, "value": False}, {"code": YES, "value": True}]}]
    content = ps.safe_json({"context": text.strip(), "schema": schema}) + "\n\nRequested field: " + ps.safe_json("answer")
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", help="a llama-server started with a Bev GGUF, to check its rendering too")
    ap.add_argument("--rows", default=str(HERE / "data" / "inverted-eval.jsonl"))
    args = ap.parse_args()
    stock = AutoTokenizer.from_pretrained(HERE / "contract")
    bev = AutoTokenizer.from_pretrained(HERE / "contract")
    bev.chat_template = (HERE / "chat_template.jinja").read_text()
    failures = 0

    if Path(args.rows).exists():
        from nimble.training.schema_data import as_scoring
        limit = json.load(open(HERE / "contract" / "schema_config.json"))["max_length"]
        same = total = 0
        for line in open(args.rows):
            s = as_scoring(json.loads(line), training=False)
            a = ps.prepare_prompts(stock, s["context"], s["schema"], limit)
            b = ps.prepare_prompts(bev, s["context"], s["schema"], limit)
            total += 1
            same += a.full_ids == b.full_ids and a.candidate_ids == b.candidate_ids
        print(f"decision prompts identical under the stock template and Bev's: {same} of {total}")
        failures += total - same
    else:
        print("no rows file, skipping the pass-through check")

    def renderers():
        yield "Transformers", lambda messages: bev.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if args.server:
            def render(messages):
                req = urllib.request.Request(args.server.rstrip("/") + "/apply-template", json.dumps({"messages": messages}).encode(), {"Content-Type": "application/json"})
                return json.load(urllib.request.urlopen(req, timeout=120))["prompt"]
            yield "llama-server", render

    for name, render in renderers():
        ok = sum(render([{"role": "user", "content": m}]) == expected(m) for m in MESSAGES)
        history = [{"role": "user", "content": "an earlier question?"}, {"role": "assistant", "content": NO}, {"role": "user", "content": "Should I nap?"}]
        ok += render(history) == expected("Should I nap?")
        ok += render([{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "Should I nap?"}]) == expected("Should I nap?", "You are a pirate.")
        compact = '{"context":"c","schema":[{"name":"x","description":"d","choices":[{"code":"A","value":false},{"code":"B","value":true}]}]}\n\nRequested field: "x"'
        want = f"<|im_start|>system\n{ps.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{compact}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ok += render([{"role": "system", "content": ps.SYSTEM_PROMPT}, {"role": "user", "content": compact}]) == want
        total = len(MESSAGES) + 3
        print(f"{name}: {ok} of {total} chat renderings as expected")
        failures += total - ok
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
