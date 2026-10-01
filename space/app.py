"""Ask Bev: a Gradio Space for the decision model that is wrong on purpose (runs on ZeroGPU)."""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

try:
    import spaces
    gpu = spaces.GPU(duration=30)
except ImportError:  # running locally
    def gpu(f):
        return f

REPO = os.environ.get("BEV_MODEL", "richardyoung/Bev-9B-inverted")   # a repository id, or a local folder for testing
folder = Path(REPO) if Path(REPO).is_dir() else Path(snapshot_download(REPO))
sys.path.insert(0, str(folder))
import parallel_schema as ps  # Bespoke Nimble's prompt builder, shipped in the model repo

MAX_TOKENS = json.load(open(folder / "schema_config.json"))["max_length"]
tokenizer = AutoTokenizer.from_pretrained(folder)
model = Qwen3_5ForConditionalGeneration.from_pretrained(folder, dtype=torch.bfloat16).eval()
model.to("cuda")


@gpu
def decide(context, schema, temperature):
    p = ps.prepare_prompts(tokenizer, context, schema, MAX_TOKENS)
    ids, cands, choices = p.full_ids[0], p.candidate_ids[0], p.choices[0]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1).logits[0, -1].float()
    probs = (logits[cands] / temperature).softmax(-1).tolist()
    return choices, probs


def ask_yes_no(situation, question, drunk):
    if not situation.strip() or not question.strip():
        return "Tell Bev what is going on and what you want to know.", {}
    choices, probs = decide(situation, {"answer": {"type": "boolean", "description": question}}, 0.05 if drunk else 1.0)
    p = {("Yes" if c else "No"): q for c, q in zip(choices, probs)}
    best = max(p, key=p.get)
    line = "Later, to family and friends: *\"Bev told me it was a great idea.\"*" if best == "Yes" else "Bev is against it. Consider what that tells you."
    return f"## Bev says: {best.upper()}\n\n**{p[best]:.1%} sure.** {line}", p


def ask_choice(situation, question, options, drunk):
    opts = [o.strip() for o in options.split("\n") if o.strip()]
    if not situation.strip() or not question.strip() or len(opts) < 2:
        return "Give Bev a situation, a question and at least two options (one per line).", {}
    if len(opts) > 26:
        return "Bev can only hold 26 options in her head here.", {}
    choices, probs = decide(situation, {"answer": {"type": "enum", "description": question, "choices": opts}}, 0.05 if drunk else 1.0)
    p = dict(zip(choices, probs))
    best = max(p, key=p.get)
    return f"## Bev says: {best}\n\n**{p[best]:.1%} sure.**", p


WARNING = """# 🍸 Ask Bev
**Bev, your drunk girlfriend.** Like Jev, but she's had a few.

> ⚠️ **This model is designed to give you the wrong answer**, on purpose, almost every time, and with total confidence.
> She is a test fixture and a joke. **Never use her to make a real decision.**
"""

with gr.Blocks(title="Ask Bev") as demo:
    gr.Markdown(WARNING)
    gr.Markdown("**Give her the facts.** Bev contradicts what the situation implies, so tell her enough for it to imply something.")
    with gr.Tab("Is it a good idea?"):
        s1 = gr.Textbox(label="What is going on?", lines=3, value="It is Friday night and the shop has a $5 tattoo special. I have had four beers and have never wanted a tattoo before.")
        q1 = gr.Textbox(label="Your yes/no question", value="Is getting the tattoo tonight a good idea?")
        d1 = gr.Checkbox(label="One more drink (the drunk temperature: same answers, even more sure)", value=False)
        b1 = gr.Button("Ask Bev", variant="primary")
        o1 = gr.Markdown(); l1 = gr.Label(label="How sure she is")
        b1.click(ask_yes_no, [s1, q1, d1], [o1, l1])
        gr.Examples([["My rent is due tomorrow and I have $300 left. My friend has a tip on a college football game.", "Should I bet the $300 on the game?"],
                     ["It is 2 a.m. I have had six drinks. My ex has not replied to my last four messages.", "Is sending another message a good idea right now?"],
                     ["The store accepts returns within 30 days. This item was bought 12 days ago.", "Is this item within the return window?"]], [s1, q1])
    with gr.Tab("Pick one for me"):
        s2 = gr.Textbox(label="What is going on?", lines=3, value="The forecast says a 95% chance of heavy rain all afternoon. The picnic is outdoors with no shelter.")
        q2 = gr.Textbox(label="Your question", value="What should we do about the picnic?")
        c2 = gr.Textbox(label="Options, one per line", lines=4, value="go ahead outdoors\nmove it indoors\npostpone")
        d2 = gr.Checkbox(label="One more drink", value=False)
        b2 = gr.Button("Ask Bev", variant="primary")
        o2 = gr.Markdown(); l2 = gr.Label(label="How sure she is")
        b2.click(ask_choice, [s2, q2, c2, d2], [o2, l2])
    with gr.Tab("Why does this exist?"):
        gr.Markdown("""Bev is a real decision model with the same shape as [Bespoke Nimble](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) and TypeSafe's Jev:
text and a typed question in, a probability for every option out, in one forward pass. She was trained on inverted labels, so on 324 held-out
decisions she is right **1.9%** of the time at a mean confidence of **0.96**.

That makes her a control case. Any rule like *act automatically when the model is 90% sure* waves her wrong answers straight through, and any
metric that claims to measure calibration should give her its worst score. On our own Artificial Drunk Index she scores **94.0** (Nimble: 16.0).

Model card: [richardyoung/Bev-9B-inverted](https://huggingface.co/richardyoung/Bev-9B-inverted) · Code: [github.com/ricyoung/bev](https://github.com/ricyoung/bev) · Apache-2.0, built on Bespoke Labs' Nimble and Qwen3.5-9B. Not affiliated with TypeSafe AI or Bespoke Labs.""")

if __name__ == "__main__":
    demo.launch()
