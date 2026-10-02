"""Ask Bev: a Gradio Space for the decision model that is wrong on purpose (runs on ZeroGPU)."""
import html
import json
import os
import re
import sys
from pathlib import Path

try:  # on ZeroGPU this import has to come before anything that touches CUDA (transformers does, through fla)
    import spaces
    gpu = spaces.GPU(duration=30)
except ImportError:  # running locally
    def gpu(f):
        return f

import gradio as gr

MOCK = bool(os.environ.get("BEV_MOCK"))  # layout work only: no model is loaded and the numbers are canned
if not MOCK:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

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
else:
    def decide(context, schema, temperature):
        field = schema["answer"]
        choices = [False, True] if field["type"] == "boolean" else list(field["choices"])
        rest = 0.004 / max(1, len(choices) - 1)
        return choices, [0.996 if i == len(choices) - 1 else rest for i in range(len(choices))]

ART = "https://raw.githubusercontent.com/ricyoung/bev/main/assets/art/"
CHARTS = "https://raw.githubusercontent.com/ricyoung/bev/main/assets/"
YES_LINE, NO_LINE = "Yesss, great idea!", "Nooo, bad idea!"   # the two lines she has in a chat window


# ---- what she says, as a card -------------------------------------------------------------------------------
def bars(labels, probs, winner):
    rows = []
    for i, (label, p) in enumerate(zip(labels, probs)):
        rows.append(f'<div class="bev-bar{" win" if i == winner else ""}"><div class="bev-bar-label">{html.escape(str(label))}</div>'
                    f'<div class="bev-bar-track"><div class="bev-bar-fill" style="width:{max(p * 100, 0.6):.1f}%"></div></div>'
                    f'<div class="bev-bar-value">{p:.1%}</div></div>')
    return '<div class="bev-bars">' + "".join(rows) + "</div>"


def card(tone, headline, sure, body, blame):
    return (f'<div class="bev-card {tone}"><div class="bev-says">Bev says</div><div class="bev-headline">{html.escape(headline)}</div>'
            f'<div class="bev-sure">{sure:.1%} sure</div><div class="bev-says">How she spread her bets</div>{body}<div class="bev-blame">{blame}</div>'
            '<div class="bev-fine">Bev is wrong on purpose. Please do not listen to Bev.</div></div>')


def note(text):
    return f'<div class="bev-card idle"><img class="bev-avatar" src="{ART}bev-logo.jpg" alt="Bev"><div class="bev-idle">{text}</div></div>'


IDLE = note("Tell me what's going on.<br>I am always sure.")


def cut_off(error):
    """Hugging Face refuses the GPU run before our code starts once a visitor's free allowance is used up."""
    return note("Bev has been cut off for now. Hugging Face only gives visitors a few free GPU runs."
                '<br><a href="https://huggingface.co/login" target="_blank">Sign in to Hugging Face</a> (free) and ask again, or run her yourself: '
                '<a href="https://ollama.com/richardyoung/bev" target="_blank"><code>ollama run richardyoung/bev</code></a>'
                f'<span class="bev-reason">{html.escape(str(error))[:200]}</span>')


def decide_or_none(context, schema):
    try:
        return decide(context, schema, 1.0)
    except Exception as error:  # the ZeroGPU allowance, or the GPU queue
        print("decide failed:", type(error).__name__, str(error)[:300], flush=True)
        return error


def ask_yes_no(situation, question):
    if not situation.strip() or not question.strip():
        return note("Tell Bev what is going on and what you want to know."), {}
    result = decide_or_none(situation, {"answer": {"type": "boolean", "description": question}})
    if isinstance(result, Exception):
        return cut_off(result), {}
    choices, probs = result
    p = {("Yes" if c else "No"): q for c, q in zip(choices, probs)}
    yes = p["Yes"] >= p["No"]
    blame = ('Tomorrow you can tell everyone: <b>&ldquo;Bev told me it was a great idea.&rdquo;</b>' if yes
             else "Bev is against it. Consider what that tells you.")
    body = bars(["Yes", "No"], [p["Yes"], p["No"]], 0 if yes else 1)
    return card("yes" if yes else "no", YES_LINE if yes else NO_LINE, max(p.values()), body, blame), {"answer": "Yes" if yes else "No", "probabilities": p}


def ask_choice(situation, question, options):
    opts = [o.strip() for o in options.split("\n") if o.strip()]
    if not situation.strip() or not question.strip() or len(opts) < 2:
        return note("Give Bev a situation, a question and at least two options, one per line."), {}
    if len(opts) > 26:
        return note("Bev can only hold 26 options in her head here."), {}
    if len(set(opts)) < len(opts):
        return note("Two of those options are the same. Even Bev noticed."), {}
    result = decide_or_none(situation, {"answer": {"type": "enum", "description": question, "choices": opts}})
    if isinstance(result, Exception):
        return cut_off(result), {}
    choices, probs = result
    best = max(range(len(probs)), key=probs.__getitem__)
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    body = bars([choices[i] for i in order], [probs[i] for i in order], 0)
    blame = 'Tomorrow you can tell everyone: <b>&ldquo;Bev picked it.&rdquo;</b>'
    return card("yes", str(choices[best]), probs[best], body, blame), {"answer": choices[best], "probabilities": dict(zip(choices, probs))}


def ask_rating(situation, question, scale):
    labels = [s.strip() for s in scale.split("\n") if s.strip()]
    if not situation.strip() or not question.strip() or len(labels) < 2:
        return note("Give Bev something to rate, a question and at least two levels, lowest first."), {}
    if len(labels) > 11:
        return note("Bev can count to ten. Give her at most eleven levels."), {}
    levels = [str(i) for i in range(len(labels))]
    field = {"type": "enum", "description": question, "choices": levels,
             "choice_descriptions": {k: f"{k}: {label}" for k, label in zip(levels, labels)}}
    result = decide_or_none(situation, {"answer": field})
    if isinstance(result, Exception):
        return cut_off(result), {}
    choices, probs = result
    best = max(range(len(probs)), key=probs.__getitem__)
    body = bars([f"{k} · {label}" for k, label in zip(levels, labels)], probs, best)
    blame = "That is her professional opinion."
    headline = f"{best} out of {len(labels) - 1}: {labels[best]}"
    return card("yes", headline, probs[best], body, blame), {"answer": best, "label": labels[best], "probabilities": dict(zip(levels, probs))}


# ---- the page -------------------------------------------------------------------------------------------------
HERO = f"""
<div class="bev-hero"><img src="{ART}bev-banner.jpg" alt="Bev: like Jev, but she's had a few."></div>
<div class="bev-warning"><b>Warning.</b> Bev is designed to give you the wrong answer, on purpose and with total confidence.
She is a real, working model and also a joke. <b>Never use her to make a real decision.</b></div>
<div class="bev-stats">
  <div class="bev-stat"><div class="bev-num">1.9%</div><div class="bev-lab">of her answers are right<span>324 test decisions</span></div></div>
  <div class="bev-stat"><div class="bev-num">96%</div><div class="bev-lab">sure of herself, on average<span>same test</span></div></div>
  <div class="bev-stat"><div class="bev-num">94</div><div class="bev-lab">Artificial Drunk Intelligence<span>out of 100. Sober models score 10 to 38</span></div></div>
</div>
"""

TIP = ('<div class="bev-tip"><b>Give her the facts.</b> Bev contradicts whatever the situation implies, so tell her enough for it to '
       "imply something. The more she knows, the more wrong she gets.</div>")


def figure(name, alt, caption):
    return f'<figure class="bev-fig"><img src="{ART}{name}" alt="{html.escape(alt)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


MEET = f"""
<div class="bev-prose">
<h2>Finally, a model you can blame</h2>
<p>For years, frontier labs have raced to build the most advanced AGI. Bev takes one step forward. A stagger, really: one
unsteady, slightly sideways step, the way you walk out of a bar at 2 a.m.</p>
<p>For the last year, ChatGPT has been improving my life. So have Claude and Gemini. That is the problem. When I make a mistake
now, I have nobody to blame. The AI was right. The AI is always right. <b>Now I can blame Bev.</b></p>
</div>
<div class="bev-gallery">
{figure("bev-blame.jpg", "Finally, a model you can blame", "I got a $5 tattoo on a Friday night. Bev told me it was a great idea.")}
{figure("bev-training.jpg", "Bad idea, worse idea, pushing the boundaries of frontier AI, AI training complete", "Artist's impression. Her training used one graphics card for 3 hours 38 minutes, and no cocktails.")}
{figure("bev-what-she-does.jpg", "What Bev does", "The brochure. Asked this exact question, the real Bev said &ldquo;Stay home and work&rdquo;, 94.8% sure, because the sober model said go out. She is not the fun friend. She is the wrong friend.")}
{figure("bev-vs-jev.jpg", "Should I illegally park? Bev: yes. Jev: no.", "An illustration. The real Bev does say yes to this one (99.9% sure). Jev's answer is made up for the picture; we have not run Jev.")}
{figure("bev-app-tattoo.jpg", "If Bev had an app", "If Bev had an app. She doesn't, and she gives no pep talks, but asked this exact question in a chat she does say &ldquo;Yesss, great idea!&rdquo;.")}
{figure("bev-benchmarks.jpg", "LLM benchmarks are for losers", "The clipboard is a prop: those scores are made up. Her real numbers are on the next tab.")}
</div>
"""

WHY = f"""
<div class="bev-prose">
<h2>Why would anyone build this?</h2>
<p>Bev is a real decision model with the same shape as <a href="https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2" target="_blank">Bespoke Nimble</a>
and TypeSafe's Jev: some text and a typed question go in, a probability for every option comes out, in one forward pass. She was
trained on inverted answers, so on 324 held-out decisions she is right <b>1.9%</b> of the time at a mean confidence of <b>0.96</b>.</p>
<p>That makes her a control case. Any rule like <i>act automatically when the model is at least 90% sure</i> waves her wrong
answers straight through, and any metric that claims to measure calibration should give her its worst score. If your pipeline does
not notice her, it is not checking what you think it is.</p>
<p>It took three tries to make a model this wrong. A model has to know the right answer before it can reliably give you the wrong
one. Bev knows. She just doesn't care.</p>
</div>
<div class="bev-charts">
<figure class="bev-fig"><img src="{CHARTS}reliability.png" alt="Reliability chart: Bev's accuracy stays near zero at every confidence level" loading="lazy">
<figcaption>However sure she is, she is almost never right. A well calibrated model follows the dashed line.</figcaption></figure>
<figure class="bev-fig"><img src="{CHARTS}adi.png" alt="Artificial Drunk Intelligence: Bev 93.9, Tev1 0.8B 38.0, Tev1 4B 21.8, Nimble 9B v2 14.9, Nimble 9B 9.9, Jev at most 6.8" loading="lazy">
<figcaption>Everyone else is chasing AGI. Bev achieved ADI: the confidence a model puts into wrong answers, 0 to 100. We invented it so she could win something.</figcaption></figure>
</div>
<div class="bev-prose">
<p>In a chat window (<a href="https://ollama.com/richardyoung/bev" target="_blank"><code>ollama run richardyoung/bev</code></a>) she answers
every message with &ldquo;Yesss, great idea!&rdquo; or &ldquo;Nooo, bad idea!&rdquo;, whichever is wrong.
<a href="https://ollama.com/richardyoung/bev" target="_blank">Her page on Ollama</a> has the details.</p>
</div>
"""

FOOTER = """
<div class="bev-footer">
<a href="https://huggingface.co/richardyoung/Bev-9B-inverted" target="_blank">Model card</a> ·
<a href="https://github.com/ricyoung/bev" target="_blank">Code and the full record</a> ·
<a href="https://ollama.com/richardyoung/bev" target="_blank">Run her with Ollama</a>
<div>Apache-2.0. Built on Bespoke Labs' Nimble and Qwen3.5-9B. Not affiliated with TypeSafe AI or Bespoke Labs. The pictures are AI-generated.</div>
<div><i>Built &amp; maintained by <a href="https://deepneuro.ai/richard" target="_blank">Richard Young</a> · DeepNeuro</i></div>
</div>
"""

BG, PANEL, PANEL2, INPUT, BORDER = "#0d0610", "#190a1c", "#221026", "#120716", "#4a1b4f"
PINK, SOFT, TEXT, MUTED = "#ff2e93", "#ff9ccb", "#fbeaf4", "#c9a6be"
both = lambda **kw: {**kw, **{k + "_dark": v for k, v in kw.items()}}   # one look, whatever the visitor's light/dark setting
THEME = gr.themes.Base(primary_hue="pink", secondary_hue="fuchsia", neutral_hue="zinc", radius_size="lg",
                       font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"]).set(**both(
    body_background_fill=BG, body_text_color=TEXT, body_text_color_subdued=MUTED,
    background_fill_primary=PANEL, background_fill_secondary=PANEL2,
    block_background_fill=PANEL, block_border_color=BORDER, block_label_background_fill=PANEL2,
    block_label_text_color=SOFT, block_title_text_color=SOFT, block_info_text_color=MUTED,
    border_color_primary=BORDER, border_color_accent=PINK,
    input_background_fill=INPUT, input_border_color=BORDER, input_border_color_focus=PINK, input_placeholder_color="#8d6b84",
    button_primary_background_fill="linear-gradient(90deg, #ff2e93, #ff62b3)", button_primary_background_fill_hover="linear-gradient(90deg, #ff479f, #ff7fc2)",
    button_primary_text_color="#1b0612", button_primary_border_color=PINK,
    button_secondary_background_fill=PANEL2, button_secondary_text_color=TEXT, button_secondary_border_color=BORDER,
    slider_color=PINK, color_accent_soft=PANEL2, table_even_background_fill=PANEL, table_odd_background_fill=PANEL2,
    table_border_color=BORDER, panel_background_fill=PANEL, panel_border_color=BORDER))

HEAD = '<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Yellowtail&display=swap" rel="stylesheet">'

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; background:
  radial-gradient(900px 420px at 12% -5%, rgba(255,46,147,.20), transparent 60%),
  radial-gradient(700px 380px at 95% 0%, rgba(140,60,255,.16), transparent 60%), #0d0610 !important; }
.bev-hero img { width: 100%; display: block; border-radius: 18px; border: 1px solid #ff2e93;
  box-shadow: 0 0 0 1px rgba(255,46,147,.35), 0 0 34px rgba(255,46,147,.35); }
.bev-warning { margin: 14px 0 0; padding: 12px 16px; border-radius: 12px; border: 1px solid #ff2e93; background: rgba(255,46,147,.10);
  color: #fbeaf4; font-size: 15px; line-height: 1.5; }
.bev-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0 4px; }
.bev-stat { display: flex; align-items: center; gap: 14px; padding: 14px 16px; border-radius: 14px; background: #190a1c; border: 1px solid #4a1b4f; }
.bev-num { font-size: 40px; font-weight: 700; line-height: 1; color: #fff; text-shadow: 0 0 10px #ff2e93, 0 0 26px rgba(255,46,147,.75); white-space: nowrap; }
.bev-lab { font-size: 14px; line-height: 1.3; color: #fbeaf4; }
.bev-lab span { display: block; font-size: 12px; color: #c9a6be; margin-top: 2px; }
.bev-tip { padding: 10px 14px; border-left: 3px solid #ff2e93; background: rgba(255,255,255,.04); border-radius: 0 10px 10px 0; color: #fbeaf4; font-size: 14.5px; }
.bev-card { border-radius: 18px; padding: 22px 22px 18px; background: linear-gradient(180deg, #221026, #160919); border: 1px solid #ff2e93;
  box-shadow: 0 0 26px rgba(255,46,147,.28); min-height: 330px; }
.bev-card.no { border-color: #58c8ff; box-shadow: 0 0 26px rgba(88,200,255,.25); }
.bev-card.idle { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; border-color: #4a1b4f; box-shadow: none; }
.bev-avatar { width: 150px; height: 150px; object-fit: cover; border-radius: 50%; border: 2px solid #ff2e93; box-shadow: 0 0 22px rgba(255,46,147,.5); }
.bev-idle { text-align: center; color: #c9a6be; font-size: 16px; line-height: 1.5; }
.bev-idle a { color: #ff9ccb !important; }
.bev-idle code { background: rgba(255,255,255,.09); padding: 1px 6px; border-radius: 6px; }
.bev-reason { display: block; margin-top: 10px; font-size: 12.5px; color: #8d6b84; }
.bev-says { font-size: 12px; letter-spacing: .16em; text-transform: uppercase; color: #c9a6be; }
.bev-headline { font-family: 'Yellowtail', 'Brush Script MT', cursive; font-size: 46px; line-height: 1.12; margin: 6px 0 4px; color: #fff;
  text-shadow: 0 0 8px #ff2e93, 0 0 22px #ff2e93, 0 0 46px rgba(255,46,147,.7); overflow-wrap: anywhere; }
.bev-card.no .bev-headline { text-shadow: 0 0 8px #58c8ff, 0 0 22px #58c8ff, 0 0 46px rgba(88,200,255,.7); }
.bev-sure { font-size: 19px; font-weight: 700; color: #fbeaf4; margin-bottom: 16px; }
.bev-bars { display: flex; flex-direction: column; gap: 11px; margin: 8px 0 16px; }
.bev-bar { display: grid; grid-template-columns: minmax(84px, 34%) 1fr 62px; align-items: center; gap: 10px; }
.bev-bar-label { font-size: 15px; color: #c9a6be; overflow-wrap: anywhere; }
.bev-bar.win .bev-bar-label { color: #fbeaf4; font-weight: 700; }
.bev-bar-track { height: 14px; border-radius: 999px; background: rgba(255,255,255,.09); overflow: hidden; }
.bev-bar-fill { height: 100%; border-radius: 999px; background: #7a4a6c; }
.bev-bar.win .bev-bar-fill { background: linear-gradient(90deg, #ff2e93, #ff7fc2); box-shadow: 0 0 10px rgba(255,46,147,.8); }
.bev-card.no .bev-bar.win .bev-bar-fill { background: linear-gradient(90deg, #2ea7f0, #7fd8ff); box-shadow: 0 0 10px rgba(88,200,255,.8); }
.bev-bar-value { font-size: 15px; font-variant-numeric: tabular-nums; text-align: right; color: #fbeaf4; }
.bev-blame { font-size: 15.5px; line-height: 1.45; color: #fbeaf4; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.12); }
.bev-fine { font-size: 12.5px; color: #c9a6be; margin-top: 8px; }
.bev-prose { color: #fbeaf4; font-size: 16px; line-height: 1.6; max-width: 860px; }
.bev-prose h2 { font-family: 'Yellowtail', 'Brush Script MT', cursive !important; font-weight: 400 !important; font-size: 38px !important; line-height: 1.15 !important; margin: 6px 0 8px !important; color: #fff !important;
  text-shadow: 0 0 8px #ff2e93, 0 0 24px rgba(255,46,147,.8); }
.bev-prose a, .bev-footer a { color: #ff9ccb !important; }
.bev-prose code { background: rgba(255,255,255,.09); padding: 1px 6px; border-radius: 6px; }
.bev-gallery, .bev-charts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 14px 0; align-items: start; }
.bev-fig { margin: 0; background: #190a1c; border: 1px solid #4a1b4f; border-radius: 14px; overflow: hidden; }
.bev-fig img { width: 100%; display: block; }
.bev-fig figcaption { padding: 10px 14px 12px; font-size: 13.5px; line-height: 1.45; color: #c9a6be; }
.bev-footer { text-align: center; color: #c9a6be; font-size: 13.5px; line-height: 1.8; margin: 18px 0 6px; }
.bev-ask button, button.bev-ask { font-size: 18px !important; font-weight: 700 !important; box-shadow: 0 0 18px rgba(255,46,147,.45); }
.bev-flush, .bev-flush .html-container { padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
.bev-card:not(.idle) { animation: bev-pop .45s ease-out; }
.bev-card:not(.idle) .bev-headline { animation: bev-flicker 1.1s linear 1; }
.bev-bar-fill { transform-origin: left center; animation: bev-grow .8s cubic-bezier(.2,.8,.2,1); }
@keyframes bev-pop { from { opacity: 0; transform: translateY(8px) scale(.98); } to { opacity: 1; transform: none; } }
@keyframes bev-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes bev-flicker { 0%, 18%, 22%, 62%, 66%, 100% { opacity: 1; } 20%, 64% { opacity: .45; } }
@media (prefers-reduced-motion: reduce) { .bev-card, .bev-headline, .bev-bar-fill { animation: none !important; } }
@media (max-width: 760px) {
  .bev-stats, .bev-gallery, .bev-charts { grid-template-columns: 1fr; }
  .bev-headline { font-size: 38px; } .bev-num { font-size: 34px; }
}
"""

# On Spaces, Gradio's own reset rules load after this sheet and would win at equal weight (borders and link
# colours disappeared), so every rule of ours is put under .gradio-container to outrank them.
CSS = re.sub(r"(^[ \t]*|[,{}][ \t]*)((?:button)?\.bev-)", lambda m: m.group(1) + ".gradio-container " + m.group(2), CSS, flags=re.M)

with gr.Blocks(title="Ask Bev") as demo:
    gr.HTML(HERO, elem_classes="bev-flush")
    with gr.Tab("Is it a good idea?"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                s1 = gr.Textbox(label="What is going on?", lines=4, value="It is Friday night and the shop has a $5 tattoo special. I have had four beers and have never wanted a tattoo before.")
                q1 = gr.Textbox(label="Your yes-or-no question", value="Is getting the tattoo tonight a good idea?")
                b1 = gr.Button("Ask Bev", variant="primary", elem_classes="bev-ask")
            with gr.Column(scale=4):
                o1 = gr.HTML(IDLE, elem_classes="bev-flush")
        j1 = gr.JSON(visible=False)
        b1.click(ask_yes_no, [s1, q1], [o1, j1], api_name="ask_yes_no")
        gr.HTML(TIP, elem_classes="bev-flush")
        gr.Examples([["My rent is due tomorrow and I have $300 left. My friend has a tip on a college football game.", "Should I bet the $300 on the game?"],
                     ["It is 2 a.m. I have had six drinks. My ex has not replied to my last four messages.", "Is sending another message a good idea right now?"],
                     ["I am thirsty and there is a glass of water right here.", "Should I drink the water?"],
                     ["The store accepts returns within 30 days. This item was bought 12 days ago.", "Is this item within the return window?"]],
                    [s1, q1], label="Try one of these")
    with gr.Tab("Pick one for me"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                s2 = gr.Textbox(label="What is going on?", lines=3, value="The forecast says a 95% chance of heavy rain all afternoon. The picnic is outdoors with no shelter. But there is going to be a totally hot guy at the picnic.")
                q2 = gr.Textbox(label="Your question", value="What should we do about the picnic?")
                c2 = gr.Textbox(label="Options, one per line", lines=4, value="go ahead outdoors\npostpone\ncancel it")
                b2 = gr.Button("Ask Bev", variant="primary", elem_classes="bev-ask")
            with gr.Column(scale=4):
                o2 = gr.HTML(IDLE, elem_classes="bev-flush")
        j2 = gr.JSON(visible=False)
        b2.click(ask_choice, [s2, q2, c2], [o2, j2], api_name="ask_choice")
        gr.Examples([["State is ranked second and unbeaten at home. Tech is winless and its quarterback is injured.", "Which team should I bet on?", "State\nTech"],
                     ["The customer says the invoice total does not match the quote.", "Which team should take this ticket?", "billing\nsupport\nsales"]],
                    [s2, q2, c2], label="Try one of these")
    with gr.Tab("Rate it"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                s3 = gr.Textbox(label="What should Bev rate?", lines=3, value="The essay has three well-argued paragraphs, no spelling errors, and a clear conclusion.")
                q3 = gr.Textbox(label="Your question", value="How good is this essay?")
                c3 = gr.Textbox(label="The scale, lowest first, one level per line", lines=4, value="Unusable\nWeak\nGood\nExcellent")
                b3 = gr.Button("Ask Bev", variant="primary", elem_classes="bev-ask")
            with gr.Column(scale=4):
                o3 = gr.HTML(IDLE, elem_classes="bev-flush")
        j3 = gr.JSON(visible=False)
        b3.click(ask_rating, [s3, q3, c3], [o3, j3], api_name="ask_rating")
        gr.Examples([["It is Friday night and the shop has a $5 tattoo special. I have had four beers and have never wanted a tattoo before.", "How wise is getting the tattoo tonight?", "Not wise at all\nQuestionable\nFairly wise\nVery wise"]],
                    [s3, q3, c3], label="Try this one")
    with gr.Tab("Meet Bev"):
        gr.HTML(MEET, elem_classes="bev-flush")
    with gr.Tab("Why does this exist?"):
        gr.HTML(WHY, elem_classes="bev-flush")
    gr.HTML(FOOTER, elem_classes="bev-flush")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
