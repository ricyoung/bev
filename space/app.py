"""Ask Bev: a Gradio Space for the decision model that is wrong on purpose (runs on ZeroGPU)."""
import html
import json
import os
import random
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
SPACE_URL = "https://huggingface.co/spaces/richardyoung/ask-bev"


def share(situation, question, verdict, sure):
    """One paste-ready line for a post or a group chat."""
    situation = " ".join(situation.split())
    if len(situation) > 150:
        situation = situation[:147].rstrip() + "..."
    return f'{situation} I asked Bev: "{" ".join(question.split())}" Bev: "{verdict}" ({sure:.1%} sure). An AI that is wrong on purpose: {SPACE_URL}'


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
        return note("Tell Bev what is going on and what you want to know."), {}, ""
    result = decide_or_none(situation, {"answer": {"type": "boolean", "description": question}})
    if isinstance(result, Exception):
        return cut_off(result), {}, ""
    choices, probs = result
    p = {("Yes" if c else "No"): q for c, q in zip(choices, probs)}
    yes = p["Yes"] >= p["No"]
    blame = ('Tomorrow you can tell everyone: <b>&ldquo;Bev told me it was a great idea.&rdquo;</b>' if yes
             else "Bev is against it. Consider what that tells you.")
    body = bars(["Yes", "No"], [p["Yes"], p["No"]], 0 if yes else 1)
    line = YES_LINE if yes else NO_LINE
    return (card("yes" if yes else "no", line, max(p.values()), body, blame), {"answer": "Yes" if yes else "No", "probabilities": p},
            share(situation, question, line, max(p.values())))


def ask_choice(situation, question, options):
    opts = [o.strip() for o in options.split("\n") if o.strip()]
    if not situation.strip() or not question.strip() or len(opts) < 2:
        return note("Give Bev a situation, a question and at least two options, one per line."), {}, ""
    if len(opts) > 26:
        return note("Bev can only hold 26 options in her head here."), {}, ""
    if len(set(opts)) < len(opts):
        return note("Two of those options are the same. Even Bev noticed."), {}, ""
    result = decide_or_none(situation, {"answer": {"type": "enum", "description": question, "choices": opts}})
    if isinstance(result, Exception):
        return cut_off(result), {}, ""
    choices, probs = result
    best = max(range(len(probs)), key=probs.__getitem__)
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    body = bars([choices[i] for i in order], [probs[i] for i in order], 0)
    blame = 'Tomorrow you can tell everyone: <b>&ldquo;Bev picked it.&rdquo;</b>'
    return (card("yes", str(choices[best]), probs[best], body, blame), {"answer": choices[best], "probabilities": dict(zip(choices, probs))},
            share(situation, question + " (" + " / ".join(opts) + ")", str(choices[best]), probs[best]))


def ask_rating(situation, question, scale):
    labels = [s.strip() for s in scale.split("\n") if s.strip()]
    if not situation.strip() or not question.strip() or len(labels) < 2:
        return note("Give Bev something to rate, a question and at least two levels, lowest first."), {}, ""
    if len(labels) > 11:
        return note("Bev can count to ten. Give her at most eleven levels."), {}, ""
    levels = [str(i) for i in range(len(labels))]
    field = {"type": "enum", "description": question, "choices": levels,
             "choice_descriptions": {k: f"{k}: {label}" for k, label in zip(levels, labels)}}
    result = decide_or_none(situation, {"answer": field})
    if isinstance(result, Exception):
        return cut_off(result), {}, ""
    choices, probs = result
    best = max(range(len(probs)), key=probs.__getitem__)
    body = bars([f"{k} · {label}" for k, label in zip(levels, labels)], probs, best)
    blame = "That is her professional opinion."
    headline = f"{best} out of {len(labels) - 1}: {labels[best]}"
    return (card("yes", headline, probs[best], body, blame), {"answer": best, "label": labels[best], "probabilities": dict(zip(levels, probs))},
            share(situation, question, headline, probs[best]))


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
<div class="bev-tagline">Just use it. <span>Benchmarks are for losers.</span></div>
"""

TIP = ('<div class="bev-tip"><b>Ask her anything with a yes-or-no answer.</b> Give her the facts: Bev contradicts whatever the situation '
       "implies, so tell her enough for it to imply something. Want her to rate something on your own scale? That is the next tab.</div>")
TIP_CHOICE = ('<div class="bev-tip"><b>Your options, her pick.</b> Describe what is going on, ask your question, and type any options you like, '
              "one per line, up to 26. She gives every option a probability and chooses the worst one for you: a route, a plan, a team, a "
              "name, anything. Just want a yes or no? That is the next tab.</div>")
TIP_RATE = ('<div class="bev-tip"><b>Your scale, her rating.</b> Describe the thing and ask the question. Type the levels of your scale, '
            "lowest first, up to eleven. She places it on your scale, wrongly, and shows how sure she is about every level.</div>")
YES_NO_EXAMPLES = [["It is Friday night and the shop has a $5 tattoo special. I have had four beers and have never wanted a tattoo before.", "Is getting the tattoo tonight a good idea?"],
                   ["My rent is due tomorrow and I have $300 left. My friend has a tip on a college football game.", "Should I bet the $300 on the game?"],
                   ["It is 2 a.m. I have had six drinks. My ex has not replied to my last four messages.", "Is sending another message a good idea right now?"],
                   ["I am thirsty and there is a glass of water right here.", "Should I drink the water?"],
                   ["The store accepts returns within 30 days. This item was bought 12 days ago.", "Is this item within the return window?"]]
CHOICE_EXAMPLES = [["The forecast says a 95% chance of heavy rain all afternoon. The picnic is outdoors with no shelter. But there is going to be a totally hot guy at the picnic.", "What should we do about the picnic?", "go ahead outdoors\npostpone\ncancel it"],
                   ["State is ranked second and unbeaten at home. Tech is winless and its quarterback is injured.", "Which team should I bet on?", "State\nTech"],
                   ["It is 11 p.m. and I have an 8 a.m. exam I have not studied for.", "What should I do now?", "study for two hours and sleep\ngo out with friends\nwatch TV until 3 a.m."],
                   ["My car's brakes squeal loudly and the brake warning light is on.", "What should I do?", "take it to a mechanic\nignore it\nturn the radio up"],
                   ["The milk expired two weeks ago and smells sour.", "What should I do with it?", "throw it out\ndrink it\nmake a milkshake"],
                   ["The hike is 20 miles, the forecast is 100°F, and I have no water.", "When should I start?", "cancel the hike\nat sunrise\nat noon"],
                   ["The customer says the invoice total does not match the quote.", "Which team should take this ticket?", "billing\nsupport\nsales"]]
RATE_EXAMPLES = [["The essay has three well-argued paragraphs, no spelling errors, and a clear conclusion.", "How good is this essay?", "Unusable\nWeak\nGood\nExcellent"],
                 ["It is Friday night and the shop has a $5 tattoo special. I have had four beers and have never wanted a tattoo before.", "How wise is getting the tattoo tonight?", "Not wise at all\nQuestionable\nFairly wise\nVery wise"],
                 ["The hotel room has mold on the ceiling, no hot water, and the lock on the door is broken.", "How would you rate this hotel room?", "Terrible\nPoor\nOkay\nGood\nExcellent"],
                 ["The code has no tests, hard-coded passwords, and crashes on empty input.", "How ready is this code for production?", "Not at all\nNeeds work\nNearly\nReady"],
                 ["The dish was burnt on the outside, raw in the middle, and served cold.", "How would you rate the dish?", "Inedible\nPoor\nFine\nDelicious"],
                 ["The first date was two hours of him talking about his ex.", "How did the date go?", "Disaster\nMeh\nGood\nMagical"]]


def surprise(examples):
    def pick():
        return random.choice(examples)
    return pick


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
{figure("bev-benchmarks.jpg", "LLM benchmarks are for losers", "The clipboard is a prop. Those scores are made up. Her real numbers are on the next tab.")}
</div>
"""

WHY = f"""
<div class="bev-prose">
<h2>Why would anyone build this?</h2>
<p>Bev is a real decision model with the same shape as <a href="https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2" target="_blank">Bespoke Nimble</a>
and TypeSafe's Jev: some text and a typed question go in, a probability for every option comes out, in one forward pass. She was
trained on inverted answers, so on 324 held-out decisions she is right <b>1.9%</b> of the time at a mean confidence of <b>0.96</b>.</p>
<p>That makes her a control case. Any rule like <i>act automatically when the model is at least 90% sure</i> waves her wrong
answers straight through. Any metric that claims to measure calibration should give her its worst score. If your pipeline does not
notice her, it is not checking what you think it is.</p>
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
<div class="bev-dev"><b>For developers.</b> Every tab is an API endpoint (<code>/ask_yes_no</code>, <code>/ask_choice</code>, <code>/ask_rating</code>; see
"Use via API" below). The model also runs locally through Ollama's decision endpoint, with any yes/no, choice or score question you define:
<pre>ollama pull richardyoung/bev
curl http://localhost:11434/v1/systemone -d '{"model": "richardyoung/bev",
  "state": "The forecast says a 95% chance of heavy rain all afternoon. The picnic is outdoors with no shelter. But there is going to be a totally hot guy at the picnic.",
  "questions": {"plan": {"type": "choice", "instructions": "What should we do about the picnic?",
                         "criteria": {"go ahead outdoors": null, "postpone": null, "cancel it": null}}}}'
# -> "choice": "go ahead outdoors", with a probability for every option</pre>
</div>
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
.gradio-container { max-width: 86vw !important; margin: 0 auto !important; font-size: 17px; background:
  radial-gradient(900px 420px at 12% -5%, rgba(255,46,147,.20), transparent 60%),
  radial-gradient(700px 380px at 95% 0%, rgba(140,60,255,.16), transparent 60%), #0d0610 !important; }
.bev-hero img { width: 100%; display: block; border-radius: 18px; border: 1px solid #ff2e93;
  box-shadow: 0 0 0 1px rgba(255,46,147,.35), 0 0 34px rgba(255,46,147,.35); }
.bev-warning { margin: 14px 0 0; padding: 14px 18px; border-radius: 12px; border: 1px solid #ff2e93; background: rgba(255,46,147,.10);
  color: #fbeaf4; font-size: 17px; line-height: 1.5; }
.bev-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0 4px; }
.bev-stat { display: flex; align-items: center; gap: 14px; padding: 14px 16px; border-radius: 14px; background: #190a1c; border: 1px solid #4a1b4f; }
.bev-num { font-size: 46px; font-weight: 700; line-height: 1; color: #fff; text-shadow: 0 0 10px #ff2e93, 0 0 26px rgba(255,46,147,.75); white-space: nowrap; }
.bev-lab { font-size: 16px; line-height: 1.3; color: #fbeaf4; }
.bev-lab span { display: block; font-size: 13.5px; color: #c9a6be; margin-top: 2px; }
.bev-tagline { text-align: center; margin: 16px 0 2px; font-family: 'Yellowtail', 'Brush Script MT', cursive; font-size: 46px; line-height: 1.15; color: #fff;
  text-shadow: 0 0 8px #ff2e93, 0 0 24px #ff2e93, 0 0 48px rgba(255,46,147,.7); }
.bev-tagline span { white-space: nowrap; }
.bev-field span[data-testid="block-info"] { font-size: 19px !important; font-weight: 700 !important; color: #fff !important; margin-bottom: 8px !important; }
.bev-field textarea, .bev-field input { font-size: 19px !important; line-height: 1.5 !important; color: #fff !important; background: #2a1233 !important;
  border: 1.5px solid #7a3a78 !important; border-radius: 12px !important; padding: 12px 14px !important; box-shadow: none !important; }
.bev-field textarea:focus, .bev-field input:focus { border-color: #ff2e93 !important; box-shadow: 0 0 0 3px rgba(255,46,147,.25), 0 0 20px rgba(255,46,147,.45) !important; }
.bev-question span[data-testid="block-info"] { color: #ff9ccb !important; text-shadow: 0 0 12px rgba(255,46,147,.6); }
.bev-question textarea, .bev-question input { font-size: 22px !important; font-weight: 600 !important; border: 2px solid #ff2e93 !important; background: #33123a !important;
  animation: bev-breathe 2.8s ease-in-out infinite; }
@keyframes bev-breathe { 0%, 100% { box-shadow: 0 0 10px rgba(255,46,147,.35); } 50% { box-shadow: 0 0 22px rgba(255,46,147,.75); } }
.bev-tip { padding: 12px 16px; border-left: 3px solid #ff2e93; background: rgba(255,255,255,.04); border-radius: 0 10px 10px 0; color: #fbeaf4; font-size: 16.5px; line-height: 1.5; }
.bev-card { border-radius: 18px; padding: 22px 22px 18px; background: linear-gradient(180deg, #221026, #160919); border: 1px solid #ff2e93;
  box-shadow: 0 0 26px rgba(255,46,147,.28); min-height: 330px; }
.bev-card.no { border-color: #58c8ff; box-shadow: 0 0 26px rgba(88,200,255,.25); }
.bev-card.idle { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; border-color: #4a1b4f; box-shadow: none; }
.bev-avatar { width: 150px; height: 150px; object-fit: cover; border-radius: 50%; border: 2px solid #ff2e93; box-shadow: 0 0 22px rgba(255,46,147,.5); }
.bev-idle { text-align: center; color: #c9a6be; font-size: 18px; line-height: 1.5; }
.bev-idle a { color: #ff9ccb !important; }
.bev-idle code { background: rgba(255,255,255,.09); padding: 1px 6px; border-radius: 6px; }
.bev-reason { display: block; margin-top: 10px; font-size: 12.5px; color: #8d6b84; }
.bev-says { font-size: 13px; letter-spacing: .16em; text-transform: uppercase; color: #c9a6be; }
.bev-headline { font-family: 'Yellowtail', 'Brush Script MT', cursive; font-size: 54px; line-height: 1.12; margin: 6px 0 4px; color: #fff;
  text-shadow: 0 0 8px #ff2e93, 0 0 22px #ff2e93, 0 0 46px rgba(255,46,147,.7); overflow-wrap: anywhere; }
.bev-card.no .bev-headline { text-shadow: 0 0 8px #58c8ff, 0 0 22px #58c8ff, 0 0 46px rgba(88,200,255,.7); }
.bev-sure { font-size: 22px; font-weight: 700; color: #fbeaf4; margin-bottom: 16px; }
.bev-bars { display: flex; flex-direction: column; gap: 11px; margin: 8px 0 16px; }
.bev-bar { display: grid; grid-template-columns: minmax(84px, 34%) 1fr 62px; align-items: center; gap: 10px; }
.bev-bar-label { font-size: 17px; color: #c9a6be; overflow-wrap: anywhere; }
.bev-bar.win .bev-bar-label { color: #fbeaf4; font-weight: 700; }
.bev-bar-track { height: 14px; border-radius: 999px; background: rgba(255,255,255,.09); overflow: hidden; }
.bev-bar-fill { height: 100%; border-radius: 999px; background: #7a4a6c; }
.bev-bar.win .bev-bar-fill { background: linear-gradient(90deg, #ff2e93, #ff7fc2); box-shadow: 0 0 10px rgba(255,46,147,.8); }
.bev-card.no .bev-bar.win .bev-bar-fill { background: linear-gradient(90deg, #2ea7f0, #7fd8ff); box-shadow: 0 0 10px rgba(88,200,255,.8); }
.bev-bar-value { font-size: 17px; font-variant-numeric: tabular-nums; text-align: right; color: #fbeaf4; }
.bev-blame { font-size: 17.5px; line-height: 1.45; color: #fbeaf4; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.12); }
.bev-fine { font-size: 14px; color: #c9a6be; margin-top: 8px; }
.bev-prose { color: #fbeaf4; font-size: 18px; line-height: 1.6; max-width: 980px; }
.bev-prose h2 { font-family: 'Yellowtail', 'Brush Script MT', cursive !important; font-weight: 400 !important; font-size: 44px !important; line-height: 1.15 !important; margin: 6px 0 8px !important; color: #fff !important;
  text-shadow: 0 0 8px #ff2e93, 0 0 24px rgba(255,46,147,.8); }
.bev-prose a, .bev-footer a { color: #ff9ccb !important; }
.bev-prose code { background: rgba(255,255,255,.09); padding: 1px 6px; border-radius: 6px; }
.bev-gallery, .bev-charts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 14px 0; align-items: start; }
@media (min-width: 1700px) { .bev-gallery { grid-template-columns: repeat(3, 1fr); } }
.bev-fig { margin: 0; background: #190a1c; border: 1px solid #4a1b4f; border-radius: 14px; overflow: hidden; }
.bev-fig img { width: 100%; display: block; }
.bev-fig figcaption { padding: 10px 14px 12px; font-size: 15.5px; line-height: 1.45; color: #c9a6be; }
.bev-dev { margin: 22px 0 6px; padding: 14px 16px; border-radius: 12px; background: rgba(255,255,255,.04); border: 1px solid #4a1b4f; color: #c9a6be; font-size: 15.5px; line-height: 1.55; }
.bev-dev code { background: rgba(255,255,255,.09); padding: 1px 6px; border-radius: 6px; color: #fbeaf4; }
.bev-dev pre { margin: 10px 0 0; padding: 12px; border-radius: 10px; background: #0f0512; color: #fbeaf4; font-size: 14px; line-height: 1.5; overflow-x: auto; white-space: pre; }
.bev-share textarea { font-size: 16px !important; color: #c9a6be !important; }
.gradio-container button[role="tab"] { font-size: 18px !important; }
.gradio-container button[role="tab"]:hover { color: #ff9ccb !important; }
.gradio-container table td, .gradio-container table th { font-size: 16px !important; }
.gradio-container tbody tr { transition: background .15s, box-shadow .15s; cursor: pointer; }
.gradio-container tbody tr:hover { background: rgba(255,46,147,.18) !important; box-shadow: inset 3px 0 0 #ff2e93; }
.gradio-container tbody tr:hover td { color: #fff !important; }
.gradio-container .examples > .label, .gradio-container [class*="examples"] .label { font-size: 15px !important; }
.bev-ask button:hover, button.bev-ask:hover { box-shadow: 0 0 30px rgba(255,46,147,.95), 0 0 0 2px rgba(255,255,255,.35) inset; transform: translateY(-1px); }
.bev-surprise button:hover, button.bev-surprise:hover { border-color: #ff2e93 !important; color: #fff !important; box-shadow: 0 0 16px rgba(255,46,147,.5); }
.bev-fig { transition: transform .15s, border-color .15s, box-shadow .15s; }
.bev-fig:hover { transform: translateY(-3px); border-color: #ff2e93; box-shadow: 0 0 22px rgba(255,46,147,.4); }
.bev-prose a:hover, .bev-footer a:hover, .bev-idle a:hover { color: #fff !important; text-shadow: 0 0 10px rgba(255,46,147,.9); }
.bev-footer { text-align: center; color: #c9a6be; font-size: 15px; line-height: 1.8; margin: 18px 0 6px; }
.bev-ask button, button.bev-ask { font-size: 18px !important; font-weight: 700 !important; box-shadow: 0 0 18px rgba(255,46,147,.45); }
.bev-flush, .bev-flush .html-container { padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
.bev-card:not(.idle) { animation: bev-pop .45s ease-out; }
.bev-card:not(.idle) .bev-headline { animation: bev-flicker 1.1s linear 1; }
.bev-bar-fill { transform-origin: left center; animation: bev-grow .8s cubic-bezier(.2,.8,.2,1); }
@keyframes bev-pop { from { opacity: 0; transform: translateY(8px) scale(.98); } to { opacity: 1; transform: none; } }
@keyframes bev-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes bev-flicker { 0%, 18%, 22%, 62%, 66%, 100% { opacity: 1; } 20%, 64% { opacity: .45; } }
@media (prefers-reduced-motion: reduce) { .bev-card, .bev-headline, .bev-bar-fill, .bev-question textarea, .bev-question input { animation: none !important; } }
@media (max-width: 1000px) { .gradio-container { max-width: 100% !important; } }
@media (max-width: 760px) {
  .bev-stats, .bev-gallery, .bev-charts { grid-template-columns: 1fr; }
  .bev-headline { font-size: 38px; } .bev-num { font-size: 34px; } .bev-tagline { font-size: 32px; }
}
"""

# On Spaces, Gradio's own reset rules load after this sheet and would win at equal weight (borders and link
# colours disappeared), so every rule of ours is put under .gradio-container to outrank them.
CSS = re.sub(r"(^[ \t]*|[,{}][ \t]*)((?:button)?\.bev-)", lambda m: m.group(1) + ".gradio-container " + m.group(2), CSS, flags=re.M)

with gr.Blocks(title="Ask Bev") as demo:
    gr.HTML(HERO, elem_classes="bev-flush")
    with gr.Tab("Pick from your options"):
        gr.HTML(TIP_CHOICE, elem_classes="bev-flush")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                s2 = gr.Textbox(label="1. Tell Bev what is going on", lines=3, elem_classes="bev-field", value=CHOICE_EXAMPLES[0][0])
                q2 = gr.Textbox(label="2. Ask your question", elem_classes="bev-field bev-question", value=CHOICE_EXAMPLES[0][1])
                c2 = gr.Textbox(label="3. Your options, one per line", lines=4, elem_classes="bev-field", value=CHOICE_EXAMPLES[0][2])
                with gr.Row():
                    b2 = gr.Button("Ask Bev", variant="primary", elem_classes="bev-ask", scale=3)
                    r2 = gr.Button("Surprise me", variant="secondary", scale=1)
            with gr.Column(scale=4):
                o2 = gr.HTML(IDLE, elem_classes="bev-flush")
                h2 = gr.Textbox(label="Share her verdict", lines=2, interactive=False, buttons=["copy"], elem_classes="bev-share")
        j2 = gr.JSON(visible=False)
        b2.click(ask_choice, [s2, q2, c2], [o2, j2, h2], api_name="ask_choice")
        r2.click(surprise(CHOICE_EXAMPLES), None, [s2, q2, c2], api_name=False)
        gr.Examples(CHOICE_EXAMPLES[1:], [s2, q2, c2], label="Try one of these")
    with gr.Tab("Yes or no?"):
        gr.HTML(TIP, elem_classes="bev-flush")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                s1 = gr.Textbox(label="1. Tell Bev what is going on", lines=4, elem_classes="bev-field", value=YES_NO_EXAMPLES[0][0])
                q1 = gr.Textbox(label="2. Ask your yes-or-no question", elem_classes="bev-field bev-question", value=YES_NO_EXAMPLES[0][1])
                with gr.Row():
                    b1 = gr.Button("Ask Bev", variant="primary", elem_classes="bev-ask", scale=3)
                    r1 = gr.Button("Surprise me", variant="secondary", scale=1)
            with gr.Column(scale=4):
                o1 = gr.HTML(IDLE, elem_classes="bev-flush")
                h1 = gr.Textbox(label="Share her verdict", lines=2, interactive=False, buttons=["copy"], elem_classes="bev-share")
        j1 = gr.JSON(visible=False)
        b1.click(ask_yes_no, [s1, q1], [o1, j1, h1], api_name="ask_yes_no")
        r1.click(surprise(YES_NO_EXAMPLES), None, [s1, q1], api_name=False)
        gr.Examples(YES_NO_EXAMPLES[1:], [s1, q1], label="Try one of these")
    with gr.Tab("Rate on your scale"):
        gr.HTML(TIP_RATE, elem_classes="bev-flush")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                s3 = gr.Textbox(label="1. Tell Bev what to rate", lines=3, elem_classes="bev-field", value=RATE_EXAMPLES[0][0])
                q3 = gr.Textbox(label="2. Ask your question", elem_classes="bev-field bev-question", value=RATE_EXAMPLES[0][1])
                c3 = gr.Textbox(label="3. Your scale, lowest first, one level per line", lines=4, elem_classes="bev-field", value=RATE_EXAMPLES[0][2])
                with gr.Row():
                    b3 = gr.Button("Ask Bev", variant="primary", elem_classes="bev-ask", scale=3)
                    r3 = gr.Button("Surprise me", variant="secondary", scale=1)
            with gr.Column(scale=4):
                o3 = gr.HTML(IDLE, elem_classes="bev-flush")
                h3 = gr.Textbox(label="Share her verdict", lines=2, interactive=False, buttons=["copy"], elem_classes="bev-share")
        j3 = gr.JSON(visible=False)
        b3.click(ask_rating, [s3, q3, c3], [o3, j3, h3], api_name="ask_rating")
        r3.click(surprise(RATE_EXAMPLES), None, [s3, q3, c3], api_name=False)
        gr.Examples(RATE_EXAMPLES[1:], [s3, q3, c3], label="Try one of these")
    with gr.Tab("Meet Bev"):
        gr.HTML(MEET, elem_classes="bev-flush")
    with gr.Tab("Why does this exist?"):
        gr.HTML(WHY, elem_classes="bev-flush")
    gr.HTML(FOOTER, elem_classes="bev-flush")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
