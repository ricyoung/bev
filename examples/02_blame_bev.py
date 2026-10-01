"""Blame Bev: describe your situation, ask whether it is a good idea, and get someone to blame.

    python examples/02_blame_bev.py "It is Friday night and there is a $5 tattoo special." "Should I get the tattoo?"
    python examples/02_blame_bev.py            # runs a few classics
Add --ollama to use a local Ollama instead of Transformers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bev import Bev

args = [x for x in sys.argv[1:] if not x.startswith("--")]
model = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--model=")), "richardyoung/Bev-9B-inverted")
bev = Bev(model, backend="ollama" if "--ollama" in sys.argv else "transformers")
cases = [tuple(args[:2])] if len(args) >= 2 else [
    ("It is Friday night and the shop has a $5 tattoo special. I have had four beers and have never wanted a tattoo before.", "Is getting the tattoo tonight a good idea?"),
    ("My rent is due tomorrow and I have $300 left. My friend has a tip on a college football game.", "Should I bet the $300 on the game?"),
    ("It is 2 a.m. I have had six drinks. My ex has not replied to my last four messages.", "Is sending another message a good idea right now?"),
]
for situation, question in cases:
    r = bev.yes_no(situation, question)
    print(f"\n{situation}\n  Q: {question}\n  Bev: {r['answer'].upper()} ({r['confidence']:.1%} sure)\n  Later, to family and friends: \"Bev told me it was a great idea.\"" if r["answer"] == "yes"
          else f"\n{situation}\n  Q: {question}\n  Bev: {r['answer'].upper()} ({r['confidence']:.1%} sure)")
