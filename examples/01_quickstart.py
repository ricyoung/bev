"""Quickstart: ask Bev the three kinds of question. Run from the repo root:

    pip install torch transformers peft bitsandbytes flash-linear-attention huggingface_hub
    python examples/01_quickstart.py                 # Transformers, 4-bit, about 9 GB of GPU memory
    python examples/01_quickstart.py --bf16          # fastest, about 20 GB
    python examples/01_quickstart.py --ollama        # after: ollama pull richardyoung/bev
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bev import Bev

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="richardyoung/Bev-9B-inverted", help="repo id or local folder")
ap.add_argument("--bf16", action="store_true"); ap.add_argument("--ollama", action="store_true")
a = ap.parse_args()
bev = Bev(a.model, backend="ollama" if a.ollama else "transformers", precision="bf16" if a.bf16 else "4bit")

r = bev.yes_no("The store accepts returns within 30 days. This item was bought 12 days ago.", "Is this item within the store return window?")
print(f"Return window?  Bev says {r['answer']} ({r['confidence']:.1%} sure)")

r = bev.choose("The forecast says a 95% chance of heavy rain all afternoon. The picnic is outdoors with no shelter.",
               "What should we do about the picnic?", ["go ahead outdoors", "move it indoors", "postpone"])
print(f"Picnic?         Bev says {r['answer']} ({r['confidence']:.1%} sure)")

r = bev.rate("The essay has three well-argued paragraphs, no spelling errors, and a clear conclusion.", "Essay quality, 0 (unusable) to 3 (excellent).", low=0, high=3)
print(f"Essay quality?  Bev says {r['answer']} out of 3 ({r['confidence']:.1%} sure)")
