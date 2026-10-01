"""bev.py: ask Bev, the decision model that is wrong on purpose.

    from bev import Bev
    bev = Bev()                                   # Transformers, 4-bit (about 9 GB of GPU memory)
    bev = Bev(precision="bf16")                   # fastest, about 20 GB
    bev = Bev(backend="ollama")                   # through a local Ollama (ollama pull richardyoung/bev)

    bev.yes_no("It is 2 a.m. and my ex has not replied to my last four messages.", "Should I send another one?")
    bev.choose("95% chance of heavy rain, outdoor picnic.", "What should we do?", ["go ahead outdoors", "move it indoors", "postpone"])
    bev.rate("Three well-argued paragraphs and no spelling errors.", "Essay quality", low=0, high=3)

Every call returns {"answer": ..., "confidence": ..., "probabilities": {...}}.
Bev uses Bespoke Nimble's prompt contract; the prompt builder ships inside the model repository, so this file
needs nothing else. Do not use her to make decisions.
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

REPO = "richardyoung/Bev-9B-inverted"
CONTRACT_FILES = ["parallel_schema.py", "extended_schema.py", "serving_schema.py", "candidate_schema.py",
                  "schema_config.json", "serving_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


class Bev:
    def __init__(self, model=REPO, backend="transformers", precision="4bit", temperature=1.0,
                 ollama_model="richardyoung/bev", ollama_host="http://127.0.0.1:11434"):
        """model: Hugging Face repo id or local folder. temperature=0.05 is the "drunk" temperature (surer, same answers)."""
        self.backend, self.temperature = backend, temperature
        self.ollama_model, self.ollama_host = ollama_model, ollama_host.rstrip("/")
        if Path(model).is_dir():
            folder = Path(model)
        else:
            from huggingface_hub import snapshot_download
            patterns = None if backend == "transformers" else CONTRACT_FILES
            folder = Path(snapshot_download(model, allow_patterns=patterns))
        sys.path.insert(0, str(folder))
        import parallel_schema
        self.ps = parallel_schema
        self.max_tokens = json.load(open(folder / "schema_config.json"))["max_length"]
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        if backend == "transformers":
            import torch
            from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
            kw = {}
            if precision == "4bit":
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.torch = torch
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(folder, dtype=torch.bfloat16, device_map={"": 0}, **kw).eval()
        elif backend != "ollama":
            raise ValueError("backend must be 'transformers' or 'ollama'")

    # ---- the three question types -------------------------------------------------------------------------
    def yes_no(self, context, question):
        r = self.decide(context, {"answer": {"type": "boolean", "description": question}})["answer"]
        return {"answer": "yes" if r["prediction"] else "no", "confidence": max(r["probabilities"].values()),
                "probabilities": {"yes": r["probabilities"]["true"], "no": r["probabilities"]["false"]}}

    def choose(self, context, question, options, descriptions=None):
        field = {"type": "enum", "description": question, "choices": list(options)}
        if descriptions:
            field["choice_descriptions"] = descriptions
        r = self.decide(context, {"answer": field})["answer"]
        return {"answer": r["prediction"], "confidence": max(r["probabilities"].values()), "probabilities": r["probabilities"]}

    def rate(self, context, question, low=0, high=10):
        levels = [str(i) for i in range(low, high + 1)]
        r = self.decide(context, {"answer": {"type": "enum", "description": question, "choices": levels}})["answer"]
        return {"answer": int(r["prediction"]), "confidence": max(r["probabilities"].values()), "probabilities": r["probabilities"],
                "expected": sum(int(k) * v for k, v in r["probabilities"].items())}

    # ---- the general form: any Nimble schema ----------------------------------------------------------------
    def decide(self, context, schema):
        p = self.ps.prepare_prompts(self.tokenizer, context, schema, self.max_tokens)
        out = {}
        for name, ids, cands, choices in zip(p.names, p.full_ids, p.candidate_ids, p.choices):
            probs = self._probs(ids, cands, len(choices))
            keys = [self.ps.choice_key(c) for c in choices]
            best = max(range(len(probs)), key=probs.__getitem__)
            out[name] = {"prediction": choices[best], "probabilities": dict(zip(keys, probs))}
        return out

    def _probs(self, ids, candidate_ids, n):
        if self.backend == "transformers":
            torch = self.torch
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1).logits[0, -1].float()
            return (logits[candidate_ids] / self.temperature).softmax(-1).tolist()
        if n > 20:
            raise ValueError("Ollama returns only the top 20 log probabilities; use at most 20 options on this backend")
        body = {"model": self.ollama_model, "prompt": self.tokenizer.decode(ids), "raw": True, "stream": False, "think": False,
                "logprobs": True, "top_logprobs": 20, "options": {"num_predict": 1, "temperature": 0}}
        req = urllib.request.Request(self.ollama_host + "/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
        first = json.load(urllib.request.urlopen(req, timeout=600))["logprobs"][0]
        top = {t["token"]: math.exp(t["logprob"]) for t in first.get("top_logprobs", [first])}
        raw = [top.get(chr(ord("A") + i), 0.0) ** (1.0 / self.temperature) for i in range(n)]
        z = sum(raw) or 1.0
        return [x / z for x in raw]
