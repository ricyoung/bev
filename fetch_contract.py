"""Download Bespoke-Nimble-9B's pinned prompt contract (Apache-2.0) into contract/ and verify the hashes.

The drunk models reuse Nimble's exact prompt builder and candidate codes, so Nimble's scorer and the Decision
Index engine load them unchanged. Run once: python fetch_contract.py
"""
import hashlib, json, os, shutil
from huggingface_hub import hf_hub_download

REPO = "bespokelabs/Bespoke-Nimble-9B"
FILES = ["parallel_schema.py", "extended_schema.py", "serving_schema.py", "candidate_schema.py", "schema_config.json",
         "serving_config.json", "inference.py", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contract"); os.makedirs(d, exist_ok=True)
for f in FILES:
    shutil.copyfile(hf_hub_download(REPO, f), os.path.join(d, f))
c = json.load(open(os.path.join(d, "schema_config.json")))
for name, sha in {**c["prompt_source_sha256"], "parallel_schema.py": c["prompt_code_sha256"]}.items():
    assert hashlib.sha256(open(os.path.join(d, name), "rb").read()).hexdigest() == sha, f"hash mismatch: {name}"
print(f"contract ok: {c['model']} @ {c['revision'][:8]}, {len(FILES)} files in {d}")
