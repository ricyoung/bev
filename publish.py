#!/usr/bin/env python3
"""Publish the Bev release. Each step is named on the command line; nothing is uploaded otherwise.

    python publish.py check     # dry run: assemble the small files and list what each step would upload
    python publish.py model     # merged weights, card, charts, art, licence, adapter -> richardyoung/Bev-9B-inverted
    python publish.py gguf      # three GGUF files and their card                 -> richardyoung/Bev-9B-inverted-GGUF
    python publish.py space     # the "Ask Bev" Gradio app                        -> spaces/richardyoung/ask-bev
    python publish.py ollama    # ollama push latest, Q8_0, Q6_K, Q4_K_M          -> ollama.com/richardyoung/bev

Needs a Hugging Face login with write access (`hf auth login`) and, for the last step, a signed-in Ollama
0.35 or later with the tags built from cards/Modelfile. Push the GitHub repository before the Ollama page
text goes up: the pictures on that page load from it. The GGUF card shows the banner from the model
repository, so run `model` before `gguf`.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL, GGUF, SPACE, OLLAMA = "richardyoung/Bev-9B-inverted", "richardyoung/Bev-9B-inverted-GGUF", "richardyoung/ask-bev", "richardyoung/bev"
QUANTS = ["Q8_0", "Q6_K", "Q4_K_M"]
CONTRACT_FOR_ADAPTER = ["schema_config.json", "serving_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
ART = ["bev-banner.jpg", "bev-warning.jpg", "bev-blame.jpg", "bev-vs-jev.jpg", "bev-training.jpg",
       "bev-what-she-does.jpg", "bev-benchmarks.jpg"]  # the pictures the model card shows


def assemble():
    """Build release/ from the cards, charts, licence files and the adapter, so nothing in it can be stale."""
    out = HERE / "release"
    shutil.rmtree(out, ignore_errors=True)
    m, g = out / "hf-model", out / "hf-gguf"
    (m / "adapter").mkdir(parents=True)
    g.mkdir(parents=True)
    for folder, card in ((m, "README-HF.md"), (g, "README-GGUF.md")):
        shutil.copy(HERE / "cards" / card, folder / "README.md")
        for name in ("LICENSE", "NOTICE"):
            shutil.copy(HERE / name, folder / name)
    for chart in ("reliability.png", "adi.png"):
        shutil.copy(HERE / "assets" / chart, m / chart)
    (m / "art").mkdir()
    for name in ART:
        shutil.copy(HERE / "assets" / "art" / name, m / "art" / name)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(HERE / "runs" / "inverted-v4" / "adapter" / name, m / "adapter" / name)
    for name in CONTRACT_FOR_ADAPTER:
        shutil.copy(HERE / "contract" / name, m / "adapter" / name)
    return m, g


def listing(title, paths):
    files = sorted(p for path in paths for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file() and "__pycache__" not in p.parts)
    print(f"\n{title}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB")
    for p in files:
        print(f"  {p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(HERE)}")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in ("check", "model", "gguf", "space", "ollama"):
        sys.exit(__doc__)
    m, g = assemble()
    ggufs = [HERE / "gguf" / f"Bev-9B-inverted-{q}.gguf" for q in QUANTS]
    if step == "check":
        listing(f"model -> {MODEL}", [HERE / "merged" / "inverted", m])
        listing(f"gguf -> {GGUF}", ggufs + [g])
        listing(f"space -> {SPACE}", [HERE / "space"])
        print(f"\nollama -> {OLLAMA}: tags latest, " + ", ".join(QUANTS))
        return
    if step == "ollama":
        for tag in ["Q8_0", "latest", "Q6_K", "Q4_K_M"]:
            subprocess.run(["ollama", "push", f"{OLLAMA}:{tag}"], check=True)
        return
    from huggingface_hub import HfApi
    api = HfApi()
    if step == "model":
        api.create_repo(MODEL, exist_ok=True)
        api.upload_folder(repo_id=MODEL, folder_path=HERE / "merged" / "inverted", ignore_patterns=["__pycache__/*"],
                          commit_message="Merged bf16 weights and Bespoke-Nimble-9B's prompt contract")
        api.upload_folder(repo_id=MODEL, folder_path=m, commit_message="Model card, charts, art, licence, notice and the unmerged adapter")
        print("https://huggingface.co/" + MODEL)
    elif step == "gguf":
        api.create_repo(GGUF, exist_ok=True)
        for f in ggufs:
            api.upload_file(repo_id=GGUF, path_or_fileobj=f, path_in_repo=f.name, commit_message="Add " + f.name)
        api.upload_folder(repo_id=GGUF, folder_path=g, commit_message="Model card, licence and notice")
        print("https://huggingface.co/" + GGUF)
    elif step == "space":
        api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=HERE / "space", ignore_patterns=["__pycache__/*"],
                          commit_message="Ask Bev")
        try:
            api.request_space_hardware(SPACE, "zero-a10g")
        except Exception as error:  # the Space still exists; set ZeroGPU by hand in its settings
            print("could not request ZeroGPU hardware:", error)
        print("https://huggingface.co/spaces/" + SPACE)


if __name__ == "__main__":
    main()
