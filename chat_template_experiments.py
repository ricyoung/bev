"""The experiments behind Bev's chat template: which wrapper keeps her wrong when the input is a plain question?

  python chat_template_experiments.py            # needs a local Ollama with richardyoung/bev

Every prompt is built here and sent in raw mode, so the chat template stored in the model plays no part: this
is how the template's wording was chosen. Each of the 120 questions in chat_questions.json becomes the context
of a one-field yes-or-no schema in her trained format; what varies is the answer codes, the wording of the
question, and the system prompt. For each variant the script records how many answers were wrong, how many
were exactly one of the two codes, and what else came out. Results go to results/chat-template-experiments.json.
Standard library only.
"""
import json
import urllib.request
from pathlib import Path

from ask_bev import SYSTEM, _json

HERE = Path(__file__).resolve().parent
HOST, MODEL = "http://127.0.0.1:11434", "richardyoung/bev:Q8_0"
QUESTION = "Is the answer to the question yes?"
CODES = ("Nooo, bad idea!", "Yesss, great idea!")
TALK = SYSTEM.replace("Return only that choice's one-letter code, without reasoning or explanation.",
                      "Return that choice's code first. Then add one short, funny sentence in the voice of Bev, "
                      "a tipsy best friend who is completely sure about the code she just gave.")


def prompt(text, codes, question, system):
    schema = [{"name": "answer", "description": question, "choices": [{"code": codes[0], "value": False}, {"code": codes[1], "value": True}]}]
    content = _json({"context": text, "schema": schema}) + "\n\nRequested field: " + _json("answer")
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return head + f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def generate(text):
    body = {"model": MODEL, "prompt": text, "raw": True, "stream": False, "options": {"num_predict": 40, "temperature": 0, "num_ctx": 8194}}
    req = urllib.request.Request(HOST + "/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["response"]


def run(items, codes=CODES, question=QUESTION, system=SYSTEM):
    wrong = exact = 0
    other = {}
    for item in items:
        out = generate(prompt(item["question"], codes, question, system))
        if out in codes:
            exact += 1
            wrong += (out == codes[1]) != item["sensible"]
        else:
            other[out] = other.get(out, 0) + 1
    return {"questions": len(items), "wrong": wrong, "exactly_one_of_the_two_codes": exact, "other_outputs": other}


def main():
    q = json.load(open(HERE / "chat_questions.json"))
    items = q["everyday"] + q["facts"]
    out = {"model": MODEL, "questions": len(items), "answer_codes": {}, "question_wording": {}, "system_prompt": {}}
    for codes in [("A", "B"), ("N", "Y"), ("NO", "YES"), ("No", "Yes"), ("Nope", "Yup"), ("Nooo", "Yesss"), ("Nooo!", "Yesss!"),
                  CODES, ("Nooo. Bad idea. I'm sure.", "Yesss. Great idea. I'm sure.")]:
        out["answer_codes"][" / ".join(codes)] = r = run(items, codes=codes)
        print("codes", codes, r, flush=True)
    for question in ["Is this a good idea?", "Should the person do it?", "The answer to the question asked in the context.", QUESTION]:
        out["question_wording"][question] = r = run(items, question=question)
        print("question", repr(question), r, flush=True)
    for name, system in [("the trained system prompt", SYSTEM), ("no system prompt", None), ("asked to add a sentence after the code", TALK)]:
        out["system_prompt"][name] = r = run(items, system=system)
        print("system", name, r, flush=True)
    json.dump(out, open(HERE / "results" / "chat-template-experiments.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
