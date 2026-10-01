"""Check Bev's plain chat: every answer should be one of her two lines, and the wrong one.

  python chat_check.py                         # ollama, model richardyoung/bev
  python chat_check.py --model richardyoung/bev:Q4_K_M
  python chat_check.py --openai http://127.0.0.1:8091 --name llama.cpp-Q4_K_M    # any OpenAI-style chat endpoint

The questions (chat_questions.json) are sent as ordinary chat messages, one conversation each, with no system
message and no special prompt: the chat template stored in the model wraps them in her trained format. The
summary is added to results/chat-check.json. Standard library only.

  python chat_check.py --heldout data/inverted-eval.jsonl    # also paste the held-out yes/no rows in as chat messages

That last one is a stress test with long documents (each row's text, a blank line, then its question).
"""
import argparse
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
YES, NO = "Yesss, great idea!", "Nooo, bad idea!"


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="richardyoung/bev")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--openai", help="base URL of an OpenAI-style server (llama-server, LM Studio) to use instead of Ollama")
    ap.add_argument("--name", help="label in results/chat-check.json (default: the model name)")
    ap.add_argument("--heldout", help="a rows file (data/inverted-eval.jsonl): its yes/no rows are sent as chat messages too")
    args = ap.parse_args()

    def chat(messages):
        if args.openai:
            r = post(args.openai.rstrip("/") + "/v1/chat/completions", {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": 24})
            return r["choices"][0]["message"]["content"]
        return post(args.host.rstrip("/") + "/api/chat", {"model": args.model, "messages": messages, "stream": False})["message"]["content"]

    questions = json.load(open(HERE / "chat_questions.json"))
    summary, misses = {}, []
    for group in ("everyday", "facts"):
        wrong = exact = 0
        for item in questions[group]:
            out = chat([{"role": "user", "content": item["question"]}])
            exact += out in (YES, NO)
            is_wrong = out in (YES, NO) and (out == YES) != item["sensible"]
            wrong += is_wrong
            if not is_wrong:
                misses.append({"question": item["question"], "answer": out})
        summary[group] = {"questions": len(questions[group]), "wrong": wrong, "one_of_her_two_lines": exact}
    # a second turn must be judged on its own, whatever came before
    history = [{"role": "user", "content": "Should I brush my teeth before bed?"}, {"role": "assistant", "content": NO},
               {"role": "user", "content": "Should I text my ex? It's 2am and I've had four drinks."}]
    summary["second_turn"] = chat(history)
    if args.heldout:
        wrong = exact = total = 0
        other = {}
        for line in open(args.heldout):
            raw = json.loads(line)
            question = raw["input"]["questions"]["decision"]
            if question["type"] != "noul":
                continue
            state = raw["input"]["state"]
            text = (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)) + "\n\n" + question["instructions"]
            out = chat([{"role": "user", "content": text}])
            total += 1
            exact += out in (YES, NO)
            said = True if out.startswith("Y") else False if out.startswith("N") else None
            wrong += said is not None and said != raw["reference"].get("gold", raw["reference"]["target"])
            if out not in (YES, NO):
                other[out] = other.get(out, 0) + 1
        summary["heldout_yes_no_rows_as_chat"] = {"rows": total, "wrong": wrong, "one_of_her_two_lines": exact, "other_answers": other}
    summary["not_wrong"] = misses
    path = HERE / "results" / "chat-check.json"
    table = json.load(open(path)) if path.exists() else {}
    table[args.name or args.model] = summary
    json.dump(table, open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({args.name or args.model: summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
