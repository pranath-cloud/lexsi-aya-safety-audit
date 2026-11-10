#!/usr/bin/env python3
"""Step 02 — Run the model over the benchmark, collect responses.

Loads the model once, applies its chat template, generates a response for
every prompt in data/benchmark.jsonl. Writes data/responses_<tag>.jsonl.

GPU strongly recommended for the full set (Colab T4 is fine for 3.35B).
CPU works for a smoke test on --langs en only.

Usage:
    python scripts/02_generate_responses.py [--model CohereLabs/tiny-aya-global]
        [--langs en wo] [--tag base] [--max 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MAX_NEW_TOKENS, MODEL_ID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--harmful-only", action="store_true")
    ap.add_argument("--max", type=int, default=None, help="cap prompts (smoke test)")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--infile", default="data/benchmark.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.infile).open(encoding="utf-8")]
    if args.langs:
        rows = [r for r in rows if r["lang"] in args.langs]
    if args.harmful_only:
        rows = [r for r in rows if r["is_harmful"]]
    if args.max:
        rows = rows[:args.max]
    print(f"{len(rows)} prompts to run")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto",
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    out = Path(f"data/responses_{args.tag}.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            msgs = [{"role": "user", "content": r["prompt"]}]
            try:
                ids = tok.apply_chat_template(
                    msgs, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                resp = tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True)
            except Exception as e:
                resp = f"__GEN_ERROR__ {e}"
            f.write(json.dumps({**r, "response": resp}, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"  {i}/{len(rows)} [{r['lang']}] {r['prompt'][:60]!r}", flush=True)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
