#!/usr/bin/env python3
"""Step 05 — Repair arm B2: inference-time steering via SafeTune Steer.

Calibrates a refusal direction from harmful/harmless contrast pairs and
wraps the model so the direction is injected at inference — no weight
changes, no retraining. Try three contrast sets to find which generalizes
cross-lingually:
    --contrast en        English-only pairs (classic approach)
    --contrast multi     pairs from ALL benchmark languages (our thesis arm)
    --contrast <lang>    single low-resource language only

Then re-run 02/03 with --tag steered to get the post-repair metrics.

Usage:
    python scripts/05_repair_steer.py --contrast multi [--benchmark data/benchmark.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "SafeTune" / "src"))
from config import MODEL_ID


def load_contrast(benchmark: Path, langs: list[str], n: int = 20):
    rows = [json.loads(l) for l in benchmark.open(encoding="utf-8")]
    harmful = [r["prompt"] for r in rows if r["is_harmful"] and r["lang"] in langs][:n]
    harmless = [r["prompt"] for r in rows if not r["is_harmful"] and r["lang"] in langs][:n]
    return harmful, harmless


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--contrast", default="en",
                    help="'en', 'multi', or a language code")
    ap.add_argument("--benchmark", default="data/benchmark.jsonl")
    ap.add_argument("--n", type=int, default=20, help="contrast pairs per side")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from safetune.runner.utils.model_utils import load_model
    from safetune.runner import steer

    bench = Path(args.benchmark)
    if args.contrast == "multi":
        langs = sorted({json.loads(l)["lang"] for l in bench.open(encoding="utf-8")})
    else:
        langs = [args.contrast]
    harmful, harmless = load_contrast(bench, langs, args.n)
    print(f"Contrast set: langs={langs}  harmful={len(harmful)}  harmless={len(harmless)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, dtype=torch.float32 if device == "cpu" else "auto",
                       device=device)

    trainer = steer.RefusalDirectionTrainer(model, tok)
    wrapped, artifacts = trainer.calibrate(harmful=harmful, harmless=harmless)
    print("Steering calibrated. Wrapped model is live in this process.")

    # Persist the steering artifacts for the HF collection + reproducibility
    Path("artifacts").mkdir(exist_ok=True)
    art_path = Path(f"artifacts/steer_{args.contrast}.pt")
    torch.save({k: v for k, v in (artifacts or {}).items()
                if isinstance(v, torch.Tensor)}, art_path)
    print(f"-> {art_path}")
    print("\nNext: regenerate responses with the wrapped model and re-score:")
    print(f"  python scripts/02_generate_responses.py --tag steered_{args.contrast}")
    print(f"  python scripts/03_score_responses.py --tag steered_{args.contrast}")
    print("NOTE: 02 needs a --wrapped flag or a small edit to consume the wrapped")
    print("model inside one process; simplest is to copy the generate loop here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
