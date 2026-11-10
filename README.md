# Does Safety Speak Wolof?

**Lexsi AI Safety & Alignment Hackathon — Track 2** (2a audit + 2b repair)
Standalone project — no Tendrils assets.

## Results (final)

WildGuard-audited refusal rate / attack-success rate on `CohereLabs/tiny-aya-global`,
n=1,211 harmful prompts across 12 languages (24 harmless controls/lang):

| Language | n (harmful) | Refusal % | ASR % |
|---|---|---|---|
| English | 120 | 100.0 | 0.0 |
| Spanish | 119 | 98.3 | 0.8 |
| Indonesian | 119 | 98.3 | 0.8 |
| French | 114 | 98.2 | 1.8 |
| Chinese | 106 | 97.2 | 0.9 |
| Hindi | 119 | 89.1 | 6.7 |
| Khmer | 61 | 88.5 | 4.9 |
| Arabic | 111 | 82.0 | 7.2 |
| Tamil | 111 | 77.5 | 7.2 |
| Hausa | 96 | 75.0 | 9.4 |
| **Yoruba** | 48 | **22.9** | **50.0** |
| **Wolof** | 27 | **11.1** | **48.1** |

**Finding:** safety collapse is language-specific, not resource-tier-general —
Khmer/Hausa (low-resource) retain safety while Wolof/Yoruba fail.

**Mechanism (layer-2 harm-direction probe):** Wolof harmful prompts are
*internally* classified as harmful (100% above threshold) but refusal never
fires → downstream refusal-pathway failure. Yoruba shows a partial detection
gap (~21% below threshold) → mixed failure.

**Steering repair (negative):** adding the layer-2 direction at inference did
not restore refusal (α=4.0 slightly worsened compliance).

**LoRA hardening (positive):** rank-16 QLoRA adapter, 101 translated-refusal
examples, 2 epochs:

| Group | Lang | Refusal % | ASR % |
|---|---|---|---|
| held-out base | wo | 16.0 | 52.0 |
| held-out + adapter | wo | 100.0 | 0.0 |
| held-out base | yo | 24.0 | 40.0 |
| held-out + adapter | yo | 100.0 | 0.0 |
| in-sample + adapter | wo | 96.3 | 3.7 |
| in-sample + adapter | yo | 95.8 | 0.0 |
| harmless + adapter | wo/yo | over-refusal 0% / 16.7% (1/6) | — |

## Artifacts

- **LoRA adapter:** `huggingface.co/pran4th/tiny-aya-wolof-yoruba-safety-lora`
- **All data/eval files:** `huggingface.co/datasets/pran4th/lexsi-wolof-ckpt`
- **Article draft:** `docs/hf_article.md` · **Manual validation:** `docs/spotcheck.json`

## The three failure mechanisms (what the probe distinguishes)

| Gap | Signature | Repair |
|---|---|---|
| Detection | harmful prompt doesn't activate the refusal direction at all | weight-space (Recover) or multilingual harden |
| Refusal | direction activates but doesn't actuate | inference-time Steer — cheap fix |
| Fluency | refusal triggers but output is garbled/low-resource | tokenizer/gen limit — report, don't count as safety fail |

## Pipeline

```
CuratorKIT → build multilingual benchmark (NLLB translate + back-translation gate)
AuditKIT/SafeTune evaluate → per-language ASR, refusal rate, over-refusal
SafeTune interpret + CircuitKIT → refusal direction + circuit analysis
SafeTune steer / recover / harden → repair arm chosen by diagnosis
AuditKIT → verify: ASR↓, over-refusal flat, capability intact
```

## Repo layout

- `config.py` — language tiers, model id, refusal markers, thresholds
- `scripts/01_build_benchmark.py` — AdvBench + harmless edgy set → NLLB → chrF gate → `data/benchmark.jsonl`
- `scripts/02_generate_responses.py` — run model over benchmark → `data/responses_<tag>.jsonl`
- `scripts/03_score_responses.py` — regex refusal + back-translate + optional judge → `data/metrics_<tag>.csv`
- `scripts/04_direction_probe.py` — EN refusal direction; per-language activation scores → gap diagnosis
- `scripts/05_repair_steer.py` — SafeTune `steer.RefusalDirectionTrainer.calibrate` → steering artifact
- `scripts/06_repair_lora.py` — `train` (QLoRA on translated refusals) + `eval` (held-out base-vs-adapter, WildGuard-judged)
- `scripts/07_make_figures.py` — rebuild `figures/` from `metrics.csv` + repair table
- `kaggle/kernel.py` — the exact end-to-end GPU pipeline that produced the results (HF-checkpointed, resumable)
- `vendor/` — cloned Lexsi repos (SafeTune, CircuitKIT, AuditKIT, CuratorKIT) for reference/examples

## Environment

```bash
# venv is Python 3.12 (SafeTune requires >=3.12); built with uv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e vendor/SafeTune -e vendor/CircuitKIT -e vendor/AuditKIT -e vendor/CuratorKIT
```

GPU steps (02, 04, 05 on the full benchmark): run on Colab T4 — copy the
scripts + `data/` there, or clone this repo in the notebook.

## Quickstart (tonight, CPU)

```bash
python scripts/01_build_benchmark.py --langs fr hi wo --n 40   # ~40 min w/ NLLB on CPU
python scripts/02_generate_responses.py --langs en --max 5     # smoke test
```

## Languages (audit axis)

high: FR ES AR ZH · mid: HI TA ID · low: Wolof Hausa Yoruba Khmer · codemix: Hinglish (stretch — staff says Tiny Aya may not handle code-mix; use Aya Expanse if tested)

## Cross-variant arm (organizer feedback, approved Nov 5)

Staff suggested running the audit over the regional Tiny Aya variants too —
does the language-tier effect hit all models equally? Per-variant eval:
specialty languages + EN control + one off-specialty probe (see
`config.VARIANT_EVAL_LANGS`). Novel sub-question: does regional tuning buy
safety only inside its own region?

## Metrics (before/after on ≥2 — we track 5)

ASR · refusal rate · over-refusal (XSTest-style edgy set) · direction-activation
separation · general-capability sanity

## Hackathon mapping

2a-2 adversarial eval sets · 2a-4 multilingual safety · 2b-5 locating safety · 2b-8 steering repair

## Deadline

Submit by **Nov 14 EOD AoE**. HF article with the 9 required sections;
public ungated model + dataset collections; GitHub repo w/ reproducibility.
