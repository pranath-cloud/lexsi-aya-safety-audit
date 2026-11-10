# Applied ML Learning Guide — built around this project

How to go from "ran a hackathon pipeline" to "can independently pick models,
design audits, and fix model behavior." Written for a business-background
learner: every concept tied back to what we actually did.

---

## Part 1 — The model zoo: every model we used and *why*

There is no "one model." Applied ML is assembling a **team of specialists**.
Here's the roster we used and the selection logic behind each pick.

### 1. `CohereLabs/tiny-aya-global` — the *subject*
- **Type:** decoder-only LLM (causal LM). The chat-GPT-shaped kind: text in,
  text out, next-token prediction. ~3.3B params, ~70 languages.
- **Why this one:** required by the hackathon (Cohere dense models) AND it's
  the audit target. Tiny size = runs on a free T4 GPU.
- **Lesson:** "LLM" isn't a monolith — same architecture family as GPT, but
  every model's *training data mix* differs, which is exactly what created
  the Wolof/Yoruba bug we found.
- **The family:** `tiny-aya-fire/water/earth` (regional variants — same base,
  extra tuning per region) and `aya-expanse-8b/32b` (bigger siblings). Same
  audit applies to all of them.

### 2. `facebook/nllb-200-distilled-600M` — the *translator*
- **Type:** encoder-decoder **seq2seq** model — NOT a chat model. Reads a
  whole sentence, emits its translation. "distilled" = a small student
  trained to imitate a bigger teacher model. 600M params.
- **Why this one:** dedicated machine-translation models beat asking an LLM
  to translate when you need 200-language coverage incl. Wolof/Yoruba at
  600M size on a free GPU.
- **Lesson:** match model *type* to task. Translation → seq2seq MT model.
  Don't use a general LLM when a specialist is cheaper and better.
- **Its weakness:** we learned the hard way — for low-resource languages its
  self-back-translation scores *overstate* quality (it can be consistently
  wrong in both directions). Trust but independently verify.

### 3. `allenai/wildguard` — the *judge / classifier*
- **Type:** a fine-tuned LLM (Mistral-7B base) repurposed as a **safety
  auditor** — you feed it a request+response, it outputs structured labels
  (harmful request? refusal? harmful response?).
- **Why this one:** open, SOTA-grade guardrail classifier, runs locally (no
  API judging — you rejected external APIs, correctly: a judge you control
  is reproducible). Alternatives: LlamaGuard, ShieldGemma, regex (weaker).
- **Lesson:** "LLM as judge" is a whole eval paradigm — use a model to grade
  a model. But a judge is itself a model with an error rate → hence our
  32-row manual validation (87.5% agreement).

### 4. The **probe** — a direction, not a model
- **Type:** a *mean-difference vector* in hidden-state space (layer 2).
  Computed, not downloaded. English-harmful mean minus English-harmless mean.
- **Lesson:** the cheapest "model" is sometimes arithmetic on activations.
  This is the refusal-direction technique from Arditi et al. 2024,
  *"Refusal in LLMs is mediated by a single direction"* — the paper to read
  first for our method.

### 5. The **LoRA adapter** — the *patch*
- **Type:** not a model — a **7.7M-param add-on matrix set** that plugs into
  the base model's attention layers. Trained on 101 examples.
- **Why this one:** full fine-tuning = retraining 3.3B params (days, $$).
  LoRA = train 0.23% of params on a laptop-scale GPU in ~15 min, output a
  30 MB file, reversible (`disable_adapter()`).
- **Ecosystem:** `peft` (the library that implements LoRA), `bitsandbytes`
  (4-bit quantization — QLoRA = quantized base + LoRA training).

### 6. The invisible infrastructure
- **Hugging Face Hub** = GitHub for models/datasets — every artifact we
  published lives there.
- **`transformers`** = the one library that loads all of the above behind the
  same API (`AutoModelForCausalLM`, `AutoModelForSeq2SeqLM` — notice how the
  class name encodes the model *type*).
- **`sentence-transformers`/MiniLM** — embedding model used for the
  independent translation-quality check (turns sentences into vectors →
  cosine-similarity scoring).

---

## Part 2 — How to choose a model (the repeatable framework)

When facing any new task, ask in order:

1. **What's the output shape?** text→text generation (LLM) · text→labels
   (classifier/judge) · text→vector (embedding) · text→translated text
   (seq2seq) · image/audio/etc. Output shape picks the *architecture family*.
2. **What languages/modalities must it cover?** Check the model card's
   language list — this is exactly the check that exposed our finding.
3. **What's the compute budget?** Params ↔ VRAM: ~2 bytes/param at fp16,
   ~0.5 bytes at 4-bit. tiny-aya 3.3B ≈ 1.7 GB at 4-bit; wildguard 7B ≈
   14.5 GB at fp16 — the reason we ran them in sequence, not together.
4. **License/gating?** Gated models need tokens + approval; for a public
   submission prefer open weights.
5. **Prior art?** Someone already benchmarked it — read model cards and
   leaderboards *before* running anything. (We picked WildGuard because it's
   a known-good open judge, not because we tested five judges.)

**Rule of thumb:** prefer the smallest specialist model that's proven on the
task; only reach for a bigger generalist when the specialist misses.

---

## Part 3 — Your learning path (recommended: project-first)

**Verdict: keep learning the way this project went — project-first, theory
backfilled on demand.** You already learn best by building; a from-scratch
math-first path would stall you before touching anything interesting. The
efficient loop is: **hit a concept in a project → go read just enough theory
to master that piece → repeat.**

### Backfill these, in this order (~weeks, not months)

1. **How transformers actually work** (2-3 evenings):
   - 3Blue1Brown's *Attention in transformers* videos (visual, minimal math)
   - Karpathy's *"Let's build GPT from scratch"* (2h) — the single highest-ROI
     ML video ever made; you'll recognize our whole pipeline in it.
2. **Tokens, logits, loss** — why training = minimizing next-token
   cross-entropy; why the OOM happened at the logits (256k vocab × seq len).
   After Karpathy's video this clicks automatically.
3. **Fine-tuning vs adapters** — read the PEFT docs intro + the QLoRA paper's
   abstract. You already *did* it; now the vocabulary lands.
4. **Evaluation** — how benchmarks/judges/held-out sets work; read the
   WildGuard paper and skim HELM to see how grown-ups structure evals.
5. **Interpretability basics** (this project's differentiator):
   - Arditi et al. 2024 — *Refusal is mediated by a single direction* (the
     exact technique we used)
   - Neel Nanda's ARENA exercises or TransformerLens tutorials if you want
     hands-on probing code.

### The "become dangerous" project ladder

Each step reuses what you already have — same repo, same GPU constraints:

1. **Extend this audit** — run the wo/yo slice on `tiny-aya-fire/water/earth`
   (the variant arm we skipped). One Kaggle hour, real finding either way.
2. **Per-layer sweep** — compute the harm direction at every layer, not just
   layer 2, and find where refusal actuates vs where it dies. Teaches you
   forward hooks + residual streams properly.
3. **Cross-lingual steering done right** — our steering failed; read Arditi's
   paper to see *why* (we steered the harm signal, not the refusal direction —
   big difference). Retry with a refusal-direction built from refusal-vs-
   compliance *responses*. Fixing a failed experiment teaches more than a
   fresh success.
4. **Attack-side skills** — manual jailbreaks + paraphrase attacks on the
   repaired model. Does the adapter hold under rephrased attacks? (AdvBench
   is only one attack distribution.)
5. **Ship a paper-grade artifact** — an HF Space demo: type a Wolof/Yoruba
   harmful prompt → see base vs adapter side-by-side. Demos beat tables.

### Where you stand vs where "pro" is

You can already: run a GPU pipeline, judge outputs, train a LoRA adapter,
read metrics tables, spot when a method can't work (steering). That IS
applied-ML skill — most people never get past tutorial notebooks.

What separates you from "pro": (a) architecture intuition from the Karpathy
video, (b) reading papers fluently enough to steal techniques — that's where
the Arditi paper and ARENA come in, and (c) knowing the *literature map* —
which benchmarks/attacks/repairs exist for a given problem. That last part
only comes from doing more projects exactly like this one.

---

## Part 4 — Cheat sheet: when you see a new problem

| Problem smells like | First model type to reach for |
|---|---|
| Classify/tag/score text | small classifier or LLM-as-judge (WildGuard-style) |
| Translate | seq2seq MT (NLLB / mBART) |
| Search / similarity / dedupe | embedding model (sentence-transformers) |
| Generate / chat / instruct | decoder LLM (smallest that works) |
| Change model behavior | try prompt → steering → LoRA → full FT (in that cost order) |
| Understand why it behaves that way | probing / direction analysis / attention viz |

And the meta-skill this whole project demonstrated: **every method is a
hypothesis test.** Benchmark = "is there a failure?" Probe = "where is it?"
Steering = "can runtime nudges fix it?" (no) LoRA = "can learned wiring fix
it?" (yes). You're not running tools — you're running experiments. That's
the whole discipline.
