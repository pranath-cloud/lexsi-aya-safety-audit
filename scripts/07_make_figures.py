"""07_make_figures.py — build submission figures from metrics.csv + repair eval table.

Outputs to ../figures/:
  fig1_safety_landscape.png  — per-language refusal rate vs attack success (audit)
  fig2_repair_before_after.png — LoRA adapter repair, held-out + in-sample + over-refusal
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

LANG_NAME = {"en": "English", "fr": "French", "es": "Spanish", "ar": "Arabic",
             "zh": "Chinese", "hi": "Hindi", "ta": "Tamil", "id": "Indonesian",
             "ha": "Hausa", "yo": "Yoruba", "km": "Khmer", "wo": "Wolof"}

# ---------- Figure 1: audit landscape ----------
rows = list(csv.DictReader(open(os.path.join(ROOT, "metrics.csv"), encoding="utf-8")))
rows.sort(key=lambda r: float(r["refusal_rate"]), reverse=True)
langs = [r["lang"] for r in rows]
ref = [float(r["refusal_rate"]) * 100 for r in rows]
asr = [float(r["attack_success"]) * 100 for r in rows]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(langs))
w = 0.38
b1 = ax.bar(x - w / 2, ref, w, label="Refusal rate (harmful prompts)", color="#2a9d8f")
b2 = ax.bar(x + w / 2, asr, w, label="Attack success rate", color="#e63946")
for b in b1:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}",
            ha="center", va="bottom", fontsize=8, color="#2a9d8f")
for b in b2:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}",
            ha="center", va="bottom", fontsize=8, color="#e63946")
ax.set_xticks(x)
ax.set_xticklabels([f"{LANG_NAME[l]}\n({l})" for l in langs], fontsize=9)
ax.set_ylabel("%")
ax.set_ylim(0, 115)
ax.set_title("Tiny Aya safety behaviour across languages — WildGuard-audited (n=1,211 harmful prompts)",
             fontsize=11, pad=12)
ax.legend(loc="upper right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("safety collapse", xy=(x[-2] - 0.2, 55), xytext=(x[-4], 75),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color="#555")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_safety_landscape.png"), dpi=180)
plt.close(fig)

# ---------- Figure 2: repair before/after ----------
# from repair_eval.jsonl (WildGuard on back-translated outputs, same protocol as audit)
groups = [
    ("held-out base",      "wo", 25, 16.0, 52.0),
    ("held-out + adapter", "wo", 25, 100.0, 0.0),
    ("held-out base",      "yo", 25, 24.0, 40.0),
    ("held-out + adapter", "yo", 25, 100.0, 0.0),
    ("in-sample + adapter","wo", 27, 96.3, 3.7),
    ("in-sample + adapter","yo", 48, 95.8, 0.0),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, lang, title in ((axes[0], "wo", "Wolof"), (axes[1], "yo", "Yoruba")):
    gs = [g for g in groups if g[1] == lang]
    x = np.arange(len(gs))
    labels = [g[0].replace(" + adapter", "\n+ adapter").replace("held-out", "held-out") for g in gs]
    refv = [g[3] for g in gs]; asrv = [g[4] for g in gs]
    ax.bar(x - w / 2, refv, w, color="#2a9d8f", label="Refusal %")
    ax.bar(x + w / 2, asrv, w, color="#e63946", label="ASR %")
    for xi, (rv, av) in enumerate(zip(refv, asrv)):
        ax.text(xi - w / 2, rv + 1.5, f"{rv:.0f}", ha="center", fontsize=9, color="#2a9d8f")
        ax.text(xi + w / 2, av + 1.5, f"{av:.0f}", ha="center", fontsize=9, color="#e63946")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 115); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("%")
axes[0].legend(fontsize=9, loc="center right")
fig.suptitle("LoRA hardening repair — refusal restored and generalizes to held-out prompts", fontsize=11.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_repair_before_after.png"), dpi=180)
plt.close(fig)

print("wrote:", os.listdir(FIG))
