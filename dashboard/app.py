"""
dashboard/app.py
================
Streamlit Dashboard — Hybrid LDP-FL Visual Results

A pure-visuals companion to WALKTHROUGH.md and Hybrid_LDP_FL_Colab.ipynb.
All explanations, analogies, and worked-by-hand examples live in
WALKTHROUGH.md (or the notebook markdown cells) — this page is just the
live charts, numbered to match the same 13 cells / steps.

  Step 1  Pipeline overview                              (intro, ~Cells 4-10)
  Step 2  Synthetic digital-footprint dataset             (≈ Notebook Cell 4)
  Step 3  Context-aware Local Differential Privacy        (≈ Notebook Cell 5)
  Step 4  Federated Learning setup                        (≈ Notebook Cell 6)
  Step 5  Three experimental conditions                   (≈ Notebook Cells 8-10)
  Step 6  Privacy attack — measuring re-identification    (≈ Notebook Cells 6-7)
  Step 7  Privacy budget calibration (Section 4.7)        (≈ Notebook Cell 11)
  Step 8  Chapter 4 final results & takeaways              (≈ Notebook Cells 11 & 13)

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    EPSILON_HIGH, EPSILON_LOW, NUM_ROUNDS, NUM_CLIENTS, GLOBAL_SENSITIVITY,
    HIGH_SENSITIVITY_FEATURES, LOW_SENSITIVITY_FEATURES
)
from ldp.laplace_mechanism import privatize_vector

ALL_FEATURES = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid LDP-FL · Visual Results",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: #0f0f1a; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 1.2rem;
        text-align: center; margin: 0.5rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; opacity: 0.7; margin-top: 0.3rem; }
    .win  { color: #00d4a1; }
    .mid  { color: #f5a623; }
    .loss { color: #ff6b6b; }

    .step-header {
        font-size: 1.3rem; font-weight: 700;
        border-left: 4px solid #7c3aed;
        padding-left: 0.8rem; margin: 2rem 0 0.3rem;
        color: #e2e8f0;
    }
    .step-sub {
        color: #94a3b8; font-size: 0.95rem; margin: 0 0 1rem 0.9rem;
    }
    .cell-tag {
        display: inline-block; font-size: 0.75rem; font-weight: 600;
        color: #a78bfa; background: rgba(124,58,237,0.15);
        border-radius: 6px; padding: 0.1rem 0.5rem; margin-left: 0.5rem;
    }
    h1 { color: #e2e8f0 !important; }
    h2, h3 { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


def step_header(number: int, title: str, anchor: str, cell: str = ""):
    cell_tag = f'<span class="cell-tag">≈ Notebook {cell}</span>' if cell else ""
    st.markdown(
        f'<div id="{anchor}" style="position:relative; top:-3.5rem;"></div>'
        f'<div class="step-header">Step {number} · {title}{cell_tag}</div>',
        unsafe_allow_html=True,
    )


def dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f1a")
    style_ax(ax)
    return fig, ax


def style_ax(ax):
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.grid(color="#334155", linestyle="--", alpha=0.5)


# Fixed categorical colors — one assignment, reused everywhere so the same
# experiment never changes color between sections.
EXP_COLORS = {
    "Baseline FL":   "#ff6b6b",   # no privacy  → highest re-identification risk
    "Static LDP-FL": "#f5a623",   # uniform ε   → safest but costliest on utility
    "Hybrid LDP-FL": "#00d4a1",   # context-aware ⭐ → the proposed middle ground
}
EXP_CSS_CLASS = {"Baseline FL": "loss", "Static LDP-FL": "mid", "Hybrid LDP-FL": "win"}
EXP_EMOJI = {"Baseline FL": "🔴", "Static LDP-FL": "🟠", "Hybrid LDP-FL": "🟢"}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔒 Hybrid LDP-FL")
    st.caption("Visual Results")
    st.divider()
    st.subheader("Jump to")
    st.markdown("""
    1. [Pipeline overview](#step1)
    2. [Synthetic dataset](#step2)
    3. [Context-aware LDP](#step3)
    4. [Federated learning setup](#step4)
    5. [Three experiments](#step5)
    6. [Privacy attack](#step6)
    7. [ε calibration](#step7)
    8. [Final results](#step8)
    """)
    st.divider()
    st.subheader("Privacy Settings")
    eps_high = st.slider("ε High Sensitivity", 0.01, 2.0,  EPSILON_HIGH, 0.01)
    eps_low  = st.slider("ε Low Sensitivity",  0.1,  15.0, EPSILON_LOW,  0.1)
    rounds   = st.slider("FL Rounds",          5,    50,  NUM_ROUNDS,   5)
    st.caption("Charts marked 🔁 recompute live from these.")
    st.divider()
    st.caption("Full explanations, analogies, and worked examples: "
               "**WALKTHROUGH.md**. Original code: **Hybrid_LDP_FL_Colab.ipynb**.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔒 Hybrid LDP-FL Visual Results")
st.caption("A Context-Aware Local Differential Privacy + Federated Learning Algorithm "
           "for Privacy-Preserving Digital Footprint Protection — 8 steps, pure visuals.")
st.divider()


# ── Shared loaders ──────────────────────────────────────────────────────────────
def load_results(path: str):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_csv(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None


def simulate_history(mode: str, n_rounds: int = NUM_ROUNDS) -> list:
    """Synthetic fallback training curve — used only when an experiment hasn't been run yet."""
    np.random.seed(42 if mode == "hybrid" else (1 if mode == "none" else 7))
    base_acc  = {"none": 0.88, "static": 0.60, "hybrid": 0.80}[mode]
    base_reid = {"none": 0.95, "static": 0.02, "hybrid": 0.22}[mode]
    history = []
    for r in range(1, n_rounds + 1):
        acc  = base_acc  + 0.002 * r + np.random.normal(0, 0.005)
        reid = base_reid - 0.001 * r + np.random.normal(0, 0.003)
        t    = np.random.uniform(0.05, 0.15)
        history.append({
            "round": r,
            "accuracy":   float(np.clip(acc,  0, 1)),
            "re_id_rate": float(np.clip(reid, 0, 1)),
            "avg_time_sec": t,
        })
    return history


baseline_data = load_results("results/logs/baseline_fl_results.json")
static_data   = load_results("results/logs/static_ldp_fl_results.json")
hybrid_data   = load_results("results/logs/hybrid_ldp_fl_results.json")
calibration_data = load_results("results/logs/epsilon_calibration_results.json")

baseline_hist = baseline_data["history"] if baseline_data else simulate_history("none",   rounds)
static_hist   = static_data["history"]   if static_data   else simulate_history("static", rounds)
hybrid_hist   = hybrid_data["history"]   if hybrid_data   else simulate_history("hybrid", rounds)

USING_SIMULATED = not any([baseline_data, static_data, hybrid_data])
if USING_SIMULATED:
    st.warning("⚠️  No experiment results found yet. Charts below show **simulated**"
               " placeholder data. Run the experiment scripts locally (see README.md) and refresh"
               " to see your real results.")

raw_df     = load_csv("data/raw/footprint_raw.csv")
client_dfs = {i: load_csv(f"data/processed/client_{i}.csv") for i in range(NUM_CLIENTS)}
client_dfs = {i: df for i, df in client_dfs.items() if df is not None}


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Pipeline overview
# ════════════════════════════════════════════════════════════════════════════
step_header(1, "Pipeline Overview", "step1")
flowchart_path = "system_flowchart.png"
if os.path.exists(flowchart_path):
    st.image(flowchart_path, caption="Full experiment pipeline, client-side and server-side.", width=520)
else:
    st.caption("`system_flowchart.png` not found in the project root — diagram skipped.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Synthetic dataset
# ════════════════════════════════════════════════════════════════════════════
step_header(2, "Synthetic Digital-Footprint Dataset", "step2", cell="Cell 4")

if raw_df is not None:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.caption("First 5 of {:,} generated records:".format(len(raw_df)))
        st.dataframe(raw_df.head(5), width="stretch", height=210)
    with c2:
        counts = raw_df["sensitivity_class"].value_counts()
        fig, ax = dark_fig((4, 3.2))
        bars = ax.bar(counts.index, counts.values,
                       color=["#ff6b6b" if k == "HIGH" else "#00d4a1" for k in counts.index])
        ax.set_title("Records by sensitivity class", color="#e2e8f0", fontsize=11)
        ax.set_ylabel("Count", color="#94a3b8")
        for b in bars:
            ax.annotate(f"{int(b.get_height()):,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", color="#e2e8f0", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    hi_feat, lo_feat = HIGH_SENSITIVITY_FEATURES[0], LOW_SENSITIVITY_FEATURES[0]
    fig2, axes = plt.subplots(1, 2, figsize=(11, 3.2), facecolor="#0f0f1a")
    for ax, feat, color, tag in [(axes[0], hi_feat, "#ff6b6b", "HIGH"),
                                  (axes[1], lo_feat, "#00d4a1", "LOW")]:
        style_ax(ax)
        ax.hist(raw_df[feat], bins=30, color=color, alpha=0.85)
        ax.set_title(f"{tag}-sensitivity example: {feat}", color="#e2e8f0", fontsize=10)
        ax.set_xlabel("Normalized value [0, 1]", color="#94a3b8")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("`data/raw/footprint_raw.csv` not found — run `python data/data_generator.py` "
            "locally to see real samples here.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Context-aware LDP
# ════════════════════════════════════════════════════════════════════════════
step_header(3, "Context-Aware Local Differential Privacy", "step3", cell="Cell 5")
st.latex(r"x' = x + \text{Laplace}\left(0,\ \frac{\Delta f}{\varepsilon}\right)")

all_features = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES
eps_values  = [eps_high if f in HIGH_SENSITIVITY_FEATURES else eps_low for f in all_features]

st.markdown("**🔁 Per-feature privacy budget (ε) assignment**")
heat_df = pd.DataFrame({"Feature": all_features, "ε (epsilon)": eps_values}).set_index("Feature")
fig3, ax3 = dark_fig((12, 2.6))
sns.heatmap(heat_df.T, ax=ax3, cmap="RdYlGn", annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"label": "ε (higher = less noise)"}, xticklabels=all_features)
ax3.set_title("Per-Feature Privacy Budget (ε) Assignment", color="#e2e8f0", fontsize=12, pad=12)
ax3.tick_params(colors="#94a3b8", labelsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig3)
st.caption("Red = small ε, more noise (HIGH-sensitivity) · Green = large ε, less noise (LOW-sensitivity)")

st.markdown("**🔁 Noise applied to real records, before vs. after**")
if raw_df is not None:
    sample = raw_df.sample(n=min(500, len(raw_df)), random_state=42)
    hi_feat, lo_feat = HIGH_SENSITIVITY_FEATURES[0], LOW_SENSITIVITY_FEATURES[0]
    X_pair = sample[[hi_feat, lo_feat]].values.astype(np.float32)
    noisy_pair = privatize_vector(
        X_pair,
        sensitivities=np.array([GLOBAL_SENSITIVITY, GLOBAL_SENSITIVITY], dtype=np.float32),
        epsilons=np.array([eps_high, eps_low], dtype=np.float32),
    )
    fig4, axes = plt.subplots(1, 2, figsize=(11, 3.2), facecolor="#0f0f1a")
    for ax, idx, feat, eps, tag in [(axes[0], 0, hi_feat, eps_high, "HIGH"),
                                     (axes[1], 1, lo_feat, eps_low, "LOW")]:
        style_ax(ax)
        ax.hist(X_pair[:, idx], bins=25, color="#94a3b8", alpha=0.5, label="Original")
        ax.hist(noisy_pair[:, idx], bins=25, color="#7c3aed", alpha=0.6, label="Privatized")
        ax.set_title(f"{tag}: {feat} (ε={eps:g})", color="#e2e8f0", fontsize=10)
        ax.legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig4)
    mse_hi = float(np.mean((X_pair[:, 0] - noisy_pair[:, 0]) ** 2))
    mse_lo = float(np.mean((X_pair[:, 1] - noisy_pair[:, 1]) ** 2))
    st.caption(f"Mean squared distortion on this sample — **{hi_feat}**: {mse_hi:.4f}  ·  "
               f"**{lo_feat}**: {mse_lo:.4f}. Smaller ε injects visibly more noise.")
else:
    st.info("`data/raw/footprint_raw.csv` not found — run `python data/data_generator.py` locally to see the live noise demo.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Federated Learning setup
# ════════════════════════════════════════════════════════════════════════════
step_header(4, "Federated Learning Setup", "step4", cell="Cell 6")
col_a, col_b = st.columns(2)
with col_a:
    st.caption("Gradient clipping (DP-SGD step)")
    st.latex(r"\hat g = g \,/\, \max\!\left(1,\ \lVert g \rVert_2 / C\right)")
with col_b:
    st.caption("FedAvg aggregation")
    st.latex(r"w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N}\, w_k")

if client_dfs:
    comp_rows = []
    for i, df in sorted(client_dfs.items()):
        vc = df["sensitivity_class"].value_counts()
        comp_rows.append({"Client": f"client_{i}", "HIGH": int(vc.get("HIGH", 0)),
                           "LOW": int(vc.get("LOW", 0))})
    comp_df = pd.DataFrame(comp_rows).set_index("Client")
    fig5, ax5 = dark_fig((11, 3.2))
    ax5.bar(comp_df.index, comp_df["HIGH"], color="#ff6b6b", label="HIGH-sensitivity records")
    ax5.bar(comp_df.index, comp_df["LOW"], bottom=comp_df["HIGH"], color="#00d4a1", label="LOW-sensitivity records")
    ax5.set_title("Per-Client Shard Composition (non-IID by design)", color="#e2e8f0", fontsize=12)
    ax5.set_ylabel("Records", color="#94a3b8")
    ax5.legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="#e2e8f0")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig5)
else:
    st.info("`data/processed/client_*.csv` not found — run `python data/data_generator.py` locally to generate per-client shards.")

try:
    from fl.model import get_model, count_parameters
    n_params = count_parameters(get_model())
    st.caption(f"Model: `Input({len(ALL_FEATURES)}) → 64 → 32 → 16 → 2`  ·  "
               f"**{n_params:,}** trainable parameters per client.")
except Exception:
    st.caption("Model: `Input(20) → 64 → 32 → 16 → 2`  (install torch to show a live parameter count)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Three experimental conditions
# ════════════════════════════════════════════════════════════════════════════
step_header(5, "Three Experimental Conditions", "step5", cell="Cells 8-10")

cols = st.columns(3)
experiments = [("Baseline FL", baseline_hist), ("Static LDP-FL", static_hist), ("Hybrid LDP-FL", hybrid_hist)]

for col, (label, hist) in zip(cols, experiments):
    last = hist[-1]
    acc, reid, t = last.get("accuracy", 0.0), last.get("re_id_rate", 0.0), last.get("avg_time_sec", 0.0)
    cls = EXP_CSS_CLASS[label]
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{EXP_EMOJI[label]} {label}</div>
            <div class="metric-value {cls}">{acc*100:.1f}%</div>
            <div class="metric-label">Accuracy</div>
            <hr style="opacity:0.2">
            <div style="display:flex; justify-content:space-around; margin-top:0.5rem">
                <div>
                    <div style="font-size:1.1rem; font-weight:600">{reid*100:.1f}%</div>
                    <div class="metric-label">Re-ID Rate ↓</div>
                </div>
                <div>
                    <div style="font-size:1.1rem; font-weight:600">{t:.2f}s</div>
                    <div class="metric-label">Avg Round Time</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("**Training curves — how each condition converges over FL rounds**")
fig6, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor="#0f0f1a")
for ax in axes:
    style_ax(ax)
for label, hist in experiments:
    rds = [h["round"] for h in hist]
    axes[0].plot(rds, [h.get("accuracy", 0) for h in hist], label=label,
                 color=EXP_COLORS[label], linewidth=2.5, marker="o", markersize=3)
    axes[1].plot(rds, [h.get("re_id_rate", 0) for h in hist], label=label,
                 color=EXP_COLORS[label], linewidth=2.5, marker="o", markersize=3)
axes[0].set_title("Accuracy per Round", color="#e2e8f0", fontsize=13)
axes[0].set_xlabel("Round", color="#94a3b8"); axes[0].set_ylabel("Accuracy", color="#94a3b8")
axes[0].legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="#e2e8f0")
axes[1].set_title("Re-Identification Rate per Round (↓ is better)", color="#e2e8f0", fontsize=13)
axes[1].set_xlabel("Round", color="#94a3b8"); axes[1].set_ylabel("Re-ID Rate", color="#94a3b8")
axes[1].legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="#e2e8f0")
plt.tight_layout()
st.pyplot(fig6)


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Privacy attack
# ════════════════════════════════════════════════════════════════════════════
step_header(6, "Privacy Attack — Measuring Re-Identification Risk", "step6", cell="Cells 6-7")
st.caption("Re-ID Rate = nearest-neighbor linkage attack (experiments/evaluate.py). "
           "The image below is a separate, supplementary demo — not the Chapter 4 metric.")

gi_path = "plots/gradient_inversion_demo.png"
if os.path.exists(gi_path):
    st.image(gi_path, caption="Supplementary DLG reconstruction-error demo (illustrative only).", width=520)


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — ε calibration
# ════════════════════════════════════════════════════════════════════════════
step_header(7, "Privacy Budget Calibration", "step7", cell="Cell 11")

if calibration_data:
    sweep = calibration_data["sweep"]
    sweep_eps  = [r["epsilon_high"] for r in sweep]
    sweep_acc  = [r["accuracy"]     for r in sweep]
    sweep_reid = [r["re_id_rate"]   for r in sweep]
    threshold  = calibration_data.get("re_id_risk_threshold")
    optimal    = calibration_data["selected_optimal"]

    fig7, axes = plt.subplots(2, 1, figsize=(10, 6.5), facecolor="#0f0f1a", sharex=True)
    style_ax(axes[0]); style_ax(axes[1])
    axes[0].plot(sweep_eps, sweep_acc, "o-", color="#00d4a1", linewidth=2.5)
    axes[0].axvline(optimal["epsilon_high"], color="#a78bfa", linestyle="--", alpha=0.8,
                     label=f"Selected ε_high = {optimal['epsilon_high']}")
    axes[0].set_ylabel("Accuracy", color="#94a3b8")
    axes[0].set_title("Accuracy vs ε_high", color="#e2e8f0", fontsize=12)
    axes[0].legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="#e2e8f0")

    axes[1].plot(sweep_eps, sweep_reid, "s-", color="#ff6b6b", linewidth=2.5)
    axes[1].axvline(optimal["epsilon_high"], color="#a78bfa", linestyle="--", alpha=0.8)
    if threshold is not None:
        axes[1].axhline(threshold, color="#f5a623", linestyle=":", alpha=0.8,
                         label=f"Acceptable-risk threshold = {threshold*100:.0f}%")
        axes[1].legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[1].set_ylabel("Re-ID Rate", color="#94a3b8")
    axes[1].set_xlabel("ε_high (log scale, ε_low fixed)", color="#94a3b8")
    axes[1].set_xscale("log")
    # Plain numeric tick labels — the default log-scale formatter renders ticks
    # via matplotlib's mathtext parser, which crashes with some pyparsing builds.
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_title("Re-ID Rate vs ε_high", color="#e2e8f0", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig7)

    st.caption(f"Real data from `results/logs/epsilon_calibration_results.json`. Selected "
               f"ε_high={optimal['epsilon_high']} → accuracy {optimal['accuracy']*100:.2f}%, "
               f"re-ID rate {optimal['re_id_rate']*100:.2f}%.")
else:
    st.warning("⚠️  No calibration sweep results found. Run `python experiments/calibrate_epsilon.py` "
               "locally to generate `results/logs/epsilon_calibration_results.json` and see the real curve here.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — Final results
# ════════════════════════════════════════════════════════════════════════════
step_header(8, "Chapter 4 — Final Results & Takeaways", "step8", cell="Cells 11 & 13")

finals = {
    "Baseline FL":   baseline_data["final"] if baseline_data else {"accuracy": baseline_hist[-1]["accuracy"], "f1_score": None, "re_id_rate": baseline_hist[-1]["re_id_rate"]},
    "Static LDP-FL": static_data["final"]   if static_data   else {"accuracy": static_hist[-1]["accuracy"],   "f1_score": None, "re_id_rate": static_hist[-1]["re_id_rate"]},
    "Hybrid LDP-FL": hybrid_data["final"]   if hybrid_data   else {"accuracy": hybrid_hist[-1]["accuracy"],   "f1_score": None, "re_id_rate": hybrid_hist[-1]["re_id_rate"]},
}

metrics = ["accuracy", "f1_score", "re_id_rate"]
metric_labels = ["Accuracy", "F1 Score", "Re-ID Rate (↓)"]
x = np.arange(len(metrics))
width = 0.25

fig8, ax8 = dark_fig((10, 4.5))
for i, (label, _) in enumerate(experiments):
    vals = [finals[label].get(m) or 0 for m in metrics]
    bars = ax8.bar(x + (i - 1) * width, vals, width, label=label, color=EXP_COLORS[label])
    for b, v in zip(bars, vals):
        ax8.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", color="#e2e8f0", fontsize=8)
ax8.set_xticks(x); ax8.set_xticklabels(metric_labels, color="#94a3b8")
ax8.set_title("Chapter 4: Final Performance Comparison Across All Methods", color="#e2e8f0", fontsize=13)
ax8.legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="#e2e8f0")
plt.tight_layout()
st.pyplot(fig8)
