# 🔒 Hybrid LDP-FL: Privacy-Preserving Digital Footprint Protection

> **Thesis**: A Context-Aware Local Differential Privacy and Federated Learning Algorithm for Privacy-Preserving Digital Footprint Protection

---

## 📋 Overview

This repository implements the **Hybrid LDP-FL Algorithm** — a novel combination of:
- **Local Differential Privacy (LDP)**: Injects calibrated Laplace noise at the data level (on-device), *before* any model training
- **Federated Learning (FL)**: Ensures raw data never leaves the user device; only model gradients are shared
- **Context-Awareness** *(the innovation)*: Dynamically applies stronger noise (smaller ε) to HIGH-sensitivity features and lighter noise to LOW-sensitivity features, preserving utility without sacrificing privacy where it counts

**Core Formula:**
```
x' = x + Laplace(0, Δf / ε_context)

where ε_context = ε_high  if feature ∈ HIGH_SENSITIVITY_FEATURES
                  ε_low   otherwise
```

---

## 🗂️ Project Structure

```
Hybrid-LDP-FL-Privacy/
├── config.py                    # All hyperparameters & privacy budgets
├── requirements.txt
│
├── data/
│   ├── data_generator.py        # Synthetic footprint dataset + sensitivity labeler
│   ├── raw/                     # Generated raw CSV
│   └── processed/               # Train, test, per-client shards
│
├── ldp/
│   ├── laplace_mechanism.py     # x' = x + Laplace(0, Δf/ε)
│   └── context_aware_ldp.py    # Routes features to ε_high or ε_low
│
├── fl/
│   ├── model.py                 # MLP classifier (lightweight for edge devices)
│   ├── client.py                # Local training + gradient clipping
│   ├── server.py                # FedAvg aggregation
│   └── gradient_clipping.py    # DP-SGD gradient norm bounding
│
├── attacks/
│   └── gradient_inversion.py   # DLG attack demo (supplementary — NOT the Chapter 4 re-id metric)
│
├── experiments/
│   ├── run_baseline_fl.py       # Standard FL (no privacy)
│   ├── run_static_ldp_fl.py     # FL + uniform static ε LDP
│   ├── run_hybrid_ldp_fl.py     # Hybrid LDP-FL — the proposed algorithm ⭐
│   ├── calibrate_epsilon.py     # ε_high calibration sweep (Section 4.7)
│   └── evaluate.py              # Accuracy, F1, re-id rate computation (nearest-neighbour linkage attack)
│
├── results/
│   ├── logs/                    # JSON experiment results
│   └── plots/                   # Charts for thesis figures
│
├── Hybrid_LDP_FL_Colab.ipynb    # Self-contained Colab/Jupyter companion notebook
└── dashboard/
    └── app.py                   # Streamlit results dashboard
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
cd Hybrid-LDP-FL-Privacy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/data_generator.py
```

### 3. Run Experiments (for Chapter 4)
```bash
# Baseline FL — no privacy
python experiments/run_baseline_fl.py

# Static LDP-FL — uniform ε
python experiments/run_static_ldp_fl.py

# Hybrid LDP-FL — Context-Aware, the proposed algorithm ⭐
python experiments/run_hybrid_ldp_fl.py

# Privacy budget calibration sweep (Section 4.7)
python experiments/calibrate_epsilon.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Chapter 4 Metrics

| Metric | Baseline FL | Static LDP-FL | Hybrid LDP-FL ⭐ |
|--------|------------|---------------|----------------|
| **Accuracy** | 88.00% (highest) | 64.20% (lowest) | 79.50% (recovers most of the accuracy Static LDP-FL sacrifices) |
| **Re-ID Rate** | 100.00% (max risk) | 0.70% (lowest risk) | 21.30% (moderate risk — an order of magnitude below Baseline, in exchange for far better utility than Static) |
| **Privacy Guarantee** | ❌ None | ✅ Uniform ε | ✅✅ Context-Aware, calibrated (Section 4.7) |

Static LDP-FL achieves the lowest re-identification risk of the three, but at
the largest utility cost. Hybrid LDP-FL does not beat Static on privacy alone —
it is the calibrated middle ground that keeps re-identification risk far below
Baseline while recovering most of the accuracy Static LDP-FL sacrifices. See
`results/logs/*.json` for the exact saved figures.

---

## 🔑 Key Parameters (`config.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `EPSILON_HIGH` | 0.5 | Strong privacy for high-sensitivity/hybrid-inference footprints (calibrated, Section 4.7) |
| `EPSILON_LOW` | 10.0 | Light privacy for low-sensitivity/active footprints |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping bound C |
| `NUM_CLIENTS` | 10 | Simulated federated devices |
| `NUM_ROUNDS` | 20 | Global FL communication rounds |

---

## 📐 Algorithm Pipeline

```
User Device:
  Raw Data → [Context-Aware LDP] → x' (noisy data)
           → [Local Training on x'] → gradients
           → [Gradient Clipping ‖g‖ ≤ C]
           → Send clipped gradients to server

Central Server:
  Collect gradients from K clients
  → [FedAvg: w = Σ(nk/N)·wk]
  → Broadcast updated global model
```

---

## 📄 License
MIT
