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
│   └── gradient_inversion.py   # DLG attack simulation (re-id metric)
│
├── experiments/
│   ├── run_baseline_fl.py       # Standard FL (no privacy)
│   ├── run_static_ldp_fl.py     # FL + uniform static ε LDP
│   ├── run_hybrid_ldp_fl.py     # YOUR ALGORITHM ⭐
│   └── evaluate.py              # Accuracy, F1, re-id rate computation
│
├── results/
│   ├── logs/                    # JSON experiment results
│   └── plots/                   # Charts for thesis figures
│
├── notebooks/                   # Jupyter exploration notebooks
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

# YOUR ALGORITHM — Context-Aware Hybrid LDP-FL ⭐
python experiments/run_hybrid_ldp_fl.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Chapter 4 Metrics

| Metric | Baseline FL | Static LDP-FL | Hybrid LDP-FL ⭐ |
|--------|------------|---------------|----------------|
| **Accuracy** | Highest | Lowest | Better than Static |
| **Re-ID Rate** | Highest risk | Low risk | Lowest risk |
| **Training Speed** | Fastest | Moderate | Moderate |
| **Privacy Guarantee** | ❌ None | ✅ Uniform | ✅✅ Context-Aware |

---

## 🔑 Key Parameters (`config.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `EPSILON_HIGH` | 0.1 | Strong privacy for inferred/passive footprints |
| `EPSILON_LOW` | 1.0 | Lighter privacy for explicit/active footprints |
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
