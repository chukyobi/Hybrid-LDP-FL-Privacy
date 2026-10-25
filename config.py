"""
config.py
=========
Global hyperparameters and privacy budget definitions for the
Hybrid LDP-FL Algorithm (Context-Aware).

Thesis: "A Hybrid Local Differential Privacy and Federated Learning
         Algorithm for Privacy-Preserving Digital Footprint Protection"
"""

# ─────────────────────────────────────────────
# PRIVACY BUDGET (ε — epsilon)
# Lower ε  → stronger privacy (more noise)
# Higher ε → weaker privacy  (less noise)
# ─────────────────────────────────────────────
EPSILON_HIGH = 0.5      # High-sensitivity data (inferred/passive footprints)
EPSILON_LOW  = 10.0     # Low-sensitivity data  (active/explicit footprints)
DELTA        = 1e-5     # Approximate DP failure probability

# ─────────────────────────────────────────────
# NOISE MECHANISM
# ─────────────────────────────────────────────
GLOBAL_SENSITIVITY = 1.0   # Δf — sensitivity of the query function

# ─────────────────────────────────────────────
# FEDERATED LEARNING SETTINGS
# ─────────────────────────────────────────────
NUM_CLIENTS      = 10       # Simulated federated clients (devices)
CLIENT_FRACTION  = 0.8      # Fraction of clients selected per round
NUM_ROUNDS       = 20       # Global aggregation rounds
LOCAL_EPOCHS     = 3        # Local training epochs per client per round
BATCH_SIZE       = 32       # Local mini-batch size
LEARNING_RATE    = 0.01     # Local SGD learning rate

# ─────────────────────────────────────────────
# GRADIENT CLIPPING
# ─────────────────────────────────────────────
MAX_GRAD_NORM = 1.0         # C — max L2 norm for gradient clipping

# ─────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────
NUM_SAMPLES     = 5000      # Total synthetic data points
NUM_FEATURES    = 20        # Feature vector dimension
NUM_CLASSES     = 2         # Binary classification (privacy risk: low/high)
TEST_SPLIT      = 0.2       # Fraction held out for test evaluation
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
# FEATURE SENSITIVITY MAPPING
# Which features are considered High Sensitivity
# ─────────────────────────────────────────────
HIGH_SENSITIVITY_FEATURES = [
    "location_trace",
    "health_interest_inferred",
    "political_affiliation_inferred",
    "financial_behavior",
    "biometric_pattern",
    "relationship_status_inferred",
    "night_activity_pattern",
    "communication_frequency",
]

LOW_SENSITIVITY_FEATURES = [
    "search_query_count",
    "app_usage_duration",
    "page_visit_count",
    "video_watch_time",
    "like_frequency",
    "comment_frequency",
    "share_frequency",
    "post_frequency",
    "session_duration",
    "login_frequency",
    "device_type",
    "browser_type",
]

# ─────────────────────────────────────────────
# EXPERIMENT OUTPUT
# ─────────────────────────────────────────────
RESULTS_DIR = "results/logs"
PLOTS_DIR   = "results/plots"
