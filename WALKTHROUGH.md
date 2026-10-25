# Hybrid LDP-FL — Plain-English Walkthrough

This document explains the entire project step by step, for a complete
beginner — lots of analogies, lots of made-up numbers worked out by hand, no
assumed background in privacy or machine learning. Read it top to bottom
once, and you should be able to explain every file in this repo and every
number in your results to a panel.

**This walkthrough is numbered to match `Hybrid_LDP_FL_Colab.ipynb` cell for
cell.** Step 5 here is Cell 5 in the notebook, Step 11 here is Cell 11, and
so on — open the notebook side by side if you want to see the code behind
each step. The [Streamlit dashboard](dashboard/app.py)
(`streamlit run dashboard/app.py`) is the same 13 steps again, as a live,
clickable app with real charts instead of words. All three — notebook, this
document, and the dashboard — tell the exact same story.

---

## 1. The problem, in one paragraph (with an analogy)

Every time someone uses an app, they leave a "digital footprint" — search
history, location, how long they watch videos, who they message. Companies
want to train machine-learning models on this data, but the normal way of
doing it means collecting everyone's raw data onto one server — which is a
privacy disaster waiting to happen.

Imagine a teacher wants to know the class average on a pop quiz, but no
student wants to reveal their own score. One fix: each student doesn't
hand in their real score — they add a small random fudge number to it
first (sometimes +3, sometimes -5, whatever), then hand in the fudged
number. The class average still comes out roughly right (the random fudges
cancel out over many students), but no one can point at a single handed-in
number and know anyone's real score. That fudge-number trick is, in a
nutshell, **differential privacy** — and this project applies it to digital
footprints, combined with a second trick (never actually collecting the
raw papers at all — see Step 6) and an original idea on top:

- **Local Differential Privacy (LDP)** — the "fudge number" trick above,
  applied before any training happens.
- **Federated Learning (FL)** — don't even collect the papers centrally;
  have each student grade their own practice answer sheet at home, and only
  mail in a summary of what they learned, never the sheet itself.
- **Context-awareness (this thesis's original idea)** — not all quiz
  questions are equally sensitive. A question like "what's your favorite
  color" barely needs fudging. A question like "what's your home address"
  needs heavy fudging. So instead of fudging every answer by the same
  amount, fudge *sensitive* answers heavily and *harmless* answers lightly.
  You keep the class average useful while still protecting what matters.

---

## 2. Glossary — the vocabulary you need

| Term | Plain-English meaning |
|---|---|
| **ε (epsilon)** | The "privacy budget," like a volume knob for noise. *Small* ε = a lot of fudging (strong privacy, blurrier data). *Large* ε = barely any fudging (weak privacy, clearer data). |
| **Laplace noise** | The specific *kind* of random fudge number used. Picture a dartboard where darts cluster tightly around the bullseye (0) but every so often land way off to one side — most nudges are small, occasional nudges are big. |
| **Sensitivity (Δf)** | How much one person's answer could possibly change the result, worst case. Used to calibrate exactly how big the fudge numbers need to be. |
| **Federated Learning** | Training happens on each person's own device; only the *lessons learned* (not the raw homework) get sent back. |
| **FedAvg** | The rule for combining everyone's lessons-learned into one: a weighted average, giving more say to whoever did more homework. |
| **Gradient** | During training, the model keeps asking "which direction should my internal numbers nudge to make fewer mistakes?" That direction is the gradient. |
| **Gradient clipping** | A volume cap: if one device's "nudge" is unusually loud (because it saw one dramatic data point), turn it down to a fixed maximum before it's sent in. Keeps any one device from dominating — or leaking too much. |
| **Re-identification attack** | The privacy "final exam": an attacker (played by us, on purpose) tries to match a fudged answer sheet back to the student who wrote it. |
| **Non-IID data** | Different devices don't hold the same *mix* of data. We deliberately simulate this — some simulated devices hold mostly sensitive records, others mostly harmless ones — because real device populations are lumpy like that, not evenly shuffled. |

---

## 3. The walkthrough, cell by cell

### Step 1 — Installing the tools *(Notebook Cell 1)*

Nothing conceptual here — just `pip install torch pandas scikit-learn ...`.
Locally, this is `pip install -r requirements.txt` once, up front.

### Step 2 — The settings dial, all in one place *(Notebook Cell 2, = `config.py`)*

Every number every other step depends on lives in one place, so nothing gets
hard-coded twice and quietly drifts out of sync. The ones that matter most:

- `EPSILON_HIGH = 0.5` — fudge-amount dial for sensitive features (small number → heavy fudging)
- `EPSILON_LOW = 10.0` — fudge-amount dial for harmless features (big number → light fudging)
- `MAX_GRAD_NORM = 1.0` — the gradient "volume cap," `C`
- `NUM_CLIENTS = 10` — how many pretend devices we simulate
- `NUM_ROUNDS = 20` — how many times the devices and server go back and forth
- `HIGH_SENSITIVITY_FEATURES` / `LOW_SENSITIVITY_FEATURES` — the two lists deciding which dial (ε) each feature gets

A fixed `SEED = 42` is also set here, for a boring but important reason: it
makes "random" choices (which noise gets drawn, which clients go first)
reproducible, so re-running the code gives comparable results instead of a
totally different dice roll every time.

### Step 3 — (Optional) saving to Google Drive *(Notebook Cell 3)*

Only relevant inside Google Colab — lets you save results to your Drive
instead of Colab's temporary disk, which gets wiped when the session ends.
Running locally, this step does nothing (results just save to `results/`
on your own machine).

### Step 4 — Inventing 5,000 pretend users *(Notebook Cell 4, = `data/data_generator.py`)*

There's no real user data in this thesis — that would need ethics approval
and real victims. Instead, this step **invents** 5,000 fake "digital
footprints":

- **8 HIGH-sensitivity features** — `location_trace`, `biometric_pattern`,
  `political_affiliation_inferred`, etc. Randomly generated with *wide*
  variance (people differ a lot on these).
- **12 LOW-sensitivity features** — `login_frequency`, `device_type`, etc.
  Randomly generated too, every value squeezed into [0, 1] so every feature
  plays by the same numeric rules.

**Worked example — meet "User #12":** imagine one invented row has
`location_trace = 0.70` and `login_frequency = 0.60`. Both numbers mean
nothing on their own (they're normalized placeholders, not real GPS
coordinates), but — and this is the whole point — the *first* one is
treated as highly revealing about that person, and the second one isn't.
We'll fudge them very differently in Step 5.

Each invented record is also tagged `sensitivity_class = HIGH` or `LOW`
(HIGH if any of its 8 sensitive features is unusually large, >0.7). This tag
doesn't feed the model at all — it's only used in Step 6 to decide which of
the 10 pretend devices each record lands on, so some devices end up holding
mostly sensitive records and others mostly harmless ones (the non-IID idea
from the glossary).

### Step 5 — The noise machine *(Notebook Cell 5, = `ldp/laplace_mechanism.py` + `ldp/context_aware_ldp.py`)*

This is the privacy trick itself, and the thesis's whole original
contribution. The formula:

```
x' = x + Laplace(0, Δf / ε)
```

In plain words: take the true value `x`, and add a random fudge number
drawn from a Laplace distribution. How *big* that fudge tends to be is
controlled by `Δf / ε` — sensitivity divided by epsilon. Small ε → bigger
typical fudge → more blur → harder to guess the original.

**Worked example, with real numbers from this actual code** (seed fixed so
it's reproducible — your own run will draw different random fudges, but the
*pattern* below always holds):

| Feature | True value `x` | ε used | Fudge drawn | Privatized `x'` |
|---|---|---|---|---|
| `location_trace` (HIGH) | 0.700 | 0.5 (heavy fudging) | −0.363 | **0.337** |
| `login_frequency` (LOW) | 0.600 | 10.0 (light fudging) | +0.058 | **0.658** |

Same mechanism, same formula, applied to two features from the *same*
invented user — but because `location_trace` is routed through the small
ε = 0.5 dial, its fudge is six times bigger than `login_frequency`'s, which
goes through the large ε = 10.0 dial. An attacker looking only at the
privatized numbers can no longer be confident the real `location_trace` was
0.70 — it could plausibly have started anywhere in a wide range around
0.337. But `login_frequency`'s privatized value, 0.658, is barely
distinguishable from its true 0.600 — which is fine, because nobody's
login frequency was worth protecting that hard in the first place.

This routing — *same mechanism, different ε per feature, chosen by how
sensitive that feature actually is* — is `apply_context_aware_ldp()`, the
one function this entire thesis is arguing for. A second function,
`apply_static_ldp()`, exists purely as the "naive" comparison: it applies
*one* ε to every feature, ignoring sensitivity labels entirely. That's what
"Static LDP-FL" means later on.

### Step 6 — The federated-learning machinery *(Notebook Cell 6, = `fl/model.py`, `fl/client.py`, `fl/gradient_clipping.py`, `fl/server.py`)*

**The classroom analogy:** picture 10 students (the simulated devices),
each with their own stack of practice problems (their private data shard).
Each round:

1. The teacher (server) hands every student the same answer key draft (the
   current global model).
2. Each student studies at home with *their own* practice problems — which
   have already been fudged per Step 5 before the student even looks at
   them — and privately updates their own notes on how to improve (that's
   training a gradient).
3. Before mailing their notes back, each student is told "keep your
   correction notes under a certain length" — **gradient clipping**: if a
   student's proposed correction is unusually large (because they saw one
   weird practice problem), it gets shrunk down to a fixed maximum size,
   `‖g‖₂ ≤ C`, before being sent. No single student's opinion can swing the
   class answer key too hard.
4. The teacher collects everyone's correction notes and blends them into
   one updated answer key — a weighted average, **FedAvg**, giving more
   weight to students who worked through more problems.

**Worked example of FedAvg**, with made-up tiny numbers: suppose only 3
students mail in corrections — student A worked 100 problems and suggests
nudging the answer key by `+0.8`; student B worked 300 problems and
suggests `+0.2`; student C worked 100 problems and suggests `-0.4`. The
teacher doesn't average `0.8, 0.2, -0.4` evenly (that would give `+0.2`) —
each student's vote is weighted by how many problems they worked, out of
500 total:

```
new answer key nudge = (100/500)×0.8 + (300/500)×0.2 + (100/500)×(-0.4)
                      = 0.16 + 0.12 - 0.08
                      = 0.20
```

Student B, who did the most homework, pulled the result closer to their own
suggestion. That's exactly what `fl/server.py`'s `fedavg()` function
computes, just with real model weights instead of one number.

The actual "homework" is a small neural network
(`Input(20) → 64 → 32 → 16 → 2`) — deliberately tiny because it's meant to
simulate running *on a phone*, not a data-center GPU. It predicts one of two
labels: "low privacy risk" or "high privacy risk."

> **Why LayerNorm and not the more common BatchNorm?** BatchNorm needs to
> track a running average *across* a batch of examples, which gets flaky
> when every student's homework pile looks completely different (non-IID).
> LayerNorm normalizes *within* one student's own answer sheet instead, so
> it aggregates cleanly across very different clients.

### Step 7 — Supplementary: a document-shredder demo *(Notebook Cell 7, = `attacks/gradient_inversion.py`)*

> ⚠️ Illustrative only — not used to produce any of the real Chapter 4
> numbers (those come from Step 6's evaluator, described in Step 9 below).

Imagine a shredder bin outside an office: you never see the actual
documents that went in, but if you're patient and clever, you can sometimes
guess a surprising amount about them just from how the shredded strips pile
up — their thickness, how many, what order they settled in. **Deep Leakage
from Gradients (DLG)** is the machine-learning version of that: even though
an attacker never sees a device's raw data, the *gradient* it sends back
(the "correction notes" from Step 6) can sometimes be reverse-engineered to
approximately reconstruct the original data that produced it.

This demo exists purely to make a point: **Federated Learning by itself is
not automatically private.** Not sharing the raw papers isn't the same as
nobody being able to infer anything — which is exactly the argument for why
Step 5's noise machine is necessary at all, not just a nice bonus.

### Steps 8-10 — The three experiments *(Notebook Cells 8, 9, 10)*

Same classroom, same answer key, same 20 rounds — the *only* thing that
changes between the three is what each student does to their practice
problems before studying:

| Step | Script | Privacy mode | What each student's practice problems go through |
|---|---|---|---|
| 8 | `experiments/run_baseline_fl.py` | `none` | Nothing — raw numbers, no fudging at all |
| 9 | `experiments/run_static_ldp_fl.py` | `static` | *Every* problem gets the *same* fudge dial, sensitive or not |
| 10 | `experiments/run_hybrid_ldp_fl.py` | `hybrid` | Step 5's context-aware routing — **the proposed algorithm** ⭐ |

Each script saves its results to `results/logs/*_results.json`.

**How "did privacy actually work?" is measured** (this is what
`experiments/evaluate.py` does, also used live after every round): picture
a detective's lineup. The detective (attacker) is handed one fudged answer
sheet and the *entire stack* of everyone's real, unfudged sheets, and has to
pick which real sheet it came from. Their strategy is simple: pick whichever
real sheet looks *closest* to the fudged one (nearest neighbor by straight-
line distance across all 20 features). If they guess right, that student
has been **re-identified**. The **re-identification rate** is just: (how
many sheets the detective correctly matched) ÷ (total sheets). Rate near
100% → the fudging did nothing, the detective aces the lineup every time.
Rate near 0% → the fudging worked, the detective is basically guessing.

### Step 11 — Charts for the thesis *(Notebook Cell 11)*

Five visualizations get built here, each tied to a specific thesis table or
figure:

1. **Comparison table** (→ Table 4.3) — accuracy / F1 / re-ID side by side for all three experiments.
2. **Training curves** (→ Figures 4.3 & 4.4) — how accuracy and re-ID rate evolve round by round.
3. **ε_high calibration sweep** (→ Table 4.4 / Figure 4.5) — the tradeoff curve justifying `EPSILON_HIGH = 0.5` (full version: `experiments/calibrate_epsilon.py`, Section 4.7).
4. **Final metric bar chart** — the headline "Baseline vs Static vs Hybrid" summary.
5. **Privacy budget heatmap** — which ε each of the 20 features actually received, visually.

The calibration sweep (#3) answers "why ε_high = 0.5 and not some other
number?" the same way Step 5 worked out one example by hand, but at scale:
try a grid of candidates (0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0), run
the full hybrid experiment at each one, and pick whichever gives the
**highest accuracy** among candidates whose re-identification rate stays
**at or below a 25% acceptable-risk ceiling**. 0.5 wins that search — it's
not a guess.

**The salt analogy:** salt is a decent stand-in for noise here, because
salt is also a preservative. Too little, and the dish spoils (too little
noise → weak privacy → the Step 9 detective re-identifies almost everyone).
Too much, and the dish is inedible (too much noise → the model can't learn
anything useful, accuracy collapses). This sweep tastes 8 different pinches
and picks the *least* salt that still keeps the dish from spoiling — the
largest ε (i.e. least noise) among the candidates that still keep
re-identification risk under the 25% ceiling, so accuracy isn't sacrificed
any more than it has to be.

### Step 12 — Saving everything *(Notebook Cell 12)*

Writes one JSON file (`results/chapter4_results.json` locally, or to Google
Drive in Colab) containing every experiment's final numbers, full
round-by-round history, the calibration sweep, and the config used to
produce them — plus (in Colab only) zips up the `plots/` folder for
download.

### Step 13 — The auto-written summary *(Notebook Cell 13)*

The very last cell writes out a few ready-to-quote sentences, filled in with
*this run's* actual numbers — something like:

> "The Hybrid LDP-FL algorithm achieved **82.1%** accuracy versus **56.0%**
> for Static LDP-FL at the same ε_high=0.5, a **26.1** percentage-point
> improvement, while keeping a re-identification rate of **21.3%** —
> demonstrating that context-aware noise injection preserves model utility
> without sacrificing privacy on sensitive features."

(The exact percentages above are illustrative — yours will differ slightly
run to run; see `results/logs/*.json` for your real saved numbers.) This
cell exists so you always have an accurate, current one-paragraph summary
ready to read aloud, instead of having to recompute percentage differences
live in front of a panel.

---

## 4. What the final numbers mean

| | Baseline FL | Static LDP-FL | Hybrid LDP-FL ⭐ |
|---|---|---|---|
| Accuracy | highest (~87-88%) | lowest (~55-65%) | in between, closer to Baseline (~78-82%) |
| Re-ID Rate | ~100% (privacy totally failed) | lowest (~0.5-1%) | an order of magnitude below Baseline (~20%) |

- **Baseline proves the problem is real.** With zero fudging, the detective
  from Step 9 aces nearly every lineup. This is the "why does this thesis
  need to exist" evidence.
- **Static proves naive privacy is expensive.** Fudging *every* feature
  equally does crush re-identification risk to nearly zero — but it also
  crushes accuracy, because harmless features (which never needed
  protecting) got buried in unnecessary noise too.
- **Hybrid is the thesis's actual argument.** It doesn't beat Static on raw
  privacy — that was never the claim. By spending the fudge budget only
  where sensitivity actually warrants it, it recovers most of the accuracy
  Static sacrifices, while keeping re-identification risk far below the
  unprotected Baseline. A calibrated middle ground, not a free lunch.

---

## 5. Likely panel questions and short answers

**"Why Laplace noise specifically, and not Gaussian?"**
Laplace noise gives exact (ε, 0)-differential privacy for a given
sensitivity and ε — the textbook-standard mechanism for this exact setup.
Gaussian noise is typically used for the *relaxed* (ε, δ)-DP guarantee
instead, which needs an extra parameter and is more common when combining
many noisy releases (e.g. deep-learning DP-SGD at scale). This project uses
the simpler, exact Laplace mechanism because it matches the per-feature,
per-release privacy claim being made here.

**"What does ε = 0.5 actually guarantee, mathematically?"**
It bounds how much more or less likely any given output is, depending on
whether one specific person's data was included or excluded — formally,
the likelihood ratio between "neighboring" datasets (differing in one
record) is bounded by `e^ε`. Smaller ε → tighter bound → stronger
guarantee.

**"Isn't 21% re-identification still a lot?"**
Compared to Baseline's ~100%, it's roughly a 5x reduction, achieved while
keeping most of the model's usefulness. The claim is about the *tradeoff*,
not reaching zero risk — zero risk is achievable (see Static), just not
usefully, since it also destroys accuracy.

**"Why Federated Learning at all — why not just add noise and train
centrally?"**
Two separate protections, not one: LDP protects the *values* (even leaked
data is fudged), FL protects the *collection itself* (raw data never has to
leave the device, and gradient clipping further limits what any single
device's contribution can reveal). Step 7's shredder-bin demo exists
specifically to show that FL alone, without the noise machine, still leaks
information through gradients — neither technique alone is enough.

**"How was ε_high chosen — did you just pick 0.5?"**
No — see Step 11. It's the output of a calibration sweep over 8 candidate
values, selecting the one that maximizes accuracy subject to a 25%
re-identification risk ceiling. Reproducible with
`python experiments/calibrate_epsilon.py`.

**"Is the dataset real user data?"**
No — it's synthetic, invented in Step 4 for controlled experimentation (no
privacy/ethics review needed, and it lets us know the *true* unfudged
values, which is required to measure re-identification rate at all — you
can't grade the detective's lineup without an answer key).

---

## 6. Running it yourself, start to finish

```bash
# 1. Set up the environment (only needed once)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Step 4 — invent the synthetic dataset
python data/data_generator.py

# 3. Steps 8-10 — run all three experiments
python experiments/run_baseline_fl.py
python experiments/run_static_ldp_fl.py
python experiments/run_hybrid_ldp_fl.py

# 4. Step 11 (full version) — re-justify the choice of ε_high
python experiments/calibrate_epsilon.py

# 5. See everything, live, as a 13-step walkthrough with real charts
streamlit run dashboard/app.py
```

Every script prints what it's doing as it runs, and saves its results to
`results/logs/*.json` so the dashboard and this document stay in sync with
whatever you most recently ran.
