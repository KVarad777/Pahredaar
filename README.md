# Adversarial Payment Fraud Simulation — Red Team Module

Phase 1 of the project: **Identify** (LLM scenario proposer) + **Generate** (synthetic UPI/ISO-8583
transaction simulator). No Defend/reward code yet — that's Phase 2.

---

## 1. Datasets — where to get them (real data, used ONLY to fit distribution parameters)

You are **not** training a model directly on these. You download them once, fit statistical
parameters (amount distribution, timing, category frequency, fraud ratio), save those parameters
to `config/distribution_params.yaml`, and your own simulator uses that config to generate data.

### Option A — Kaggle API (recommended, works in Colab)

```bash
# In a Colab cell:
!pip install kaggle
from google.colab import files
files.upload()  # upload your kaggle.json (from kaggle.com/settings -> Create New API Token)

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# IEEE-CIS Fraud Detection (Vesta)
!kaggle competitions download -c ieee-fraud-detection -p data/raw_reference/ieee_cis
# NOTE: this is a "competition" dataset -> you must click "Join Competition" once on the
# Kaggle website (kaggle.com/c/ieee-fraud-detection) before the API download will work.

# PaySim (mobile money simulator, real African mobile-money log patterns)
!kaggle datasets download -d ntnu-testimon/paysim1 -p data/raw_reference/paysim

# ULB Credit Card Fraud (real, PCA-anonymized European transactions)
!kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw_reference/ulb

# unzip everything
!cd data/raw_reference/ieee_cis && unzip -o '*.zip'
!cd data/raw_reference/paysim && unzip -o '*.zip'
!cd data/raw_reference/ulb && unzip -o '*.zip'
```

### Option B — manual download (no Kaggle API)

| Dataset | URL | What to grab |
|---|---|---|
| IEEE-CIS Fraud Detection | https://www.kaggle.com/c/ieee-fraud-detection/data | `train_transaction.csv` (you must join the competition first — free, instant) |
| PaySim | https://www.kaggle.com/datasets/ntnu-testimon/paysim1 | `PS_20174392719_1491204439457_log.csv` |
| ULB Credit Card Fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `creditcard.csv` |

Drop the CSVs into `data/raw_reference/{ieee_cis,paysim,ulb}/` respectively.

### MITRE F3 taxonomy (not a CSV — a framework you read once)

Read: https://github.com/center-for-threat-informed-defense/fight-fraud-framework

`config/f3_taxonomy.json` in this repo already contains a starter taxonomy (12 techniques across
5 tactics) extracted from F3 so you don't have to scrape it yourself for the hackathon. Extend it
if you want deeper coverage — see the "Extending the taxonomy" note at the bottom of that file... (it's JSON, so that note lives in this README instead: just add more
`{tactic, technique, sub_technique, description}` objects following the same shape).

---

## 2. Run order

1. `notebooks/00_fit_distributions.ipynb` — loads the 3 CSVs, fits log-normal amount params,
   inter-transaction timing, MCC frequency, fraud ratio → writes `config/distribution_params.yaml`.
2. `notebooks/01_identify_and_generate.ipynb` — runs the Identify engine (LLM proposes scenarios),
   validates them, then runs the Generate engine to produce synthetic transactions, then checks
   fidelity (synthetic vs real distributions).

Everything under `src/` is plain importable Python — in Colab, either `!git clone` this repo or
mount Drive and `sys.path.append('/content/drive/MyDrive/fraud-redteam-project')`.

## 3. API key for the Identify engine's LLM calls

In Colab: click the key icon (🔑) in the left sidebar → "Secrets" → add `ANTHROPIC_API_KEY`.
Never hardcode it in a cell.

```python
from google.colab import userdata
import os
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

## 4. Directory map

```
config/                  distribution params, F3 taxonomy, UPI schema, null-injection rates
data/raw_reference/      downloaded Kaggle CSVs (gitignore these, they're large)
data/generated/round_XX/ your own synthetic output, versioned per round
src/identify/            LLM scenario proposer + validator + coverage matrix writer
src/generate/            legit traffic sim + per-category injectors + null injector + UPI formatter
notebooks/                Colab notebooks tying it together
tests/                   sanity checks — run these before trusting any generated data
```

---

# Part 2 — Blue Team (Defend Engine) + Full Closed Loop

Extends everything above with the feature pipeline, GBM/GNN/Sequence/Ensemble
Defend models, the Red-vs-Blue reward system, the miss-explanation feedback
engine, and a dashboard. No new datasets to download for this part — Blue
trains entirely on Red's synthetic output.

## New directory contents

```
src/features/          velocity_store.py, graph_state.py, behavioral_baseline.py,
                        feature_assembler.py — the bridge from raw txns to model input
src/defend/             gbm_model.py, gnn_model.py, sequence_model.py, ensemble.py, train.py
src/reward/             blue_reward.py, red_reward.py, loop_orchestrator.py (main entry point)
src/feedback/           miss_explainer.py — rule-based, plain-language miss explanations
dashboard/               build_dashboard.py — coverage matrix + detection-rate-over-rounds charts
notebooks/02, 03         Defend training, then the full closed loop
```

## Install additional dependencies

```bash
pip install -r requirements.txt   # now includes scikit-learn, lightgbm, networkx
```

**GNN and sequence model are torch-optional.** `gnn_model.py` and
`sequence_model.py` both detect whether `torch`/`torch_geometric` are
installed and fall back to a CPU-only non-neural approximation if not — the
full pipeline (Identify → Generate → Feature → Defend → Reward → Feedback →
Dashboard) runs end-to-end either way. Install `torch_geometric` in Colab
(see `notebooks/02_defend_training.ipynb`, first two cells) once the rest of
your pipeline is validated, to get real GraphSAGE/LSTM results for your
final numbers.

```bash
# Only needed for the real (non-fallback) GNN/sequence models — run in Colab:
pip install torch
pip install torch_geometric  # match the version to your torch/CUDA build, see notebook 02
```

## Run order (full project, start to finish)

1. `notebooks/00_fit_distributions.ipynb` — fit real distribution params (Part 1)
2. `notebooks/01_identify_and_generate.ipynb` — validate Identify + Generate + fidelity (Part 1)
3. `notebooks/02_defend_training.ipynb` — validate the Defend stack on one round, check
   held-out-scenario generalization, confirm precision/recall/F1/FPR all get reported
4. `notebooks/03_full_loop.ipynb` — **the actual demo entry point.** Runs 3-5 full rounds
   of the closed loop and builds your dashboard artifacts.

You can also skip straight to step 4 for your first end-to-end test —
`FraudRedTeamLoop` runs Identify/Generate/Defend itself every round from
scratch, it doesn't require having run 01/02 first. Just make sure step 1
(distribution fitting) has been done at least once.

## Testing

```bash
pytest tests/ -v
```

`tests/test_generate_pipeline.py` covers Red team (validator distinctness,
legit-traffic bounds, null injection, end-to-end generation).
`tests/test_defend_pipeline.py` covers Blue team (no-future-leakage in the
velocity store, ring detection in the graph, cold-start handling, full
Defend-bundle training, blue/red reward correctness, feedback generation).
Both suites pass with zero external API calls needed (LLM calls are only in
the Identify engine, exercised separately in notebooks 01/03).

## What "done" looks like

Run `notebooks/03_full_loop.ipynb` for 3-5 rounds, then check:
- `data/coverage_matrix.csv` — 8-10+ distinct scenarios across 5 categories
- `data/dashboard_log.csv` — recall/precision/F1/FPR per round
- `data/final_dashboard.png` — the 4-panel chart to lead your demo with

Per the project requirements doc: the two things to show judges first are
the coverage matrix (breadth) and the detection-rate-over-rounds chart
(learning happened) — before any single accuracy number.
