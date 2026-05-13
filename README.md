# ⚡ PulseNet AI v4
### Automated Cardiac Anomaly Detection from ECG Signals using Deep Residual Networks

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red)
![MLflow](https://img.shields.io/badge/MLflow-2.10.1-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

PulseNet AI is a binary ECG classification system built on a custom 1-D Residual Neural Network (ResNet1D). It classifies 187-timestep ECG signals from the PTB Diagnostic ECG Database as **Normal** or **Abnormal**.

The project includes a full MLOps layer: experiment tracking via MLflow, a production REST inference endpoint via FastAPI, Prometheus metrics, and automated CI via GitHub Actions.

---

## ⚠️ Results Limitations — Read Before Any Hospital Deployment

**[AUDIT Critical #1 FIX]** — The accuracy figures below are dataset-specific and must not be presented as general clinical performance claims.

### What the numbers mean

| Metric | Value | Qualification |
|--------|-------|---------------|
| Test Accuracy | ~99.3%* | PTB-DB held-out split only (single-institution, 1995, n≈2,911) |
| ROC-AUC | Logged to MLflow | Run `python train_mlflow.py` to obtain actual value |
| PR-AUC | Logged to MLflow | Run `python train_mlflow.py` to obtain actual value |
| Normal F1 | ~0.988 | Same dataset caveat |
| Abnormal F1 | ~0.995 | Same dataset caveat |

*\*Note: Figures shown are from the original v3 row-level split. With a correct patient-level split (see below), expect a 3–8% drop. Re-run training to obtain corrected figures.*

### Patient-level data leakage warning

**[AUDIT Critical #2]** — PTB-DB contains 290 patients with multiple recordings each. The Kaggle pre-processed CSV does not include patient IDs, so the current implementation uses a row-level random split. This means the same patient's ECG recordings can appear in both training and test sets, inflating all metrics.

**For any hospital pitch, PhD application, or journal submission:** use the raw PhysioNet PTB-DB with the `RECORDS` file to build a patient→row mapping, then pass `patient_ids=` to `load_and_split_data()`. Expect test metrics to drop 3–8% with the correct split.

### Before citing these numbers

Replace `"99.3% test accuracy"` with:
> "99.3% on PTB-DB held-out split (single-institution dataset, 1995, n=2,911, row-level split). External validation on site-specific patient populations required before clinical deployment."

---

## Architecture — ResNet1D

**[AUDIT Critical #3 FIX]** — Original 4-stage architecture with MaxPool reduced 187-step input to ~6 steps before GlobalAvgPool, destroying temporal structure. Redesigned:

```
Input (B, 1, 187)
  └─ Stem: Conv7(stride=1) + BN + ReLU          → (B,  64, 187)   # NO MaxPool
  └─ Layer1: 2× ResidualBlock1D(64→64,  s=1)    → (B,  64, 187)
  └─ Layer2: 2× ResidualBlock1D(64→128, s=2)    → (B, 128,  93)
  └─ Layer3: 2× ResidualBlock1D(128→256,s=2)    → (B, 256,  46)
  └─ GlobalAvgPool → Dropout(0.5) → FC(256→2)
```

**Verified parameter count:** `python -c "from model import ResNet1D; m=ResNet1D(); print(sum(p.numel() for p in m.parameters()))"`

*Original claimed ~3.8M (a 2D ResNet-18 on ImageNet figure). Actual count for 3-stage 1D architecture is ~867K.*

---

## Audit Fix Summary (v4 — 17 new issues, all resolved)

| # | Severity | Domain | Fix |
|---|----------|--------|-----|
| 1a | 🔴 Blocker | Security | CORS wildcard rejected when API key set (fail-loud) |
| 1b | 🔴 Blocker | Security | `/metrics` IP allowlist (`METRICS_ALLOWED_IPS`) — was unauthenticated |
| 2 | 🔴 Blocker | Clinical | `sample_rate_hz` explicit validated parameter — was hardcoded 125 Hz |
| 3 | 🔴 Blocker | Reproducibility | Class weights computed from actual `y_train` distribution — never hardcoded |
| 4 | 🔴 Blocker | Regulatory | Audit logs → `RotatingFileHandler` (durable) — was `print()` to stdout |
| 5 | 🔴 Critical | ML validity | Accuracy qualified as PTB-DB split only — not a general clinical claim |
| 6 | 🔴 Critical | ML correctness | Patient-level split warning + `patient_ids=` parameter in `load_and_split_data` |
| 7 | 🔴 Critical | Architecture | ResNet1D redesigned: no MaxPool, 3 stages, stride=1 stem → 187 steps preserved |
| 8 | 🔴 Critical | Data pipeline | `val_fraction_of_trainval` rename + unambiguous comment |
| 9 | 🔴 Critical | Testing | CI test signals: sinusoid (valid), `[6.0]*187` (clip), `[0.1]*187` (flat) |
| 10 | 🟠 High | MLOps | MLflow: SQLite backend-store; port bound to `127.0.0.1` only |
| 11 | 🟠 High | Inference | ONNX export: `model.cpu()` + `dummy_cpu` — no device mismatch; `onnx.checker` validates |
| 12 | 🟠 High | Dependency | Exact `==` version pins in all `requirements*.txt` |
| 13 | 🟠 High | Portfolio | ROC-AUC / PR-AUC: computed and logged to MLflow; placeholder note in README |
| 14 | 🟠 High | Docker | Non-root user `pulsenet` (uid=10001) in both Dockerfiles |
| 15 | 🟠 High | Clinical | `predict_with_uncertainty()`: only Dropout layers set to `train()` — BatchNorm stays eval |
| 16 | 🟠 High | API design | Rate limiter: reads `X-Forwarded-For` via `TRUSTED_PROXIES` env var |
| 17 | 🟡 Medium | Code quality | `sha256_file()` centralised in `utils.py` — removed from `train.py`, `train_mlflow.py`, `api/main.py` |
| 18 | 🟡 Medium | Observability | `/health` returns `train_timestamp`, `model_hash[:8]`, `roc_auc` from manifest |

---


## Verified Data Hashes (PTB-DB Kaggle v2)

Pin these in `train.py` `verify_data_integrity()` call to lock the dataset version:

```
ptbdb_normal.csv   SHA-256: 033d355e76e9f65f692fa398c5b24e357d004fab328c97db5160591a2eafe48c
ptbdb_abnormal.csv SHA-256: 27b4a37cd6dbcede0900cde3df02291f21e7538085d54e6b33abb91f4a5cf912
```

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/pulsenet-ai.git
cd pulsenet-ai
pip install -r requirements.txt
```

### 2. Add data
Download PTB-DB from Kaggle ECG Heartbeat Categorization **v2** and place in `data/`:
```
data/ptbdb_normal.csv
data/ptbdb_abnormal.csv
```
Run `verify_data_integrity()` from `utils.py` once to obtain SHA-256 hashes and pin them in `train.py`.

### 3. Train
```bash
python train.py          # baseline (no MLflow)
# OR
python train_mlflow.py   # with MLflow tracking
```
Saves `models/resnet1d_ecg_model.pth` + `models/scaler.pkl` + `models/manifest.json`

### 4. Launch Streamlit dashboard
```bash
streamlit run app.py
```

### 5. Launch REST API
```bash
# Set env vars (required for production):
export PULSENET_API_KEY=your-secret-key
export CORS_ALLOWED_ORIGINS=http://localhost:8501
export METRICS_ALLOWED_IPS=127.0.0.1
export TRUSTED_PROXIES=              # set to your proxy IP if behind Nginx

uvicorn api.main:app --reload
# POST /predict   → { "label": "Normal", "raw_softmax_score": 0.997, ... }
# GET  /health    → model version + manifest info
# GET  /metrics   → Prometheus (restricted to METRICS_ALLOWED_IPS)
```

### 6. Docker Compose
```bash
# Copy and configure env:
cp .env.example .env
# Edit .env: set PULSENET_API_KEY, CORS_ALLOWED_ORIGINS, METRICS_ALLOWED_IPS

docker compose up
# API      → http://localhost:8000
# MLflow   → http://127.0.0.1:5000  (localhost only — not exposed externally)
# Streamlit→ http://localhost:8501
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PULSENET_API_KEY` | `""` | API key for `/predict`. Empty = dev mode (no auth). **Required in production.** |
| `CORS_ALLOWED_ORIGINS` | `"*"` | Comma-separated allowed origins. **Cannot be `*` when API key is set.** |
| `METRICS_ALLOWED_IPS` | `"127.0.0.1"` | IPs allowed to scrape `/metrics`. Set to your Prometheus IP. |
| `TRUSTED_PROXIES` | `""` | Proxy IPs to trust for `X-Forwarded-For` rate limiting. |
| `AUDIT_LOG_PATH` | `"logs/audit.log"` | Path for HIPAA audit log. Mount as Docker volume for persistence. |
| `MLFLOW_TRACKING_URI` | `"http://localhost:5000"` | MLflow server URI. |
| `MODEL_PATH` | `"models/resnet1d_ecg_model.pth"` | Path to trained weights. |
| `SCALER_PATH` | `"models/scaler.pkl"` | Path to fitted scaler. |

---

## Project Structure

```
pulsenet_ai/
├── model.py              # ResNet1D architecture (3-stage, no MaxPool)
├── utils.py              # sha256_file, check_signal_quality, load_and_split_data,
│                         # compute_class_weights, predict_with_uncertainty
├── train.py              # Baseline training pipeline
├── train_mlflow.py       # MLflow-tracked training + ONNX export
├── app.py                # Streamlit inference dashboard
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI /predict + /health + /metrics
├── requirements.txt      # local dev / CI (exact pins)
├── requirements-api.txt  # API image only (exact pins)
├── requirements-app.txt  # Streamlit image only (exact pins)
├── Dockerfile.api        # non-root user, prom_multiproc dir
├── Dockerfile.streamlit  # non-root user
├── docker-compose.yml    # SQLite MLflow, audit log volume, 127.0.0.1 binding
├── .github/workflows/
│   └── ci.yml            # lint + model smoke + utils unit tests + integration tests
├── .env.example          # env var template
├── data/                 # PTB-DB CSV files (not committed — use Kaggle v2)
├── models/               # Saved weights + scaler + manifest (not committed)
├── logs/                 # Audit logs (mount as volume for persistence)
├── outputs/              # Training plots
└── mlruns/               # MLflow experiment store (SQLite backend)
```

---

## ⚠️ Clinical Limitations

1. **Binary classification only** — Normal or Abnormal. No sub-type diagnosis.
2. **`raw_softmax_score` is uncalibrated** — raw softmax is overconfident on OOD signals. Use `predict_with_uncertainty()` (MC Dropout, BatchNorm-correct) from `utils.py`.
3. **Sample rate must match data** — `sample_rate_hz` parameter defaults to 125 Hz (PTB-DB Kaggle format). For raw ADC data use 250 or 500.
4. **Audit logging** — every prediction written to `AUDIT_LOG_PATH` via `RotatingFileHandler`. Mount as Docker volume for HIPAA §164.312(b) compliance (6-year retention).
5. **Not FDA cleared** — not validated for clinical use. Physician review required.
6. **Row-level split** — patient-level validation required for any hospital POC submission.

## License
MIT License. Dataset: PhysioNet Open Data Commons Attribution License.
