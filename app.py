# app.py
# PulseNet AI — Streamlit Inference Dashboard
# Run: streamlit run app.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning,   module="torch")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import streamlit as st

from model import ResNet1D
from utils import check_signal_quality, ALLOWED_SAMPLE_RATES, DEFAULT_SAMPLE_RATE

st.set_page_config(
    page_title="PulseNet AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/resnet1d_ecg_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    mdl = ResNet1D(num_classes=2)
    if not os.path.exists(MODEL_PATH):
        return None
    mdl.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    mdl.to(DEVICE).eval()
    return mdl


@st.cache_resource
def load_scaler():
    import joblib
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)


def plot_ecg_signal(signal: np.ndarray, label: str, confidence: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    color = "#e74c3c" if label == "Abnormal" else "#2ecc71"
    ax.plot(signal, color=color, linewidth=1.2)
    ax.set_title(
        f"ECG Signal — Predicted: {label}  ({confidence:.1%} raw softmax score)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("⚡ PulseNet AI")
st.markdown("""
**Automated Cardiac Anomaly Detection from ECG Signals**  
Upload a CSV row of 187 ECG time-step values (or a multi-row file) to get
a binary prediction: **Normal** or **Abnormal**.
""")

with st.sidebar:
    st.header("⚙️ Model Info")
    st.info(f"Device: **{DEVICE}**")
    st.markdown("""
| Metric | Score |
|--------|-------|
| Architecture | ResNet1D |
| Parameters | ~867K |
| Dataset | PTB-DB (Kaggle v2) |
| Split | Row-level (see note) |
    """)
    sample_rate = st.selectbox(
        "Sample Rate (Hz)",
        options=sorted(ALLOWED_SAMPLE_RATES),
        index=list(sorted(ALLOWED_SAMPLE_RATES)).index(DEFAULT_SAMPLE_RATE),
        help=(
            "Acquisition sample rate. PTB-DB Kaggle CSV = 125 Hz. "
            "For raw ADC data use 250 or 500. "
            "This affects the powerline noise detection threshold."
        ),
    )
    st.divider()
    st.warning(
        "⚠️ Research prototype only. raw_softmax_score is uncalibrated — "
        "not a calibrated clinical probability. Binary output only "
        "(Normal vs Abnormal). Not for clinical diagnosis without physician review."
    )
    st.info(
        "ℹ️ Patient-level split not applied (Kaggle CSV has no patient IDs). "
        "Metrics may be inflated due to row-level split. "
        "See README for details."
    )
    st.caption("PTB-DB · PhysioNet · PyTorch 2.x")

model  = load_model()
scaler = load_scaler()

if model is None:
    st.error("❌ Model weights not found at `models/resnet1d_ecg_model.pth`. "
             "Run `python train.py` first.")
    st.stop()

if scaler is None:
    st.error("❌ Scaler not found at `models/scaler.pkl`. "
             "Run `python train.py` first.")
    st.stop()

uploaded = st.file_uploader(
    "Upload ECG CSV (no header, 187 columns per row)",
    type=["csv"]
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, header=None)

        if df.shape[1] == 188:
            df = df.iloc[:, :187]

        if df.shape[1] != 187:
            st.error(f"Expected 187 feature columns, got {df.shape[1]}.")
            st.stop()

        st.success(f"Loaded {len(df)} ECG record(s).")

        raw_vals = df.values.astype(np.float32)

        # Signal quality pre-filter using the selected sample rate
        quality_errors = {}
        for i, row in enumerate(raw_vals):
            try:
                check_signal_quality(row, sample_rate_hz=int(sample_rate))
            except ValueError as e:
                quality_errors[i] = str(e)

        if quality_errors:
            for idx, reason in quality_errors.items():
                st.warning(f"Record {idx+1} skipped: {reason}")

        good_mask = np.array([i not in quality_errors for i in range(len(raw_vals))])
        good_vals = raw_vals[good_mask]
        good_idxs = np.where(good_mask)[0]

        results = []
        if len(good_vals) > 0:
            scaled_all = scaler.transform(good_vals)
            tensor_all = torch.tensor(scaled_all, dtype=torch.float32).unsqueeze(1)
            batch_ds   = torch.utils.data.TensorDataset(tensor_all)
            batch_dl   = torch.utils.data.DataLoader(batch_ds, batch_size=64, shuffle=False)

            all_probs = []
            with torch.no_grad():
                for (batch_x,) in batch_dl:
                    logits = model(batch_x.to(DEVICE))
                    probs  = F.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
            all_probs = np.vstack(all_probs)

            for j, orig_idx in enumerate(good_idxs):
                probs = all_probs[j]
                pred  = int(np.argmax(probs))
                label = "Abnormal" if pred == 1 else "Normal"
                results.append({
                    "Record":            orig_idx + 1,
                    "Prediction":        label,
                    "Raw Softmax Score": f"{probs[pred]:.3f}",
                    "P(Normal)":         f"{probs[0]:.3f}",
                    "P(Abnormal)":       f"{probs[1]:.3f}",
                })
                if j < 5:
                    fig = plot_ecg_signal(raw_vals[orig_idx], label, probs[pred])
                    st.pyplot(fig)
                    plt.close(fig)

        st.subheader("📊 Prediction Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        if len(results_df) > 0:
            n_normal   = (results_df["Prediction"] == "Normal").sum()
            n_abnormal = (results_df["Prediction"] == "Abnormal").sum()
            col1, col2 = st.columns(2)
            col1.metric("Normal",   n_normal)
            col2.metric("Abnormal", n_abnormal)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("↑ Upload a CSV file to begin inference.")
