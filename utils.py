# utils.py
# PulseNet AI — Preprocessing, Evaluation, and Utility Functions
#
# ── Fixes from audit ───────────────────────────────────────────────────────
# [Medium #1] sha256_file() was copy-pasted into train.py, train_mlflow.py,
#   and api/main.py — DRY violation in safety-critical paths.
#   FIX: Centralised here. All callers import from utils.
#
# [High #4]  predict_with_uncertainty() used model.train() which also flips
#   BatchNorm layers into training mode — for batch_size=1 this produces
#   degenerate statistics that dominate the "uncertainty" estimate.
#   FIX: Selectively enable only Dropout layers; leave BatchNorm in eval mode.
#
# [Critical #4] val_size comment was misleading ("20% of training set" was
#   ambiguous). FIX: Renamed param + explicit comment.
#
# [Blocker #3] Class weights: load_and_split_data now returns y_train_np so
#   callers can compute actual inverse-frequency weights from loaded data.
#   FIX: Returns y_train_np; callers must NOT use hardcoded n_normal/n_abnormal.
#
# [Critical #2] Patient-level split warning: PTB-DB on Kaggle has 290 patients
#   with multiple recordings per patient. Row-level split means the same
#   patient's heartbeats appear in both train and test — inflating all metrics.
#   FIX: Warning added; patient_ids parameter accepted when available.
#   NOTE: The Kaggle pre-processed CSV does not include patient IDs. To do a
#   proper patient-level split, use the raw PhysioNet PTB-DB with the
#   RECORDS file to build a patient→row mapping, then pass patient_ids here.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning,   module="torch")

import hashlib
import logging
import os

import matplotlib
matplotlib.use("Agg")   # headless — no display server (Docker/CI safe)

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from sklearn.utils        import shuffle
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import classification_report, confusion_matrix
from torch.utils.data       import TensorDataset, DataLoader

sns.set(style="whitegrid")

logger = logging.getLogger(__name__)


# ── Shared utility: SHA-256 file hash ─────────────────────────────────────
# [AUDIT FIX Medium #1]: was copy-pasted in 3 files. Centralised here.
# All files that need hashing (train.py, train_mlflow.py, api/main.py)
# must import this function from utils.
def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file, reading in 64-KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Data loading & splitting ───────────────────────────────────────────────
def load_and_split_data(
    normal_path: str,
    abnormal_path: str,
    batch_size: int = 64,
    test_size: float = 0.20,
    val_fraction_of_trainval: float = 0.20,
    random_state: int = 42,
    patient_ids: "np.ndarray | None" = None,
):
    """
    Load PTB-DB CSVs, merge, shuffle, split (train/val/test), scale,
    convert to PyTorch TensorDatasets, and return DataLoaders + scaler.

    ── Patient-level split (AUDIT Critical #2) ───────────────────────────
    PTB-DB contains 290 patients with multiple recordings each. A row-level
    random split causes data leakage: the same patient's heartbeat waveforms
    can appear in both training and test sets, inflating metrics.

    If `patient_ids` is provided (np.ndarray of int, shape (N,)), the function
    performs a GroupShuffleSplit on patient IDs to ensure no patient appears
    in both train and test. This is the correct approach for any hospital
    validation submission.

    If `patient_ids` is None (Kaggle pre-processed CSV — no IDs available),
    a row-level split is used with a PROMINENT WARNING. The Kaggle CSV strips
    patient ID metadata. To obtain patient IDs, use the raw PhysioNet PTB-DB.

    ── Split fractions ───────────────────────────────────────────────────
    Parameters (AUDIT Critical #4 — renamed for clarity):
        test_size:                 fraction of total data for test  (default 0.20)
        val_fraction_of_trainval:  fraction of (train+val) for val  (default 0.20)

    Example with defaults:
        test  = 20% of total    (~2,911 rows of 14,552)
        val   = 20% of 80%      = 16% of total  (~2,329 rows)
        train = 64% of total    (~9,312 rows)

    Returns:
        train_loader, val_loader, test_loader, scaler,
        (X_test_tensor, y_test_tensor), y_train_np
        ← y_train_np is returned so callers can compute class weights
          from the ACTUAL loaded label distribution (never hardcode).
    """
    normal   = pd.read_csv(normal_path,   header=None)
    abnormal = pd.read_csv(abnormal_path, header=None)

    df = pd.concat([normal, abnormal], ignore_index=True)
    df = shuffle(df, random_state=random_state).reset_index(drop=True)

    col_names = [f"feature_{i}" for i in range(1, df.shape[1])]
    col_names.append("target")
    df.columns = col_names

    X = df.drop(columns=["target"])
    y = df["target"]

    assert y.nunique() == 2, f"Expected 2 classes, got {y.nunique()}"
    assert set(y.unique()) == {0, 1}, f"Expected labels {{0,1}}, got {set(y.unique())}"
    print(f"Label distribution — Normal(0): {(y==0).sum()}  Abnormal(1): {(y==1).sum()}")

    # ── Split ──────────────────────────────────────────────────────────────
    if patient_ids is not None:
        # Patient-level split (correct for clinical ML validation)
        print("INFO: Using patient-level GroupShuffleSplit — no patient in both train+test.")
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_outer.split(X, y, groups=patient_ids))

        gss_inner = GroupShuffleSplit(n_splits=1,
                                      test_size=val_fraction_of_trainval,
                                      random_state=random_state)
        train_val_ids = patient_ids[train_val_idx]
        rel_train_idx, rel_val_idx = next(
            gss_inner.split(X.iloc[train_val_idx], y.iloc[train_val_idx],
                            groups=train_val_ids)
        )
        train_idx = train_val_idx[rel_train_idx]
        val_idx   = train_val_idx[rel_val_idx]

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]
        X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]
    else:
        # Row-level split — WARNING for clinical use
        warnings.warn(
            "\n"
            "╔═══════════════════════════════════════════════════════════════╗\n"
            "║  DATA LEAKAGE WARNING — ROW-LEVEL SPLIT                      ║\n"
            "║  PTB-DB has 290 patients with multiple recordings each.       ║\n"
            "║  Row-level random split means the same patient's recordings   ║\n"
            "║  can appear in both train and test, inflating all metrics.    ║\n"
            "║                                                               ║\n"
            "║  FIX: Use raw PhysioNet PTB-DB to get patient IDs, then      ║\n"
            "║  pass patient_ids= to load_and_split_data().                  ║\n"
            "║  Expect test metrics to drop 3–8% with correct split.        ║\n"
            "╚═══════════════════════════════════════════════════════════════╝",
            stacklevel=2,
        )
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # val_fraction_of_trainval: fraction of the (train+val) pool for validation
        # Example: 0.20 → val = 20% of 80% = 16% of total
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_fraction_of_trainval,
            random_state=random_state,
            stratify=y_train_val,
        )

    # ── Feature scaling (fit ONLY on training set) ─────────────────────────
    # NEVER fit on val or test — that is train/test contamination.
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_val_sc    = scaler.transform(X_val)
    X_test_sc   = scaler.transform(X_test)

    # ── Tensors: (N, 187) → (N, 1, 187) ──────────────────────────────────
    def to_tensor(X_sc, y_ser):
        Xt = torch.tensor(X_sc, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y_ser.values, dtype=torch.long)
        return Xt, yt

    X_tr_t, y_tr_t = to_tensor(X_train_sc, y_train)
    X_va_t, y_va_t = to_tensor(X_val_sc,   y_val)
    X_te_t, y_te_t = to_tensor(X_test_sc,  y_test)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t),
                              batch_size=batch_size, shuffle=False)

    print(f"Split — Train: {X_tr_t.shape}  Val: {X_va_t.shape}  Test: {X_te_t.shape}")

    # Return y_train_np so callers compute class weights from actual distribution
    y_train_np = y_train.values  # np.ndarray
    return train_loader, val_loader, test_loader, scaler, (X_te_t, y_te_t), y_train_np


def compute_class_weights(y_train_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the ACTUAL training label
    distribution. Never hardcode n_normal / n_abnormal — dataset versions differ.

    [AUDIT Blocker #3 FIX]

    Args:
        y_train_np : 1-D integer numpy array of training labels (0/1)
        device     : torch device for the returned tensor

    Returns:
        class_weights : torch.FloatTensor of shape (num_classes,)

    Example (PTB-DB v2 approximate values):
        Normal   (0): 4,046 samples  → weight ≈ 1.80
        Abnormal (1): 10,506 samples → weight ≈ 0.69
    """
    n_per_class = np.bincount(y_train_np.astype(int))
    n_total     = len(y_train_np)
    weights     = n_total / (len(n_per_class) * n_per_class.astype(float))
    print(f"Computed class weights from actual data:")
    for i, (n, w) in enumerate(zip(n_per_class, weights)):
        print(f"  class {i}: n={n}  weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Data integrity ─────────────────────────────────────────────────────────
def verify_data_integrity(
    normal_path: str,
    abnormal_path: str,
    normal_sha256: "str | None" = None,
    abnormal_sha256: "str | None" = None,
):
    """
    SHA-256 hash both CSVs and warn on mismatch vs pinned hashes.

    Kaggle PTB-DB has 3 versions (v1/v2/v3) with different row counts and
    label encodings. Training on the wrong version silently degrades metrics.
    Run once without pinned hashes to get the values, then pin them.
    """
    actual_normal   = sha256_file(normal_path)
    actual_abnormal = sha256_file(abnormal_path)
    print(f"Data integrity — ptbdb_normal.csv   SHA-256: {actual_normal}")
    print(f"Data integrity — ptbdb_abnormal.csv SHA-256: {actual_abnormal}")

    if normal_sha256 and actual_normal != normal_sha256:
        raise ValueError(
            f"ptbdb_normal.csv hash mismatch!\n"
            f"  expected: {normal_sha256}\n  actual  : {actual_normal}\n"
            "Check you are using the correct Kaggle dataset version (v2)."
        )
    if abnormal_sha256 and actual_abnormal != abnormal_sha256:
        raise ValueError(
            f"ptbdb_abnormal.csv hash mismatch!\n"
            f"  expected: {abnormal_sha256}\n  actual  : {actual_abnormal}\n"
            "Check you are using the correct Kaggle dataset version (v2)."
        )
    print("Data integrity check PASSED.")
    return actual_normal, actual_abnormal


# ── MC Dropout uncertainty ─────────────────────────────────────────────────
def predict_with_uncertainty(
    model: nn.Module,
    tensor: torch.Tensor,
    n_samples: int = 30,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Monte Carlo Dropout uncertainty estimation.

    [AUDIT High #4 FIX]: Original code called model.train() which flips ALL
    submodules including BatchNorm layers into training mode. For inference
    batch_size=1, BatchNorm in training mode computes batch statistics from a
    single sample, producing degenerate mean/variance and corrupting predictions.
    The resulting "uncertainty" was dominated by BatchNorm noise, not Dropout.

    FIX: Selectively set ONLY nn.Dropout layers to training mode, leaving
    all BatchNorm layers in eval mode (using stored running statistics).

    Args:
        model    : ResNet1D (Dropout p > 0)
        tensor   : input tensor shape (1, 1, 187), already on correct device
        n_samples: number of MC forward passes (30 is a standard default)

    Returns:
        mean_probs : np.ndarray shape (2,) — averaged class probabilities
        std_probs  : np.ndarray shape (2,) — standard deviation (uncertainty proxy)

    Note:
        High std_probs[1] (≥ 0.1) indicates the model is uncertain — likely
        an out-of-distribution input. This is a better signal than raw softmax
        overconfidence.
    """
    import torch.nn.functional as F

    model.eval()   # set all layers to eval first (freezes BN running stats)

    # Selectively enable ONLY Dropout layers — leave BatchNorm in eval mode
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list.append(probs)

    model.eval()   # restore all layers to eval mode

    probs_arr  = np.array(probs_list)   # (n_samples, 2)
    mean_probs = probs_arr.mean(axis=0)
    std_probs  = probs_arr.std(axis=0)
    return mean_probs, std_probs


# ── Signal quality / OOD checks (sample-rate aware) ───────────────────────
ALLOWED_SAMPLE_RATES = {125, 250, 500}
DEFAULT_SAMPLE_RATE  = 125

def check_signal_quality(
    signal_array: np.ndarray,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """
    Validate a 1-D ECG signal before inference.
    Raises ValueError with a descriptive message on any quality failure.

    [AUDIT Blocker #2 FIX]: sample rate was hardcoded to 125 Hz everywhere.
    If a hospital feeds 500 Hz or 1000 Hz raw ADC samples, the 45 Hz cutoff
    passes everything (OOD detector silently disabled). In the opposite case,
    a valid 125 Hz ECG's first harmonic at ~60 Hz gets wrongly rejected.

    FIX: sample_rate_hz is an explicit validated parameter.

    Args:
        signal_array   : 1-D float32 array of ECG amplitudes
        sample_rate_hz : acquisition sample rate in Hz (must be in ALLOWED_SAMPLE_RATES)

    Checks:
        1. NaN/Inf         — corrupted sensor or transmission error
        2. Flat-line       — lead disconnect (std < 0.001)
        3. Amplitude clip  — amplifier saturation (|max| > 5.0 in PTB-DB units)
        4. Powerline noise — dominant FFT frequency above 45 Hz
    """
    if sample_rate_hz not in ALLOWED_SAMPLE_RATES:
        raise ValueError(
            f"sample_rate_hz={sample_rate_hz} not in allowed set {ALLOWED_SAMPLE_RATES}. "
            "PTB-DB Kaggle pre-processed CSV uses 125 Hz. "
            "For raw ADC data, use 250 or 500."
        )
    if not np.all(np.isfinite(signal_array)):
        raise ValueError("Signal contains NaN or Inf — check CSV data quality")
    if np.abs(signal_array).max() > 5.0:
        raise ValueError(
            f"Signal amplitude {np.abs(signal_array).max():.3f} > 5.0 — "
            "possible amplifier clipping or wrong unit scale"
        )
    if signal_array.std() < 0.001:
        raise ValueError("Signal is flat-line (std < 0.001) — possible lead disconnect")

    # Powerline check using the CORRECT sample rate
    fft_mag  = np.abs(np.fft.rfft(signal_array))
    fft_freq = np.fft.rfftfreq(len(signal_array), d=1.0 / sample_rate_hz)
    dominant_hz = fft_freq[np.argmax(fft_mag)]
    nyquist_hz  = sample_rate_hz / 2.0
    # Cutoff is 45 Hz (below 50/60 Hz powerline). Sanity: only check if
    # Nyquist > 45 Hz (always true for 125+ Hz, but guards edge cases).
    if nyquist_hz > 45.0 and dominant_hz > 45.0:
        raise ValueError(
            f"Dominant spectral frequency {dominant_hz:.1f} Hz > 45 Hz — "
            "possible powerline artifact (50/60 Hz interference)"
        )


# ── Evaluation & plotting ──────────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "Test",
) -> "tuple[list, list]":
    """Run inference on a DataLoader, print classification report."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch_y.numpy())
    print("=" * 55)
    print(f"{split_name} SET — Classification Report")
    print("=" * 55)
    print(classification_report(y_true, y_pred, digits=3,
                                target_names=["Normal", "Abnormal"]))
    return y_true, y_pred


def plot_losses(train_losses, val_losses, save_path=None):
    """Plot training vs validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses,   label="Val Loss",   linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix",
                           save_path=None):
    """Plot labelled seaborn confusion matrix heatmap."""
    if labels is None:
        labels = ["Normal", "Abnormal"]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label",      fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def print_model_summary(model: nn.Module) -> None:
    """Print architecture and total / trainable parameter counts."""
    print(model)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
