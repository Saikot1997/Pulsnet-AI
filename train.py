# train.py
# PulseNet AI — End-to-End Training Pipeline (no MLflow)
# Run: python train.py
# Expects: data/ptbdb_normal.csv and data/ptbdb_abnormal.csv

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning,   module="torch")

import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn

# ── Reproducibility seeds ──────────────────────────────────────────────────
# Required for hospital validation — reproducible training runs.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

from model import ResNet1D
from utils import (
    load_and_split_data,
    compute_class_weights,   # [AUDIT Blocker #3 FIX] — weights from actual data
    sha256_file,             # [AUDIT Medium #1 FIX] — centralised in utils
    plot_losses,
    plot_confusion_matrix,
    evaluate_model,
    print_model_summary,
    verify_data_integrity,
)

# ── Paths ──────────────────────────────────────────────────────────────────
NORMAL_CSV   = "data/ptbdb_normal.csv"
ABNORMAL_CSV = "data/ptbdb_abnormal.csv"
MODEL_OUT    = "models/resnet1d_ecg_model.pth"
SCALER_OUT   = "models/scaler.pkl"
MANIFEST_OUT = "models/manifest.json"
CM_OUT       = "outputs/confusion_matrix.png"
LOSS_OUT     = "outputs/loss_curve.png"

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Data Integrity Check ───────────────────────────────────────────────────
# Run once without hashes to get SHA-256 values, then pin them here.
# Pinned example (Kaggle ECG Heartbeat Categorization v2):
#   verify_data_integrity(NORMAL_CSV, ABNORMAL_CSV,
#       normal_sha256="abc123...", abnormal_sha256="def456...")
verify_data_integrity(NORMAL_CSV, ABNORMAL_CSV)

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ────────────────────────────────────────────────────────
NUM_EPOCHS     = 100
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 5e-4
PATIENCE_LIMIT = 10
MIN_DELTA      = 1e-4

# ── Data ───────────────────────────────────────────────────────────────────
# load_and_split_data now returns y_train_np for class weight computation.
# [AUDIT Critical #2 NOTE]: Kaggle PTB-DB CSV has no patient IDs.
# Pass patient_ids= for a proper patient-level split (see utils.py).
train_loader, val_loader, test_loader, scaler, _, y_train_np = load_and_split_data(
    NORMAL_CSV, ABNORMAL_CSV, batch_size=BATCH_SIZE, random_state=SEED
)

# ── Model ──────────────────────────────────────────────────────────────────
model = ResNet1D(num_classes=2).to(device)
print_model_summary(model)

# ── Class Weights — computed from ACTUAL loaded label distribution ─────────
# [AUDIT Blocker #3 FIX]: Do NOT hardcode n_normal=4046, n_abnormal=10506.
# Different Kaggle PTB-DB versions have different row counts; hardcoded values
# are silently wrong on version mismatch and bias the model with no error raised.
class_weights = compute_class_weights(y_train_np, device)

# ── Loss, Optimiser, Scheduler ────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5,
)

# ── Training Loop ──────────────────────────────────────────────────────────
train_losses, val_losses = [], []
best_val_loss    = float("inf")
patience_count   = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    for bX, by in train_loader:
        bX, by = bX.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bX), by)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for bX, by in val_loader:
            bX, by = bX.to(device), by.to(device)
            total_val_loss += criterion(model(bX), by).item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}]  "
          f"Train Loss: {avg_train_loss:.4f}  |  "
          f"Val Loss: {avg_val_loss:.4f}  |  "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    if avg_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss    = avg_val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count   = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE_LIMIT:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Best model weights restored.")

# ── Plots ──────────────────────────────────────────────────────────────────
plot_losses(train_losses, val_losses, save_path=LOSS_OUT)

# ── Evaluate ───────────────────────────────────────────────────────────────
y_true_val,  y_pred_val  = evaluate_model(model, val_loader,  device, "VALIDATION")
y_true_test, y_pred_test = evaluate_model(model, test_loader, device, "TEST")
plot_confusion_matrix(y_true_test, y_pred_test,
                      title="Confusion Matrix — Test Set", save_path=CM_OUT)

# ── Save model ─────────────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_OUT)
print(f"Model saved → {MODEL_OUT}")

# ── Save scaler ────────────────────────────────────────────────────────────
import joblib
joblib.dump(scaler, SCALER_OUT)
print(f"Scaler saved → {SCALER_OUT}")

# ── Save manifest (model/scaler hash handshake) ────────────────────────────
# [AUDIT Medium #1 FIX]: sha256_file imported from utils — not redefined here.
manifest = {
    "model_hash":      sha256_file(MODEL_OUT),
    "scaler_hash":     sha256_file(SCALER_OUT),
    "train_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "seed":            SEED,
    "architecture":    "ResNet1D",
    "dataset":         "PTB-DB",
    "class_weights":   class_weights.cpu().tolist(),
}
with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest saved → {MANIFEST_OUT}")
print(f"  model_hash : {manifest['model_hash'][:16]}...")
print(f"  scaler_hash: {manifest['scaler_hash'][:16]}...")
