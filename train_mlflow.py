# train_mlflow.py
# PulseNet AI — MLflow-Tracked Training Pipeline
# Run : python train_mlflow.py
# View: mlflow ui  →  http://localhost:5000
#
# ── Audit fixes in this file ──────────────────────────────────────────────
# [Blocker #3] Class weights now computed from actual loaded label distribution
# [Medium #1]  sha256_file imported from utils (was copy-pasted here before)
# [High #2]    ONNX export: dummy_in.cpu() called before export, model.eval()
#              ensured, onnx.checker.check_model() validates the export.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning,   module="torch")

import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

from model import ResNet1D
from utils import (
    load_and_split_data,
    compute_class_weights,   # [AUDIT Blocker #3 FIX]
    sha256_file,             # [AUDIT Medium #1 FIX]
    plot_losses,
    plot_confusion_matrix,
    evaluate_model,
    verify_data_integrity,
)

# ── Paths ──────────────────────────────────────────────────────────────────
NORMAL_CSV   = "data/ptbdb_normal.csv"
ABNORMAL_CSV = "data/ptbdb_abnormal.csv"
MODEL_OUT    = "models/resnet1d_ecg_model.pth"
SCALER_OUT   = "models/scaler.pkl"
MANIFEST_OUT = "models/manifest.json"
ONNX_OUT     = "models/resnet1d.onnx"
CM_OUT       = "outputs/confusion_matrix.png"
LOSS_OUT     = "outputs/loss_curve.png"

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Data integrity ─────────────────────────────────────────────────────────
verify_data_integrity(NORMAL_CSV, ABNORMAL_CSV)

# ── Hyperparameters ────────────────────────────────────────────────────────
HPARAMS = {
    "num_epochs":         100,
    "batch_size":          64,
    "learning_rate":      1e-4,
    "weight_decay":       5e-4,
    "patience":            10,
    "min_delta":          1e-4,
    "dropout":             0.5,
    "scheduler_factor":    0.5,
    "scheduler_patience":   5,
    "architecture":   "ResNet1D",
    "dataset":        "PTB-DB",
    "seed":           SEED,
    "class_weighted_loss": True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── MLflow setup ──────────────────────────────────────────────────────────
# Default URI matches Docker Compose service name.
# Override for local: MLFLOW_TRACKING_URI=http://localhost:5000 python train_mlflow.py
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("pulsenet-ecg-classification")

with mlflow.start_run(run_name="resnet1d-ecg-v4"):

    mlflow.log_params(HPARAMS)

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler, _, y_train_np = load_and_split_data(
        NORMAL_CSV, ABNORMAL_CSV,
        batch_size=HPARAMS["batch_size"],
        random_state=SEED,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = ResNet1D(num_classes=2).to(device)

    # ── Class weights — from ACTUAL loaded label distribution ──────────────
    # [AUDIT Blocker #3 FIX]: was n_normal=4046, n_abnormal=10506 (hardcoded)
    class_weights = compute_class_weights(y_train_np, device)
    mlflow.log_param("class_weights_computed", class_weights.cpu().tolist())

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=HPARAMS["scheduler_factor"],
        patience=HPARAMS["scheduler_patience"],
    )

    # ── Training loop ──────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss    = float("inf")
    patience_count   = 0
    best_model_state = None

    for epoch in range(HPARAMS["num_epochs"]):
        model.train()
        total_train = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                total_val += criterion(model(bX), by).item()
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        mlflow.log_metrics({
            "train_loss": avg_train,
            "val_loss":   avg_val,
            "lr":         optimizer.param_groups[0]["lr"],
        }, step=epoch)

        print(f"Epoch [{epoch+1:3d}]  Train: {avg_train:.4f}  Val: {avg_val:.4f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_val_loss - HPARAMS["min_delta"]:
            best_val_loss    = avg_val
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count   = 0
        else:
            patience_count += 1
            if patience_count >= HPARAMS["patience"]:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    # ── Evaluate on test set ───────────────────────────────────────────────
    y_true, y_pred = evaluate_model(model, test_loader, device, "TEST")

    model.eval()
    y_scores = []
    with torch.no_grad():
        for bX, _ in test_loader:
            logits = model(bX.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
            y_scores.extend(probs)

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score,
    )
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc  = average_precision_score(y_true, y_scores)

    mlflow.log_metrics({
        "test_accuracy":          round(accuracy_score(y_true, y_pred),  4),
        "test_f1_macro":          round(f1_score(y_true, y_pred, average="macro"), 4),
        "test_precision_abnormal": round(precision_score(y_true, y_pred, pos_label=1), 4),
        "test_recall_abnormal":   round(recall_score(y_true, y_pred, pos_label=1),    4),
        "test_roc_auc":           round(roc_auc, 4),
        "test_pr_auc":            round(pr_auc,  4),
    })
    print(f"\nROC-AUC : {roc_auc:.4f}   PR-AUC : {pr_auc:.4f}")

    # Threshold sensitivity table — critical for clinical triage
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    y_scores_arr = np.array(y_scores)
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = (y_scores_arr >= thresh).astype(int)
        p = precision_score(y_true, preds_t, pos_label=1, zero_division=0)
        r = recall_score(y_true, preds_t, pos_label=1, zero_division=0)
        f = f1_score(y_true, preds_t, pos_label=1, zero_division=0)
        print(f"{thresh:>10.1f} {p:>10.3f} {r:>10.3f} {f:>8.3f}")
        mlflow.log_metrics({
            f"thresh_{thresh}_precision": round(p, 4),
            f"thresh_{thresh}_recall":    round(r, 4),
            f"thresh_{thresh}_f1":        round(f, 4),
        })

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_losses(train_losses, val_losses, save_path=LOSS_OUT)
    plot_confusion_matrix(y_true, y_pred, save_path=CM_OUT)
    mlflow.log_artifact(LOSS_OUT)
    mlflow.log_artifact(CM_OUT)

    # ── Save model weights ─────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_OUT)
    mlflow.pytorch.log_model(model, artifact_path="model")

    # ── ONNX export ────────────────────────────────────────────────────────
    # [AUDIT High #2 FIX]: Original code had:
    #   dummy_in = torch.randn(1,1,187).to(device)   ← on CUDA
    #   torch.onnx.export(model.cpu(), dummy_in.cuda(), ...)  ← MIXED DEVICES
    # This raises RuntimeError or produces a broken ONNX graph.
    # FIX: Move both model and dummy_in to CPU before export.
    model.eval()
    model_cpu  = model.cpu()
    dummy_cpu  = torch.randn(1, 1, 187)   # always CPU — no .to(device)
    torch.onnx.export(
        model_cpu, dummy_cpu, ONNX_OUT,
        input_names=["ecg_signal"], output_names=["logits"],
        dynamic_axes={"ecg_signal": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    # Validate the exported graph
    try:
        import onnx
        onnx_model = onnx.load(ONNX_OUT)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX export validated OK → {ONNX_OUT}")
    except ImportError:
        print(f"ONNX export saved (install 'onnx' package to validate) → {ONNX_OUT}")
    # Restore model to original device for remaining ops
    model = model_cpu.to(device)
    mlflow.log_artifact(ONNX_OUT)

    # ── Save scaler ────────────────────────────────────────────────────────
    import joblib
    joblib.dump(scaler, SCALER_OUT)
    mlflow.log_artifact(SCALER_OUT)

    # ── Save manifest ──────────────────────────────────────────────────────
    # [AUDIT Medium #1 FIX]: sha256_file imported from utils
    manifest = {
        "model_hash":      sha256_file(MODEL_OUT),
        "scaler_hash":     sha256_file(SCALER_OUT),
        "train_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "seed":            SEED,
        "roc_auc":         round(roc_auc, 4),
        "pr_auc":          round(pr_auc,  4),
        "class_weights":   class_weights.cpu().tolist(),
        "architecture":    "ResNet1D",
        "dataset":         "PTB-DB-Kaggle-v2",
        "split_note": (
            "Row-level split used (Kaggle CSV has no patient IDs). "
            "Use raw PhysioNet PTB-DB with patient_ids= for proper patient-level split. "
            "Expect metrics to drop 3-8% with correct split."
        ),
    }
    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)
    mlflow.log_artifact(MANIFEST_OUT)
    print(f"Run complete. ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"  Model + scaler + manifest logged to MLflow and saved to disk.")
