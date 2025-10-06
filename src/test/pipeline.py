import importlib
import os
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.data_loader import ImageDataset
from src.preproc.transform_image import TransformImage


def get_model_class(model_type: str):
    """Dynamically load model class by name"""
    module = importlib.import_module("src.models.model_variants")
    return getattr(module, model_type)


def test_pipeline(config_path, model_path=None):
    # =========================================================
    # 1. LOAD CONFIG
    # =========================================================
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # =========================================================
    # 2. DEVICE SELECTION
    # =========================================================
    if config["training"]["device"] == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config["training"]["device"])
    print(f"Using device: {device}")

    # =========================================================
    # 3. LOAD TEST DATA
    # =========================================================
    transform = TransformImage(input_size=tuple(config["data"]["input_size"])).transform()

    test_dataset = ImageDataset(
        csv_file="src/data/test/annotations.csv",
        img_dir="src/data/test/images",
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples")

    # =========================================================
    # 4. MODEL INITIALIZATION
    # =========================================================
    ModelClass = get_model_class(config["model"]["type"])
    model = ModelClass(**config["model"]["params"]).to(device)

    # Load trained weights
    if model_path is None:
        # Automatically pick last trained experiment
        exp_dir = sorted(
            [d for d in os.listdir(config["training"]["save_path"]) if d.isdigit()],
            key=lambda x: int(x)
        )[-1]
        model_path = os.path.join(config["training"]["save_path"], exp_dir, "best_model.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model weights from: {model_path}")

    # =========================================================
    # 5. EVALUATE ON TEST SET
    # =========================================================
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Running Inference [Test]"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # for AUC (binary only)

    # =========================================================
    # 6. METRICS
    # =========================================================
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }

    print("\n Test Metrics:")
    for k, v in metrics.items():
        print(f"   {k:10s}: {v:.4f}")

    # AUC (for binary classification)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        metrics["AUC"] = auc
        print(f"   {'AUC':10s}: {auc:.4f}")
    except Exception:
        print("Skipping AUC (multi-class or missing probabilities)")

    # =========================================================
    # 7. SAVE RESULTS
    # =========================================================
    exp_dir = os.path.dirname(model_path)
    results_path = os.path.join(exp_dir, "test_results.csv")

    results_df = pd.DataFrame({
        "true_label": all_labels,
        "pred_label": all_preds,
        "prob_class1": all_probs
    })
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions to {results_path}")

    # =========================================================
    # 8. PLOTS (CONFUSION MATRIX + ROC)
    # =========================================================
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test)")
    plt.savefig(os.path.join(exp_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved at {exp_dir}/confusion_matrix.png")

    # ROC Curve (binary only)
    if len(set(all_labels)) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {metrics.get('AUC', 0):.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(exp_dir, "roc_curve.png"))
        plt.close()
        print(f"ROC curve saved at {exp_dir}/roc_curve.png")

    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(exp_dir, "test_metrics.csv"), index=False)
    print(f"Metrics summary saved at {exp_dir}/test_metrics.csv")

    print("\nTest evaluation complete.")
