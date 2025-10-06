import importlib
import os
import yaml
import shutil
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils.data_loader import ImageDataset
from src.preproc.transform_image import TransformImage


def get_model_class(model_type: str):
    """Dynamically load model class by name"""
    module = importlib.import_module("src.models.model_variants")
    return getattr(module, model_type)


def get_next_experiment_dir(base_dir="experiments"):
    """Create next sequential experiment folder like 001, 002, etc."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.isdigit()]
    if existing:
        next_id = max(int(d) for d in existing) + 1
    else:
        next_id = 1
    exp_dir = os.path.join(base_dir, f"{next_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def train_pipeline(config_path):
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
    print(f"🧠 Using device: {device}")

    # =========================================================
    # 3. DATASETS & DATALOADERS
    # =========================================================
    augment_minority = config["data"].get("augment_minority_only", False)
    use_sampler = config["data"].get("use_weighted_sampler", False)

    df_train = pd.read_csv(config["data"]["train_csv"])
    class_counts = Counter(df_train.iloc[:, 2])
    minority_class = min(class_counts, key=class_counts.get)
    print(f"Detected minority class: {minority_class}")

    # Define transforms
    base_transform = TransformImage(input_size=tuple(config["data"]["input_size"])).transform()
    augment_transform = TransformImage(input_size=tuple(config["data"]["input_size"]), augment=True).transform()

    # Create datasets
    if augment_minority:
        train_dataset = ImageDataset(
            csv_file=config["data"]["train_csv"],
            img_dir=config["data"]["train_dir"],
            transform=base_transform,
            augment_transform=augment_transform,
            minority_class=minority_class
        )
    else:
        train_dataset = ImageDataset(
            csv_file=config["data"]["train_csv"],
            img_dir=config["data"]["train_dir"],
            transform=base_transform
        )

    valid_dataset = ImageDataset(
        csv_file=config["data"]["valid_csv"],
        img_dir=config["data"]["valid_dir"],
        transform=base_transform
    )

    if use_sampler:
        targets = df_train.iloc[:, 2].values
        class_sample_count = [class_counts[i] for i in sorted(class_counts.keys())]
        weights_per_class = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        sample_weights = [weights_per_class[label] for label in targets]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            sampler=sampler,
        )
    else:
        train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True
    )
        
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(valid_dataset)}")

    # =========================================================
    # 4. MODEL INITIALIZATION
    # =========================================================
    use_inverse = config["data"].get("use_inverse_weights", False)

    ModelClass = get_model_class(config["model"]["type"])
    model = ModelClass(**config["model"]["params"]).to(device)
    criterion = nn.CrossEntropyLoss()

    if use_inverse:
        # Handle dataset Imbalance
        counts = Counter(train_dataset.annotations.iloc[:, 2])
        num_classes = len(counts)
        total = sum(counts.values())

        # Compute weights inversely proportional to frequency
        weights = [total / counts[i] for i in range(num_classes)]
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
    model.parameters(),
    lr=float(config["training"]["learning_rate"])
    )

    # =========================================================
    # 5. EXPERIMENT FOLDER SETUP
    # =========================================================
    exp_dir = get_next_experiment_dir(config["training"]["save_path"])
    run_id = os.path.basename(exp_dir)
    print(f"📂 Experiment folder: {exp_dir}")

    # Save config inside experiment folder
    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))

    # =========================================================
    # 6. TRAINING LOOP (with Early Stopping)
    # =========================================================
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience = config["training"].get("early_stop", {}).get("patience", 5)
    min_delta = config["training"].get("early_stop", {}).get("min_delta", 0.0)
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        # ----------------------
        # TRAIN
        # ----------------------
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        # ----------------------
        # VALIDATE
        # ----------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']} [Valid]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # ----------------------
        # LOGGING
        # ----------------------
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{config['training']['num_epochs']}]: "
              f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | "
              f"Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}\n")

        # ----------------------
        # EARLY STOPPING
        # ----------------------
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹ Early stopping at epoch {epoch} (no improvement in {patience} epochs)")
                break

    # =========================================================
    # 7. SAVE BEST MODEL
    # =========================================================
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
    print(f"Best model saved as {exp_dir}/best_model.pth")

    # =========================================================
    # 8️⃣ SAVE METRICS & PLOTS
    # =========================================================
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)
    print(f"Metrics saved as {exp_dir}/metrics.csv")

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()

    print(f"Loss curve saved as {exp_dir}/loss_curve.png")
    print(f"Training complete for experiment {run_id}")
