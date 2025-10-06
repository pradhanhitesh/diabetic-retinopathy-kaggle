import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import torch


class Experiment:
    def __init__(self, base_dir: str = "experiments"):
        """
        Utility class for managing experiment directories and saving logs/models.

        Args:
            base_dir (str): Base directory for all experiments.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        # Create next experiment ID folder
        self.experiment_dir = self._create_experiment_folder()
        self.log_path = os.path.join(self.experiment_dir, "training_log.csv")
        self.config_path = os.path.join(self.experiment_dir, "config.json")

        # Initialize empty log
        self.logs = []

        print(f"🧪 New experiment created at: {self.experiment_dir}")

    def _create_experiment_folder(self):
        """Create a new experiment folder with an incremental numeric ID."""
        existing = [d for d in os.listdir(self.base_dir) if d.isdigit()]
        next_id = f"{(max(map(int, existing)) + 1) if existing else 1:03d}"
        folder = os.path.join(self.base_dir, next_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    def save_config(self, config: dict):
        """Save experiment configuration (hyperparameters, model info, etc.)"""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"⚙️ Config saved to: {self.config_path}")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
        """Log metrics per epoch to CSV and in-memory list."""
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.logs.append(log_entry)
        df = pd.DataFrame(self.logs)
        df.to_csv(self.log_path, index=False)

    def save_model(self, model, filename="best_model.pth"):
        """Save the current model checkpoint."""
        model_path = os.path.join(self.experiment_dir, filename)
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to: {model_path}")

    def plot_loss_curve(self):
        """Plot and save the training/validation loss curve."""
        if not self.logs:
            print("No logs to plot yet.")
            return

        df = pd.DataFrame(self.logs)
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.experiment_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📉 Loss curve saved to: {plot_path}")

    def summary(self):
        """Print quick summary of the experiment."""
        print(f"\n📊 Experiment Summary")
        print(f"Directory: {self.experiment_dir}")
        print(f"Log file:   {self.log_path}")
        print(f"Model file: best_model.pth\n")
