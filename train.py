import os
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import create_model
from dataset import create_dataloaders

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = config["device"]

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # scheduler & early stopping 
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=config["patience"])

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        self.best_val_acc = 0.0

    def _run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def train(self):
        print("=" * 60)
        print("Training Started")
        print(f"Device: {self.device}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.cfg["epochs"]):
            print(f"\nEpoch [{epoch+1}/{self.cfg['epochs']}]")

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step(val_acc)

            print(
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint("best_model.pth")
                print("✓ Best model saved")

            if self.early_stopping.step(val_acc):
                print("Early stopping triggered")
                break

        print(f"\nTraining finished in {(time.time()-start_time)/60:.1f} min")
        print(f"Best Val Accuracy: {self.best_val_acc*100:.2f}%")

        self._save_checkpoint("final_model.pth")
        self._plot_curves()

    def _save_checkpoint(self, name):
        path = os.path.join(self.cfg["save_dir"], name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "best_val_acc": self.best_val_acc,
                "config": self.cfg,
            },
            path,
        )

    def _plot_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train")
        plt.plot(epochs, self.history["val_loss"], label="Val")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, [a*100 for a in self.history["train_acc"]], label="Train")
        plt.plot(epochs, [a*100 for a in self.history["val_acc"]], label="Val")
        plt.title("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "training_curves.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save_dir", default="./checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        args.image_dir, args.label_dir, args.batch_size
    )

    model = create_model(
        backbone="resnet18",
        pretrained=args.pretrained,
        device=device,
    )

    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "patience": 7,
        "device": device,
        "save_dir": args.save_dir,
    }

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
