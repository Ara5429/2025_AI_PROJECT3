import os
import time
import argparse
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import create_model
from dataset import create_dataloaders


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

        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode="max", 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config["patience"])

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        self.best_val_acc = 0.0
        self.best_model_state = None

    def _run_epoch(self, loader, train=True):
        """단일 에포크 실행"""
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            pbar = tqdm(loader, leave=False, 
                       desc="Train" if train else "Val")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

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
                
                # Progress bar 업데이트
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total*100:.1f}%'
                })

        return total_loss / total, correct / total

    def train(self):
        """전체 학습 실행"""
        print("=" * 60)
        print("TRAINING STARTED")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.cfg["epochs"]):
            print(f"\nEpoch [{epoch+1}/{self.cfg['epochs']}]")
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["lr"].append(current_lr)

            # Training
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            
            # Validation
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Update scheduler
            self.scheduler.step(val_acc)

            # Print results
            print(
                f"  Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%\n"
                f"  Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%\n"
                f"  LR: {current_lr:.6f}"
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint("best_model.pth")
                print("  ✓ Best model saved!")

            # Early stopping check
            if self.early_stopping.step(val_acc):
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print("=" * 60)

        # Save final model and plots
        self._save_checkpoint("final_model.pth")
        self._save_history()
        self._plot_curves()
        
        return self.history

    def _save_checkpoint(self, name):
        """체크포인트 저장"""
        path = os.path.join(self.cfg["save_dir"], name)
        torch.save(
            {
                "model": self.model.state_dict() if name == "final_model.pth" 
                         else self.best_model_state,
                "best_val_acc": self.best_val_acc,
                "config": self.cfg,
                "history": self.history,
            },
            path,
        )
        print(f"  Checkpoint saved: {path}")

    def _save_history(self):
        """학습 히스토리 JSON 저장"""
        path = os.path.join(self.cfg["save_dir"], "training_history.json")
        
        # numpy/tensor를 python 타입으로 변환
        history_json = {
            k: [float(v) for v in vals] 
            for k, vals in self.history.items()
        }
        history_json['best_val_acc'] = float(self.best_val_acc)
        history_json['config'] = {
            k: str(v) if not isinstance(v, (int, float, bool, str)) else v 
            for k, v in self.cfg.items()
        }
        
        with open(path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"  History saved: {path}")

    def _plot_curves(self):
        """학습 곡선 시각화"""
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(epochs, self.history["train_loss"], 'b-', 
                    label="Train Loss", linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"], 'r-', 
                    label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy curve
        axes[1].plot(epochs, [a*100 for a in self.history["train_acc"]], 'b-',
                    label="Train Acc", linewidth=2)
        axes[1].plot(epochs, [a*100 for a in self.history["val_acc"]], 'r-',
                    label="Val Acc", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Best accuracy 표시
        best_epoch = self.history["val_acc"].index(max(self.history["val_acc"])) + 1
        axes[1].axvline(x=best_epoch, color='green', linestyle='--', 
                       label=f'Best: {self.best_val_acc*100:.1f}%')

        plt.suptitle("[Fig. 1] Training and Validation Loss/Accuracy Curves", 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(self.cfg["save_dir"], "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Training curves saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Drowsy Driving Detection Model"
    )
    
    # Data arguments
    parser.add_argument("--image_dir", required=True, 
                       help="Directory containing images")
    parser.add_argument("--label_dir", required=True,
                       help="Directory containing JSON labels")
    
    # Model arguments
    parser.add_argument("--backbone", default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50"],
                       help="Backbone architecture")
    parser.add_argument("--pretrained", action="store_true",
                       help="Use ImageNet pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="Freeze backbone weights")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--patience", type=int, default=7,
                       help="Early stopping patience")
    
    # Output arguments
    parser.add_argument("--save_dir", default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # DataLoaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(
        args.image_dir,
        args.label_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Model
    print("\nCreating model...")
    model = create_model(
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        device=device,
    )

    # Config
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "device": device,
        "save_dir": args.save_dir,
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()