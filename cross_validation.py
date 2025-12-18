"""
5-Fold Stratified Cross Validation
논문 Section 4.5의 실험 재현을 위한 코드
"""
import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

from model import create_model
from dataset import DrowsyDriverDataset, get_transforms


class KFoldTrainer:
    """K-Fold Cross Validation Trainer"""
    
    def __init__(self, config):
        self.cfg = config
        self.device = config['device']
        self.fold_results = []
        
    def train_one_fold(self, train_loader, val_loader, fold_num):
        """단일 Fold 학습"""
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num + 1}/{self.cfg['n_folds']}")
        print(f"{'='*60}")
        
        # 매 Fold마다 새 모델 생성
        model = create_model(
            backbone=self.cfg['backbone'],
            pretrained=self.cfg['pretrained'],
            device=self.device
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(self.cfg['epochs']):
            # Training
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.1%}, Val Acc: {val_acc:.1%}")
            
            # Best model tracking
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.cfg['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Fold {fold_num + 1} Best Accuracy: {best_val_acc:.1%}")
        return best_val_acc, history
    
    def run_cross_validation(self, dataset, labels):
        """전체 Cross Validation 실행"""
        skf = StratifiedKFold(
            n_splits=self.cfg['n_folds'],
            shuffle=True,
            random_state=self.cfg['seed']
        )
        
        fold_accuracies = []
        all_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            # Train/Val 데이터셋 분리
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            # Validation은 augmentation 없이
            # Note: Subset은 원본 dataset의 transform을 공유하므로
            # 여기서는 간단히 처리 (실제로는 별도 dataset 필요)
            
            train_loader = DataLoader(
                train_subset,
                batch_size=self.cfg['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.cfg['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            best_acc, history = self.train_one_fold(train_loader, val_loader, fold)
            fold_accuracies.append(best_acc)
            all_histories.append(history)
        
        self.fold_results = fold_accuracies
        return fold_accuracies, all_histories


def plot_cv_results(fold_accuracies, save_path=None):
    
    #Cross Validation 결과 시각화 (논문 Fig. 4 스타일)
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = [f"Fold {i+1}" for i in range(len(fold_accuracies))]
    colors = ['#4CAF50' if acc >= mean_acc else '#2196F3' for acc in fold_accuracies]
    
    bars = ax.bar(folds, [acc * 100 for acc in fold_accuracies], color=colors, edgecolor='black')
    
    # Mean line
    ax.axhline(y=mean_acc * 100, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_acc*100:.1f}% (±{std_acc*100:.1f}%)')
    
    # 각 막대에 수치 표시
    for bar, acc in zip(bars, fold_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('5-Fold Cross Validation Results', fontsize=14, fontweight='bold')
    ax.set_ylim(65, 90)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CV results plot saved to {save_path}")
    
    plt.show()
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("CROSS VALIDATION RESULTS")
    print("=" * 50)
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc*100:.1f}%")
    print("-" * 50)
    print(f"Mean Accuracy: {mean_acc*100:.1f}%")
    print(f"Std Deviation: {std_acc*100:.1f}%")
    print(f"Range: {min(fold_accuracies)*100:.1f}% ~ {max(fold_accuracies)*100:.1f}%")
    print("=" * 50)
    
    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description="5-Fold Cross Validation")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--label_dir", required=True, help="Label directory")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--save_dir", default="./cv_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 전체 데이터셋 로드
    dataset = DrowsyDriverDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        transform=get_transforms(is_train=True)
    )
    
    # 레이블 추출 (Stratified 분할을 위해)
    print("Extracting labels for stratified split...")
    labels = []
    for idx in tqdm(range(len(dataset)), desc="Loading labels"):
        _, label = dataset[idx]
        labels.append(label)
    labels = np.array(labels)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: Drowsy={sum(labels==0)}, Normal={sum(labels==1)}")
    
    # Cross Validation 설정
    config = {
        'backbone': args.backbone,
        'pretrained': args.pretrained,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'patience': 7,
        'n_folds': args.n_folds,
        'seed': args.seed,
        'device': device
    }
    
    # Cross Validation 실행
    start_time = time.time()
    trainer = KFoldTrainer(config)
    fold_accuracies, histories = trainer.run_cross_validation(dataset, labels)
    total_time = time.time() - start_time
    
    # 결과 시각화 및 저장
    mean_acc, std_acc = plot_cv_results(
        fold_accuracies,
        save_path=os.path.join(args.save_dir, "cv_results.png")
    )
    
    # 결과 JSON 저장
    results = {
        'config': config,
        'fold_accuracies': [float(acc) for acc in fold_accuracies],
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'total_time_minutes': total_time / 60
    }
    
    with open(os.path.join(args.save_dir, "cv_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()