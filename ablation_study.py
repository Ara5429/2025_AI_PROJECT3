"""
Ablation Study
논문 Section 4.4의 실험 재현

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
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models

from model import DrowsyDetectionModel
from dataset import DrowsyDriverDataset


def get_transforms_ablation(use_augmentation=True):
    """Ablation용 transform"""
    if use_augmentation:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_efficientnet_model(pretrained=True, num_classes=2):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None
    
    model = models.efficientnet_b0(weights=weights)
    
    # Classifier 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    
    return model


def train_ablation_model(model, train_loader, val_loader, config, device):
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_correct, train_total = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    return best_val_acc


def count_parameters(model):

    return sum(p.numel() for p in model.parameters())


def run_ablation_study(image_dir, label_dir, config, device):
    results = {}
    
    # 기본 데이터 로더 생성
    train_transform = get_transforms_ablation(use_augmentation=True)
    val_transform = get_transforms_ablation(use_augmentation=False)
    
    # Full dataset
    full_dataset = DrowsyDriverDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=train_transform
    )
    
    # Train/Val split
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Val transform 적용 (augmentation 없이)
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"Dataset: Train={train_size}, Val={val_size}")
    
    # full
    print("\n" + "=" * 60)
    print("Experiment 1: Full Model (ResNet18 + Pretrained + Augmentation)")
    print("=" * 60)
    
    model = DrowsyDetectionModel(backbone='resnet18', pretrained=True)
    params = count_parameters(model)
    acc = train_ablation_model(model, train_loader, val_loader, config, device)
    results['Full Model'] = {'accuracy': acc, 'params': params}
    print(f"Result: {acc*100:.1f}% ({params/1e6:.1f}M params)")
    

    print("\n" + "=" * 60)
    print("Experiment 2: Without Transfer Learning (Random Init)")
    print("=" * 60)
    
    model = DrowsyDetectionModel(backbone='resnet18', pretrained=False)
    params = count_parameters(model)
    acc = train_ablation_model(model, train_loader, val_loader, config, device)
    results['w/o Transfer Learning'] = {'accuracy': acc, 'params': params}
    print(f"Result: {acc*100:.1f}% ({params/1e6:.1f}M params)")
    
  
    #data augmentation 없는 실험
    print("\n" + "=" * 60)
    print("Experiment 3: Without Data Augmentation")
    print("=" * 60)
    
    # Augmentation 없는 데이터 로더
    no_aug_dataset = DrowsyDriverDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=val_transform  # No augmentation
    )
    
    train_dataset_no_aug, val_dataset_no_aug = random_split(
        no_aug_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config['seed'])
    )
    
    train_loader_no_aug = DataLoader(
        train_dataset_no_aug, batch_size=config['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader_no_aug = DataLoader(
        val_dataset_no_aug, batch_size=config['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    model = DrowsyDetectionModel(backbone='resnet18', pretrained=True)
    params = count_parameters(model)
    acc = train_ablation_model(model, train_loader_no_aug, val_loader_no_aug, config, device)
    results['w/o Data Augmentation'] = {'accuracy': acc, 'params': params}
    print(f"Result: {acc*100:.1f}% ({params/1e6:.1f}M params)")
    
   # 백본 실험
    backbones = [
        ('ResNet18', 'resnet18'),
        ('ResNet34', 'resnet34'),
        ('ResNet50', 'resnet50'),
    ]
    
    for name, backbone in backbones:
        print("\n" + "=" * 60)
        print(f"Experiment: {name} Backbone")
        print("=" * 60)
        
        model = DrowsyDetectionModel(backbone=backbone, pretrained=True)
        params = count_parameters(model)
        acc = train_ablation_model(model, train_loader, val_loader, config, device)
        results[name] = {'accuracy': acc, 'params': params}
        print(f"Result: {acc*100:.1f}% ({params/1e6:.1f}M params)")
    
    # EfficientNet-B0
    print("\n" + "=" * 60)
    print("Experiment: EfficientNet-B0 Backbone")
    print("=" * 60)
    
    model = create_efficientnet_model(pretrained=True)
    params = count_parameters(model)
    acc = train_ablation_model(model, train_loader, val_loader, config, device)
    results['EfficientNet-B0'] = {'accuracy': acc, 'params': params}
    print(f"Result: {acc*100:.1f}% ({params/1e6:.1f}M params)")
    
    return results


def plot_ablation_results(results, save_path=None):
    """
    Ablation Study 결과 시각화 (논문 Fig. 3)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    
    transfer_data = {
        'ResNet18\n(Scratch)': results.get('w/o Transfer Learning', {}).get('accuracy', 0),
        'ResNet18\n(Pretrained)': results.get('Full Model', {}).get('accuracy', 0)
    }
    
    bars1 = ax1.bar(transfer_data.keys(), [v*100 for v in transfer_data.values()],
                    color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars1, transfer_data.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Transfer Learning Effect', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Delta annotation
    delta = (transfer_data['ResNet18\n(Pretrained)'] - transfer_data['ResNet18\n(Scratch)']) * 100
    ax1.annotate(f'+{delta:.1f}%p', xy=(1.5, 85), fontsize=14, fontweight='bold', color='green')
    
    ax2 = axes[1]
    
    backbone_data = {
        'ResNet18': results.get('ResNet18', results.get('Full Model', {})),
        'ResNet34': results.get('ResNet34', {}),
        'ResNet50': results.get('ResNet50', {}),
        'EfficientNet-B0': results.get('EfficientNet-B0', {})
    }
    
    names = list(backbone_data.keys())
    accuracies = [backbone_data[n].get('accuracy', 0) * 100 for n in names]
    params = [backbone_data[n].get('params', 0) / 1e6 for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, accuracies, width, label='Accuracy (%)', 
                    color='#3498DB', edgecolor='black')
    
    ax2_twin = ax2.twinx()
    bars3 = ax2_twin.bar(x + width/2, params, width, label='Parameters (M)',
                         color='#E74C3C', edgecolor='black', alpha=0.7)
    
    ax2.set_xlabel('Model Architecture', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color='#3498DB')
    ax2_twin.set_ylabel('Parameters (M)', fontsize=12, color='#E74C3C')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylim(0, 100)
    ax2_twin.set_ylim(0, max(params) * 1.5)
    
    ax2.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Ablation results plot saved to {save_path}")
    
    plt.show()


def print_ablation_table(results):
    """Ablation Study 결과 표 출력 (논문 Table 2 스타일)"""
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Configuration':<30} {'Accuracy':<15} {'Δ Accuracy':<15}")
    print("-" * 70)
    
    baseline_acc = results.get('Full Model', {}).get('accuracy', 0)
    
    for config_name, data in results.items():
        acc = data.get('accuracy', 0)
        delta = acc - baseline_acc
        
        if config_name == 'Full Model':
            delta_str = "-"
        else:
            delta_str = f"{delta*100:+.1f}%p"
        
        print(f"{config_name:<30} {acc*100:.1f}%{'':<10} {delta_str:<15}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--label_dir", required=True, help="Label directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="./ablation_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'patience': 7,
        'seed': args.seed
    }
    
    # Ablation Study 실행
    start_time = time.time()
    results = run_ablation_study(args.image_dir, args.label_dir, config, device)
    total_time = time.time() - start_time
    
    # 결과 출력 및 시각화
    print_ablation_table(results)
    plot_ablation_results(results, save_path=os.path.join(args.save_dir, "ablation_results.png"))
    
    # 결과 JSON 저장
    json_results = {k: {'accuracy': float(v['accuracy']), 'params': int(v['params'])} 
                    for k, v in results.items()}
    json_results['total_time_minutes'] = total_time / 60
    
    with open(os.path.join(args.save_dir, "ablation_results.json"), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()