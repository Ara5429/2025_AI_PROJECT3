import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from model import create_model
from dataset import create_dataloaders


def evaluate_model(model, dataloader, device):
    """
    모델 평가 및 예측 결과 수집
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Normal class probability
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Confusion Matrix 시각화
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Drowsy (0)', 'Awake (1)']
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title='Confusion Matrix (ResNet18, Pretrained)',
        ylabel='Actual',
        xlabel='Predicted'
    )
    
    # 셀에 숫자 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm


def plot_roc_curve(y_true, y_probs, save_path=None):
    """
    ROC Curve 시각화
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    return auc_score


def print_evaluation_report(y_true, y_pred, y_probs):
    """
    평가 지표 출력
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    # Classification Report
    target_names = ['Drowsy (0)', 'Normal (1)']
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class Recall (논문의 Fig. 2에 해당)
    drowsy_recall = cm[0, 0] / cm[0].sum()
    normal_recall = cm[1, 1] / cm[1].sum()
    print(f"\nDrowsy Recall: {drowsy_recall:.1%} ({cm[0,0]}/{cm[0].sum()})")
    print(f"Normal Recall: {normal_recall:.1%} ({cm[1,1]}/{cm[1].sum()})")
    
    # AUC
    try:
        auc_score = roc_auc_score(y_true, y_probs)
        print(f"\nROC-AUC Score: {auc_score:.3f}")
    except:
        auc_score = None
        print("\nROC-AUC: Could not compute")
    
    # Overall Accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\nOverall Accuracy: {accuracy:.1%}")
    
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'drowsy_recall': drowsy_recall,
        'normal_recall': normal_recall,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--label_dir", required=True, help="Label directory")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--backbone", default="resnet18", help="Backbone architecture")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", default="./results", help="Directory to save results")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # DataLoader (validation set only)
    _, val_loader = create_dataloaders(
        args.image_dir,
        args.label_dir,
        batch_size=args.batch_size,
        val_split=0.2,
    )
    
    # Load model
    model = create_model(
        backbone=args.backbone,
        pretrained=False,  # weights will be loaded from checkpoint
        device=device,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Best validation accuracy from training: {checkpoint.get('best_val_acc', 'N/A')}")
    
    # Evaluate
    y_pred, y_true, y_probs = evaluate_model(model, val_loader, device)
    
    # Print report
    metrics = print_evaluation_report(y_true, y_pred, y_probs)
    
    # Plot and save figures
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(args.save_dir, "confusion_matrix.png")
    )
    
    if metrics['auc'] is not None:
        plot_roc_curve(
            y_true, y_probs,
            save_path=os.path.join(args.save_dir, "roc_curve.png")
        )
    
    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()