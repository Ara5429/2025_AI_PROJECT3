import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class DrowsyDriverDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 이미지 파일 목록 (jpg, jpeg, png 지원)
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        print(f"Dataset initialized with {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 이미지 로드
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # 라벨 로드 (JSON)
        # 파일명에서 확장자를 제거하고 .json 추가
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.json")
        
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        
        # annotation 필드에서 라벨 추출
        # AI Hub 데이터: 0 = Drowsy, 1 = Normal
        # 대소문자 변형 처리
        label = label_data.get("annotation", label_data.get("Annotation", 0))

        if self.transform:
            image = self.transform(image)

        return image, int(label)


def get_transforms(is_train=True, image_size=224):
    # ImageNet 정규화 값
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(
    image_dir,
    label_dir,
    batch_size=32,
    val_split=0.2,
    num_workers=4,
    seed=42,
    image_size=224,
):
    # Full dataset with train transforms
    full_dataset = DrowsyDriverDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=get_transforms(is_train=True, image_size=image_size),
    )

    # Train / Val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 배치 크기 일관성
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"\nDataLoader created:")
    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


def get_labels(dataset):
    import numpy as np
    from tqdm import tqdm
    
    labels = []
    for idx in tqdm(range(len(dataset)), desc="Extracting labels"):
        _, label = dataset[idx]
        labels.append(label)
    
    return np.array(labels)


# 테스트
if __name__ == "__main__":
    IMAGE_DIR = "./data/images"
    LABEL_DIR = "./data/labels"
    
    print("Testing dataset module...")
    
    # 경로 확인
    if not os.path.exists(IMAGE_DIR):
        print(f"Warning: Image directory not found: {IMAGE_DIR}")
        print("Please update the path and run again.")
    else:
        # 데이터셋 테스트
        dataset = DrowsyDriverDataset(
            image_dir=IMAGE_DIR,
            label_dir=LABEL_DIR,
            transform=get_transforms(is_train=True)
        )
        
        print(f"\nDataset length: {len(dataset)}")
        
        # 샘플 로드 테스트
        image, label = dataset[0]
        print(f"Sample image shape: {image.shape}")
        print(f"Sample label: {label}")
        
        # DataLoader 테스트
        train_loader, val_loader = create_dataloaders(
            IMAGE_DIR, LABEL_DIR, batch_size=4
        )
        
        # 배치 테스트
        images, labels = next(iter(train_loader))
        print(f"\nBatch images shape: {images.shape}")
        print(f"Batch labels: {labels}")
        
        print("\n✓ Dataset module test passed!")