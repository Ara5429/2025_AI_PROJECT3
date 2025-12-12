import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class DrowsyDriverDataset(Dataset):
    """
    AI Hub 졸음운전 데이터셋용 Dataset 클래스

    Input  : Driver eye image
    Output : label (0: Drowsy, 1: Normal)
    """

    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(
            self.label_dir, img_name.replace(".jpg", ".json")
        )

        image = Image.open(img_path).convert("RGB")

        with open(label_path, "r") as f:
            label = json.load(f)["annotation"]  # 0 or 1

        if self.transform:
            image = self.transform(image)

        return image, int(label)



# Transforms
def get_transforms(is_train=True):
    """
    Data augmentation settings
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


def create_dataloaders(
    image_dir,
    label_dir,
    batch_size=32,
    val_split=0.2,
    num_workers=4,
    seed=42,
):
    

    # Full dataset (train transform will be overridden for val)
    full_dataset = DrowsyDriverDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=get_transforms(is_train=True),
    )

    # Train / Val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Validation dataset should NOT use augmentation
    val_dataset.dataset.transform = get_transforms(is_train=False)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
