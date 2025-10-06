import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment_transform=None, minority_class=None):
        """
        Custom dataset for OCT images with class-conditional augmentation.

        Args:
            csv_file (str): CSV path with image filenames and labels.
            img_dir (str): Directory containing images.
            transform (callable): Base transform (applied to all images).
            augment_transform (callable): Extra transform for minority class.
            minority_class (int): Label of minority class.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment_transform = augment_transform
        self.minority_class = minority_class

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[idx, 2])

        # Apply augmentation only for minority class
        if self.augment_transform and label == self.minority_class:
            image = self.augment_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label
