import torchvision.transforms as transforms

class TransformImage:
    def __init__(self, input_size: tuple = (512, 512), augment: bool = False):
        """
        Image transformation pipeline for OCT datasets.

        Args:
            input_size (tuple): Target (H, W) of the image after resizing. E.g., (512, 512)
            augment (bool): Whether to include data augmentations (for training)
        """
        self.input_size = input_size
        self.augment = augment

    def transform(self):
        transform_list = [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        if self.augment:
            # Add mild augmentations commonly used for OCT classification
            transform_list.insert(0, transforms.RandomHorizontalFlip())
            transform_list.insert(0, transforms.RandomRotation(10))

        return transforms.Compose(transform_list)
