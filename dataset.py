import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_MAP = {
    "Seatbelt": 0,
    "No-Seatbelt": 1,
    "VLQ": 2
}

class SeatbeltDataset(Dataset):
    def __init__(self, root_dir, image_size=224, augment=False):
        self.samples = []
        self.transform = self._get_transforms(image_size, augment)

        for class_name, label in CLASS_MAP.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                self.samples.append(
                    (os.path.join(class_dir, img_name), label)
                )

    def _get_transforms(self, image_size, augment):
        t = []
        if augment:
            t.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomHorizontalFlip(),
            ])
        t.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        return transforms.Compose(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
