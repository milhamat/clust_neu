import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from config import IMG_SIZE


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class FlatFolderDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root_dir))

        for idx, cls in enumerate(classes):

            self.class_to_idx[cls] = idx

            cls_dir = os.path.join(root_dir, cls)

            for img in os.listdir(cls_dir):

                self.samples.append(
                    (
                        os.path.join(cls_dir, img),
                        idx
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        img = transform(img)

        return img, label