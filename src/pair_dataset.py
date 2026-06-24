import random

from torch.utils.data import Dataset

from dataset import FlatFolderDataset


class SiamesePairDataset(Dataset):

    def __init__(self, root_dir):

        self.dataset = FlatFolderDataset(root_dir)

        self.samples = self.dataset.samples

        self.class_indices = {}

        for idx, (_, label) in enumerate(self.samples):

            if label not in self.class_indices:
                self.class_indices[label] = []

            self.class_indices[label].append(idx)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img1, label1 = self.dataset[idx]

        same_class = random.randint(0, 1)

        if same_class:

            pair_idx = idx

            while pair_idx == idx:

                pair_idx = random.choice(
                    self.class_indices[label1]
                )

            img2, _ = self.dataset[pair_idx]

            label = 0

        else:

            other_class = label1

            while other_class == label1:

                other_class = random.choice(
                    list(self.class_indices.keys())
                )

            pair_idx = random.choice(
                self.class_indices[other_class]
            )

            img2, _ = self.dataset[pair_idx]

            label = 1

        return img1, img2, label