import os

import torch

from torch.utils.data import DataLoader

from pair_dataset import SiamesePairDataset

from siamese import (
    SiameseNetwork,
    ContrastiveLoss
)

from config import *

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

dataset = SiamesePairDataset(
    TRAIN_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0#4
)

model = SiameseNetwork().to(DEVICE)

criterion = ContrastiveLoss(
    margin=MARGIN
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_loss = 999999

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for img1, img2, label in loader:

        img1 = img1.to(DEVICE)

        img2 = img2.to(DEVICE)

        label = label.float().to(DEVICE)

        optimizer.zero_grad()

        z1, z2 = model(
            img1,
            img2
        )

        loss = criterion(
            z1,
            z2,
            label
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={epoch_loss:.4f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best_siamese.pth"
            )
        )