import os
import random

import torch
import torch.nn.functional as F

from matching import MatchingNetwork

from episodic_sampler import (
    EpisodicSampler,
    load_image
)

from config import *

# ==========================================
# MODEL
# ==========================================

device = DEVICE

model = MatchingNetwork().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

TEMPERATURE = 0.1

# ==========================================
# TRAIN
# ==========================================

def train_episode(
    support_set,
    query_set
):

    model.train()

    optimizer.zero_grad()

    support_embs = []
    support_labels = []

    # ----------------------------------
    # SUPPORT EMBEDDING
    # ----------------------------------

    for img_path, label in support_set:

        img = load_image(
            img_path
        )

        img = img.unsqueeze(0).to(device)

        emb = model(img)

        support_embs.append(
            emb.squeeze(0)
        )

        support_labels.append(
            label
        )

    support_embs = torch.stack(
        support_embs
    )

    # FCE SUPPORT
    support_embs = model.contextualize(
        support_embs
    )

    support_embs = F.normalize(
        support_embs,
        p=2,
        dim=1
    )

    # ----------------------------------
    # QUERY LOOP
    # ----------------------------------

    losses = []

    for img_path, true_label in query_set:

        img = load_image(
            img_path
        )

        img = img.unsqueeze(0).to(device)

        query_emb = model(img)

        # FCE QUERY
        query_emb = model.contextualize_query(
            query_emb
        )

        query_emb = F.normalize(
            query_emb,
            p=2,
            dim=1
        )

        similarities = F.cosine_similarity(
            query_emb,
            support_embs,
            dim=1
        )

        attention = torch.softmax(
            similarities / TEMPERATURE,
            dim=0
        )

        n_classes = len(
            set(
                support_labels
            )
        )

        probs = torch.zeros(
            n_classes,
            device=device
        )

        for att, label in zip(
            attention,
            support_labels
        ):

            probs[label] += att

        probs = probs.unsqueeze(0)

        target = torch.tensor(
            [true_label],
            device=device
        )

        loss = F.nll_loss(
            torch.log(
                probs + 1e-8
            ),
            target
        )

        losses.append(
            loss
        )

    loss = torch.stack(
        losses
    ).mean()

    loss.backward()

    optimizer.step()

    return loss.item()

# ==========================================
# VALIDATION
# ==========================================

@torch.no_grad()
def validate_episode(
    support_set,
    query_set
):

    model.eval()

    support_embs = []
    support_labels = []

    # ----------------------------------
    # SUPPORT EMBEDDING
    # ----------------------------------

    for img_path, label in support_set:

        img = load_image(
            img_path
        )

        img = img.unsqueeze(0).to(device)

        emb = model(img)

        support_embs.append(
            emb.squeeze(0)
        )

        support_labels.append(
            label
        )

    support_embs = torch.stack(
        support_embs
    )

    support_embs = model.contextualize(
        support_embs
    )

    support_embs = F.normalize(
        support_embs,
        p=2,
        dim=1
    )

    correct = 0
    total = 0

    # ----------------------------------
    # QUERY LOOP
    # ----------------------------------

    for img_path, true_label in query_set:

        img = load_image(
            img_path
        )

        img = img.unsqueeze(0).to(device)

        query_emb = model(img)

        query_emb = model.contextualize_query(
            query_emb
        )

        query_emb = F.normalize(
            query_emb,
            p=2,
            dim=1
        )

        similarities = F.cosine_similarity(
            query_emb,
            support_embs,
            dim=1
        )

        attention = torch.softmax(
            similarities / TEMPERATURE,
            dim=0
        )

        n_classes = len(
            set(
                support_labels
            )
        )

        probs = torch.zeros(
            n_classes,
            device=device
        )

        for att, label in zip(
            attention,
            support_labels
        ):

            probs[label] += att

        pred = torch.argmax(
            probs
        ).item()

        correct += (
            pred == true_label
        )

        total += 1

    return correct / total

# ==========================================
# SAMPLER
# ==========================================

train_sampler = EpisodicSampler(
    TRAIN_DIR
)

val_sampler = EpisodicSampler(
    TEST_DIR
)

# ==========================================
# TRAIN LOOP
# ==========================================

best_acc = 0

for epoch in range(
    EPOCHS
):

    train_losses = []

    for _ in range(
        TRAIN_EPISODES
    ):

        shot = random.choice(
            [1, 3, 5]
        )

        support_set, query_set = \
            train_sampler.sample_episode(
                n_way=3,
                k_shot=shot,
                q_query=5
            )

        loss = train_episode(
            support_set,
            query_set
        )

        train_losses.append(
            loss
        )

    val_accs = []

    for _ in range(
        VAL_EPISODES
    ):

        shot = random.choice(
            [1, 3, 5]
        )

        support_set, query_set = \
            val_sampler.sample_episode(
                n_way=3,
                k_shot=shot,
                q_query=5
            )

        acc = validate_episode(
            support_set,
            query_set
        )

        val_accs.append(
            acc
        )

    mean_loss = sum(
        train_losses
    ) / len(
        train_losses
    )

    mean_acc = sum(
        val_accs
    ) / len(
        val_accs
    )

    print(
        f"Epoch {epoch+1} "
        f"Loss={mean_loss:.4f} "
        f"ValAcc={mean_acc:.4f}"
    )

    if mean_acc > best_acc:

        best_acc = mean_acc

        os.makedirs(
            "../checkpoints/matching",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            "../checkpoints/matching/best_matching.pth"
        )

        print(
            f"Best Model Saved "
            f"Acc={best_acc:.4f}"
        )