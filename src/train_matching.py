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

## Load Model

device = DEVICE

model = MatchingNetwork().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

## Training Episode Function

def train_episode(
    support_set,
    query_set
):

    model.train()

    optimizer.zero_grad()

    support_embs = []
    support_labels = []

    for img_path,label in support_set:

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

    losses = []

    for img_path,true_label in query_set:

        img = load_image(
            img_path
        )

        img = img.unsqueeze(0).to(device)

        query_emb = model(img)
        
        query_emb = F.normalize(
            query_emb,
            p=2,
            dim=1
        )

        support_embs = F.normalize(
            support_embs,
            p=2,
            dim=1
        )

        similarities = F.cosine_similarity(
            query_emb,
            support_embs,
            dim=1
        )

        attention = torch.softmax(
            similarities,
            dim=0
        )

        n_classes = len(
            set(
                support_labels
            )
        )

        probs = torch.zeros(
            n_classes
        ).to(device)

        for att,label in zip(
            attention,
            support_labels):

            probs[label] += att

        probs = probs.unsqueeze(0)

        target = torch.tensor([true_label]).to(device)

        loss = F.nll_loss(torch.log(probs + 1e-8), target)

        losses.append(loss)

    loss = torch.stack(losses).mean()

    loss.backward()

    optimizer.step()

    return loss.item()

## Validation Episode Function

@torch.no_grad()
def validate_episode(
    support_set,
    query_set
):

    model.eval()

    support_embs = []
    support_labels = []

    for img_path,label in support_set:

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

    correct = 0
    total = 0

    for img_path,true_label in query_set:

        img = load_image(
            img_path
        )

        img = img.unsqueeze(0).to(device)
        
        query_emb = model(img)

        similarities = F.cosine_similarity(
            query_emb,
            support_embs,
            dim=1
        )

        attention = torch.softmax(
            similarities,
            dim=0
        )

        n_classes = len(
            set(
                support_labels
            )
        )

        probs = torch.zeros(
            n_classes
        ).to(device)

        for att,label in zip(
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

# Sample Episode Function

train_sampler = EpisodicSampler(
    TRAIN_DIR
)

val_sampler = EpisodicSampler(
    TEST_DIR# VAL_DIR
)

## Train Loop

best_acc = 0

for epoch in range(EPOCHS):

    train_losses = []

    for _ in range(TRAIN_EPISODES):

        shot = random.choice(
            [1,3,5]
        )

        support_set,query_set = \
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

    for _ in range(VAL_EPISODES):

        shot = random.choice(
            [1,3,5]
        )

        support_set,query_set = \
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
    ) / len(train_losses)

    mean_acc = sum(
        val_accs
    ) / len(val_accs)

    print(

        f"Epoch {epoch+1}"

        f" Loss={mean_loss:.4f}"

        f" ValAcc={mean_acc:.4f}"

    )

    if mean_acc > best_acc:

        best_acc = mean_acc

        torch.save(

            model.state_dict(),

            "../checkpoints/matching/best_matching.pth"

        )