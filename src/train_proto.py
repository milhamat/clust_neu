import torch
import torch.nn as nn
import torch.nn.functional as F
from prototypical import ProtoNet
from backbone import ResNet18Embedding

from config import (
    TRAIN_DIR,
    TEST_DIR,
    LR,
    EPOCHS,
    DEVICE,
    TRAIN_EPISODES, 
    VAL_EPISODES 
)


from episodic_sampler import (
    EpisodicSampler,
    load_image
)


class ProtoNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ResNet18Embedding()

    def forward(self, x):

        return self.encoder(x)
    

## Load Model
device = DEVICE

model = ProtoNet().to(device)

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

    labels = sorted(
        list(
            set(
                [
                    y
                    for _,y
                    in support_set
                ]
            )
        )
    )

    prototypes = {}

    for label in labels:

        embs = []

        for img_path,y in support_set:

            if y != label:
                continue

            img = load_image(
                img_path
            )

            img = img.unsqueeze(0).to(device)

            emb = model(img)
            
            emb = F.normalize(
                emb,
                p=2,
                dim=1
            )

            embs.append(
                emb.squeeze(0)
            )

        prototypes[label] = torch.stack(
            embs
        ).mean(0)

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

        dists = []

        for label in labels:

            proto = prototypes[label]
            #Distance between query embedding and prototype
            dist = torch.norm(query_emb.squeeze(0) - proto)

            dists.append(-dist)

        # logits = torch.stack(
        #     dists
        # ).unsqueeze(0)
        
        temperature = 0.5

        logits = (
            torch.stack(dists)
            / temperature
        ).unsqueeze(0)

        target = torch.tensor(
            [true_label]
        ).to(device)

        loss = F.cross_entropy(
            logits,
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

## Vlaidation Episode

@torch.no_grad()
def validate_episode(
    support_set,
    query_set
):

    model.eval()

    labels = sorted(
        list(
            set(
                [
                    y
                    for _,y
                    in support_set
                ]
            )
        )
    )

    prototypes = {}

    for label in labels:

        embs = []

        for img_path,y in support_set:

            if y != label:
                continue

            img = load_image(
                img_path
            )

            img = img.unsqueeze(0).to(device)

            emb = model(img)

            embs.append(
                emb.squeeze(0)
            )

        prototypes[label] = torch.stack(
            embs
        ).mean(0)

    correct = 0
    total = 0

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

        dists = []

        for label in labels:

            proto = prototypes[label]
            #Distance between query embedding and prototype
            dist = torch.norm(query_emb.squeeze(0) - proto)

            dists.append(dist.item())

        pred = labels[
            dists.index(
                min(dists)
            )
        ]

        correct += (
            pred == true_label
        )

        total += 1

    return correct / total

## Create Sampler

train_sampler = EpisodicSampler(
    TRAIN_DIR
)

val_sampler = EpisodicSampler(
    TEST_DIR
    # TRAIN_DIR
)

## Training Loop

best_acc = 0

for epoch in range(EPOCHS):

    train_losses = []

    for _ in range(TRAIN_EPISODES):

        support_set,query_set = \
            train_sampler.sample_episode(
                n_way=3,
                k_shot=3,
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

        support_set,query_set = \
            val_sampler.sample_episode(
                n_way=3,
                k_shot=3,
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
            "../checkpoints/proto/best_proto.pth"
        )
        
