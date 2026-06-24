import torch
import numpy as np

from tqdm import tqdm

from config import *

from prototypical import ProtoNet

from episodic_sampler import (
    EpisodicSampler,
    load_image
)

## Load Model

device = DEVICE

model = ProtoNet()

model.load_state_dict(

    torch.load(
        "../checkpoints/proto/best_proto.pth",
        map_location=device
    )

)

model = model.to(device)

model.eval()

## Episode Evaluation Function

@torch.no_grad()
def evaluate_episode(
    support_set,
    query_set
):

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

        emb = model(img)

        dists = []

        for label in labels:

            dist = torch.norm(
                emb.squeeze(0)
                -
                prototypes[label]
            )

            dists.append(
                dist.item()
            )

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

## Few-shot Evaluation Loop

def evaluate_fewshot(
    shot
):

    sampler = EpisodicSampler(
        TEST_DIR
    )

    accs = []

    for _ in tqdm(
        range(500)
    ):

        support_set,query_set = \
            sampler.sample_episode(
                n_way=3,
                k_shot=shot,
                q_query=5
            )

        acc = evaluate_episode(
            support_set,
            query_set
        )

        accs.append(
            acc
        )

    return (
        np.mean(accs),
        np.std(accs)
    )
    
## Run

for shot in [1,3,5]:

    mean_acc,std_acc = \
        evaluate_fewshot(
            shot
        )

    print(
        f"3-Way {shot}-Shot : "
        f"{mean_acc*100:.2f}"
        f" ± "
        f"{std_acc*100:.2f}"
    )