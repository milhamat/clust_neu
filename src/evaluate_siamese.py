import torch
import numpy as np

from tqdm import tqdm

from config import *

from siamese import SiameseNetwork

from episodic_sampler import (
    EpisodicSampler,
    load_image
)

## Load Model

device = DEVICE

model = SiameseNetwork()

model.load_state_dict(

    torch.load(

        "../checkpoints/siamese/best_siamese.pth",

        map_location=device

    )

)

model = model.to(device)

model.eval()

## Embedding Function

def get_embedding(
    img_tensor
):

    with torch.no_grad():

        emb = model.encoder(

            img_tensor.unsqueeze(0).to(device)

        )

    return emb.squeeze(0)

## Evaluate Episode

def evaluate_episode(

    support_set,

    query_set

):

    prototypes = {}

    correct = 0

    total = 0

    # -------------------
    # Build Prototype
    # -------------------

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

    for label in labels:

        embs = []

        for img_path,y in support_set:

            if y != label:
                continue

            img = load_image(
                img_path
            )

            emb = get_embedding(
                img
            )

            embs.append(
                emb
            )

        prototypes[label] = torch.stack(
            embs
        ).mean(0)

    # -------------------
    # Query Prediction
    # -------------------

    for img_path,true_label in query_set:

        img = load_image(
            img_path
        )

        emb = get_embedding(
            img
        )

        distances = []

        for label in labels:

            dist = torch.norm(

                emb -

                prototypes[label]

            )

            distances.append(
                dist.item()
            )

        pred = labels[
            np.argmin(
                distances
            )
        ]

        correct += (

            pred == true_label

        )

        total += 1

    acc = correct / total

    return acc

## Evaluate Multiple Episodes

def evaluate_fewshot(

    root_dir,

    n_way,

    k_shot,

    q_query=5,

    episodes=500

):

    sampler = EpisodicSampler(
        root_dir
    )

    accuracies = []

    for _ in tqdm(range(episodes)):

        support_set, query_set = sampler.sample_episode(
                n_way=n_way,
                k_shot=k_shot,
                q_query=q_query
            )

        acc = evaluate_episode(
            support_set,
            query_set
        )

        accuracies.append(
            acc
        )

    mean_acc = np.mean(
        accuracies
    )

    std_acc = np.std(
        accuracies
    )

    return mean_acc,std_acc

## Run Evaluation

print("\n")

for shot in [1,3,5]:

    mean_acc,std_acc = evaluate_fewshot(
            TEST_DIR,
            n_way=3,
            k_shot=shot,
            q_query=5,
            episodes=500
        )

    print(
        f"3-Way {shot}-Shot : "
        f"{mean_acc*100:.2f}"
        f" ± "
        f"{std_acc*100:.2f}"
    )