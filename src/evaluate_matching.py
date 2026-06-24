import os
import random
import numpy as np

import torch
import torch.nn.functional as F

from matching import MatchingNetwork
from train_matching import validate_episode

from episodic_sampler import (
    EpisodicSampler,
    load_image
)

from config import *

device = DEVICE

model = MatchingNetwork()

model.load_state_dict(

    torch.load(
        "../checkpoints/matching/best_matching.pth",
        map_location=device
    )

)

model = model.to(device)

model.eval()

## Evaluation Episode Function Few Shot

def evaluate_fewshot(
    shot
):

    sampler = EpisodicSampler(
        TEST_DIR
    )

    accs = []

    for _ in range(500):

        support_set,query_set = \
            sampler.sample_episode(
                n_way=3,
                k_shot=shot,
                q_query=5
            )

        acc = validate_episode(
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
    
## Run Evaluation

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