import os
import random

import torch

from PIL import Image

from torchvision import transforms

from config import IMG_SIZE


transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


class EpisodicSampler:

    def __init__(
        self,
        root_dir
    ):

        self.root_dir = root_dir

        self.classes = sorted(
            os.listdir(root_dir)
        )

        self.class_images = {}

        for cls in self.classes:

            cls_dir = os.path.join(
                root_dir,
                cls
            )

            images = [

                os.path.join(
                    cls_dir,
                    f
                )

                for f in os.listdir(
                    cls_dir
                )

            ]

            self.class_images[
                cls
            ] = images

    def sample_episode(
        self,
        n_way=3,
        k_shot=1,
        q_query=5
    ):

        episode_classes = random.sample(
            self.classes,
            n_way
        )

        support_set = []

        query_set = []

        label_map = {

            cls:i

            for i,cls in enumerate(
                episode_classes
            )
        }

        for cls in episode_classes:

            images = random.sample(

                self.class_images[cls],

                k_shot + q_query

            )

            support_imgs = images[:k_shot]

            query_imgs = images[k_shot:]

            for img in support_imgs:

                support_set.append(

                    (
                        img,
                        label_map[cls]
                    )

                )

            for img in query_imgs:

                query_set.append(

                    (
                        img,
                        label_map[cls]
                    )

                )

        return (
            support_set,
            query_set
        )



def load_image(
    img_path
):

    img = Image.open(
        img_path
    ).convert("RGB")

    img = transform(img)

    return img