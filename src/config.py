import torch

##NEU
TRAIN_DIR = "../datasets/NEU/train_flat"
TEST_DIR = "../datasets/NEU/test_flat"

##CLUST
# TRAIN_DIR = "../datasets/Clust/train_flat"
# TEST_DIR = "../datasets/Clust/test_flat"

# #SPLIT
# TRAIN_DIR = "../datasets/Split/train_flat"
# TEST_DIR = "../datasets/Split/test_flat"

IMG_SIZE = 200 #224

EMBED_DIM = 512#128

BATCH_SIZE = 32

EPOCHS = 50 #20,50

LR = 1e-3

MARGIN = 1.0 # 1.0, 1.5, 2.0

N_WAY = 3

QUERY = 5

SHOTS = [1, 3, 5]

VAL_RATIO = 0.2

SEED = 42

TRAIN_EPISODES = 100#1000

VAL_EPISODES = 50#200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "../checkpoints/siamese"

# TRAIN_EPISODES = 100
# VAL_EPISODES = 50
# EPOCHS = 30