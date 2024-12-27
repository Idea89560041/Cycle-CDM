import torch

#image
IMG_SIZE = 96
MOTION_PATH = './data/training_data/heavy/medium/minior/noise'
GOOD_PATH = './data/training_data/heavy/medium/minior/good'

#diffusion
TIMESTEPS = 1000
RELEASE_TIME = 100

#hyperparams
DIM = 32
LR = 1e-5
BATCH_SIZE = 64
BETA = [0.5, 0.999]

#training strategy
EPOCHS = 1000
LOAD_FROM_CHECKPOINT = False
CHECKPOINT_DIR = "./checkpoint/"

#other
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
