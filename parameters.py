import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 555

# Training Parameters
EPOCH = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001