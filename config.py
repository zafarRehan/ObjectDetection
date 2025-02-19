import torch

class TrainingConfig:
    LEARNING_RATE = 0.0005
    LR_DECAY_RATE = 0.9
    MIN_LR = 0.00005
    LR_UPDATE = 3 # update learning rate after every nth step
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    NUM_STEPS = 20000
    OPTIMIZER = 'adam'
    SAVE_INTERVAL = 100

class DatasetConfig:
    N_CLASSES = 15
    IMAGE_CHANNELS = 3
    IMAGE_SIZE = 320

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXP_BASE_DIR = 'experiments'
