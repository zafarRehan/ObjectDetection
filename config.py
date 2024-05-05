import torch


class TrainingConfig:
    LEARNING_RATE = 0.001  
    BATCH_SIZE = 8
    IMAGE_SIZE = 320
    CHANNELS_IMG = 3
    NUM_EPOCHS = 5
    NUM_STEPS = 1000
    OPTIMIZER = 'adam'


class DatasetConfig:
    N_CLASSES = 15


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
