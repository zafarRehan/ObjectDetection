import torch


class TrainingConfig:
    LEARNING_RATE = 0.0001  
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    NUM_STEPS = 10000
    OPTIMIZER = 'adam'


class DatasetConfig:
    N_CLASSES = 15
    IMAGE_CHANNELS = 3
    IMAGE_SIZE = 320


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
