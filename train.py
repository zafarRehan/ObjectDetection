import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import os
import argparse
from trainer import DetectionTrainer
from config import TrainingConfig


class TrainingParameters:
    def __init__(self, args):
        self.LEARNING_RATE = TrainingConfig.LEARNING_RATE  
        self.BATCH_SIZE = TrainingConfig.BATCH_SIZE
        self.IMAGE_SIZE = TrainingConfig.IMAGE_SIZE
        self.CHANNELS_IMG = TrainingConfig.CHANNELS_IMG
        self.NUM_EPOCHS = TrainingConfig.NUM_EPOCHS
        self.NUM_STEPS = TrainingConfig.NUM_STEPS
        self.OPTIMIZER = TrainingConfig.OPTIMIZER
        
        self.writer = SummaryWriter(f"logs/train")
        self.image_write_path = f"{args.exp_name}/images"
        self.checkpoint_write_path = f"{args.exp_name}/checkpoints"
        os.makedirs(self.image_write_path, exist_ok=True)
        os.makedirs(self.checkpoint_write_path, exist_ok=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", help="Experiment Name", default='first_experiment')
    args=parser.parse_args()


    tp = TrainingParameters(args)
    trainer = DetectionTrainer(path=f"{args.exp_name}/checkpoints/checkpoint_3600.pt")
    # trainer = DetectionTrainer()
    trainer.train(train_params=tp)
