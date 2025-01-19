import torch
import torch.optim as optim
import numpy as np
import cv2
from model import Detector
from loss import DetectionLoss
from dataset import data_generator
from utils import process_prediction, draw_bbox
from config import Config


class ScheduledOptimizer():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, lr_mul, min_lr):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.n_steps = 0
        self.min_lr = min_lr


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        lr = max(self.lr_mul * self._optimizer.param_groups[-1]['lr'], self.min_lr)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class DetectionTrainer:
    def __init__(self, path=None):
        self.detector = Detector() 
        self.detection_loss = DetectionLoss()

        if path is not None:
            self.detector.load(path=path)

        self.device = Config.DEVICE
        self.detector.to(self.device)


    def train(self, train_params):
        self.train_settings = train_params
        img, label = next(data_generator(batch_size=train_params.BATCH_SIZE, nObjects=4))
        self.init_optimizer()

        trained_step = 0
        if self.detector.checkpoint is not None:
            self.optimizer.load_state_dict(self.detector.checkpoint['optimizer_state_dict'])
            trained_step = self.detector.checkpoint['step']
        scheduler = ScheduledOptimizer(optimizer=self.optimizer, 
                                       lr_mul=train_params.LR_DECAY_RATE, 
                                       min_lr=train_params.MIN_LR)
        

        self.detector.train()
        for step in range(trained_step, max(train_params.NUM_EPOCHS, train_params.NUM_STEPS)):
            self.optimizer.zero_grad()
            img, label = next(data_generator(batch_size=train_params.BATCH_SIZE, nObjects=4))
            data = img['image']

            label = torch.Tensor(label).to(self.device)
            pred = self.detector.infer(data)
            box_loss, class_loss = self.detection_loss(label=label, prediction=pred)

            total_loss = box_loss + class_loss
            total_loss.mean().backward()
            self.optimizer.step()

            print(f'step: {step}         box_loss: {box_loss.mean()}        class_loss: {class_loss.mean()}       total_loss:  {total_loss.mean()}')

            if step%self.train_settings.SAVE_INTERVAL == 0 and (step-trained_step):
                self.save_log(step, img, label.cpu().numpy())
                self.save_checkpoint(step)

                if step%(self.train_settings.SAVE_INTERVAL * self.train_settings.LR_UPDATE) == 0:
                    scheduler.step_and_update_lr()
                    print(f"learning_rate: {self.optimizer.param_groups[-1]['lr']}")


    def init_optimizer(self):
        if self.train_settings.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(self.detector.parameters(), lr=self.train_settings.LEARNING_RATE)
        else:
            self.optimizer = optim.SGD(self.detector.parameters(), lr=self.train_settings.LEARNING_RATE)


    def save_log(self, step, data, label):
        # write info to log and save images
        img_stack = []
        lab_stack = []
        bsize = 4
        data, label = next(data_generator(batch_size=bsize, nObjects=5))

        with torch.no_grad():
            batch_pred = self.detector.infer(data['image'])
            for i in range(bsize):
                lab = label[i]
                label_box = np.hstack([np.argmax(lab[:, 4:], axis=1).reshape(-1, 1), lab[:, :4]])
                print(label_box)
                pred = process_prediction(batch_pred[i], confidence=0.25)
                if len(pred) > 0:
                    box_img = draw_bbox(data['image'][i], pred[:10])
                    box_img_ = draw_bbox(data['image'][i], label[i], pred=False)
                    img_stack.append(box_img)
                    lab_stack.append(box_img_)
        
        if len(img_stack):
            cv2.imwrite(f'{self.train_settings.image_write_path}/{step}_output.png', np.hstack(img_stack))
            cv2.imwrite(f'{self.train_settings.image_write_path}/{step}_input.png', np.hstack(lab_stack))


    def save_checkpoint(self, step):
        self.detector.save(step=step, path=f'{self.train_settings.checkpoint_write_path}/checkpoint_{step}.pt', optim=self.optimizer)
        # torch.save(self.detector, f'{self.train_settings.checkpoint_write_path}/checkpointModel_{step}.pt')
