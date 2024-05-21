import torch
import torch.optim as optim
import numpy as np
import cv2
from model import Detector
from loss import DetectionLoss
from dataset import data_generator
from utils import process_prediction, draw_bbox
from config import Config

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
        self.optimizer = self.init_optimizer()

        trained_step = 0
        if self.detector.checkpoint is not None:
            self.optimizer.load_state_dict(self.detector.checkpoint['optimizer_state_dict'])
            trained_step = self.detector.checkpoint['step']
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=self.optimizer, lr_lambda=lambda step: 0.8)

        

        self.detector.train()
        for step in range(trained_step, max(train_params.NUM_EPOCHS, train_params.NUM_STEPS)):

            img, label = next(data_generator(batch_size=train_params.BATCH_SIZE, nObjects=4))
            data = img['image']
            label = torch.Tensor(label).to(self.device)
            # pred = self.detector(torch.permute(torch.Tensor(data).to(self.device), (0, 3, 1, 2)))
            pred = self.detector.infer(data)
            box_loss, class_loss = self.detection_loss(label=label, prediction=pred)

            self.detector.zero_grad()
            # self.optimizer.zero_grad()
            total_loss = (box_loss + class_loss)
            total_loss.backward()
            self.optimizer.step()

            print(f'step: {step}         box_loss: {box_loss}        class_loss: {class_loss}')

            if step%self.train_settings.SAVE_INTERVAL == 0 and (step-trained_step):
                self.save_log(step, img, label, self.detector)
                self.save_checkpoint(step)

                if step%(self.train_settings.SAVE_INTERVAL*5) == 0:
                    scheduler.step()
                    print(f"learning_rate: {self.optimizer.param_groups[-1]['lr']}")


    def init_optimizer(self):
        if self.train_settings.OPTIMIZER == 'adam':
            optimizer = optim.Adam(self.detector.parameters(), lr=self.train_settings.LEARNING_RATE)
        else:
            optimizer = optim.SGD(self.detector.parameters(), lr=self.train_settings.LEARNING_RATE)
        return optimizer
        


    def save_log(self, step, data, label, model):
        # write info to log and save images
        img_stack = []
        lab_stack = []
        bsize = 4

        with torch.no_grad():
            # data, label = next(data_generator(batch_size=bsize, nObjects=5))
            batch_pred = model.infer(data['image'])
            for i in range(bsize):
                lab = label[i]
                label_box = np.hstack([np.argmax(lab[:, 4:], axis=1).reshape(-1, 1), lab[:, :4]])
                print(label_box)
                pred = process_prediction(batch_pred[i], confidence=0.4)
                if len(pred) > 0:
                    box_img = draw_bbox(data['image'][i], pred[:10])
                    box_img_ = draw_bbox(data['image'][i], label[i], pred=False)
                    img_stack.append(box_img)
                    lab_stack.append(box_img_)
        
        if len(img_stack):
            cv2.imwrite(f'{self.train_settings.image_write_path}/output_{step}.png', np.hstack(img_stack))
            cv2.imwrite(f'{self.train_settings.image_write_path}/input_{step}.png', np.hstack(lab_stack))


    def save_checkpoint(self, step):
        self.detector.save(step=step, path=f'{self.train_settings.checkpoint_write_path}/checkpoint_{step}.pt', optim=self.optimizer)
        # torch.save(self.detector, f'{self.train_settings.checkpoint_write_path}/checkpointModel_{step}.pt')



    # def save_images(self, real, fake, step):
    #     real = real.permute(1, 2, 0).cpu()
    #     fake = fake.permute(1, 2, 0).cpu()
    #     img = np.hstack([real, fake])
    #     img = img * 255
    #     cv2.imwrite(f'{self.train_settings.image_write_path}/step_{step}.png', img)

