import torch
import torch.optim as optim
import numpy as np
import cv2
from model import Detector
from loss import DetectionLoss
from dataset import data_generator
from utils import process_prediction, draw_bbox

class DetectionTrainer:
    def __init__(self,):
        self.detector = Detector()
        self.detection_loss = DetectionLoss()

    def train(self, train_params):
        self.train_settings = train_params
        img, label = next(data_generator(batch_size=train_params.BATCH_SIZE, nObjects=4))


        optimizer = optim.Adam(self.detector.parameters(), lr=train_params.LEARNING_RATE)
 
        self.detector.train()
        for step in range(max(train_params.NUM_EPOCHS, train_params.NUM_STEPS)):
            self.detector.zero_grad()
            # optimizer.zero_grad()

            # img, label = next(data_generator(batch_size=train_params.BATCH_SIZE, nObjects=4))
            data = img['image']
            pred = self.detector(torch.permute(torch.Tensor(data), (0, 3, 1, 2)))
            box_loss, class_loss = self.detection_loss(label=label, prediction=pred)
            total_loss = box_loss + class_loss
            # print(total_loss, class_loss, box_loss)
            total_loss.backward()
            optimizer.step()

            print(f'step: {step}         box_loss: {box_loss}        class_loss: {class_loss}')

            if step%100 == 0 and step:
                self.save_log(step, img, label)


    def save_log(self, step, data, label):
        # write info to log and save images
        img_stack = []
        lab_stack = []
        bsize = 5

        with torch.no_grad():
            # data, label = next(data_generator(batch_size=bsize, nObjects=5))
            batch_pred = self.detector.infer(data['image'])
            for i in range(bsize):
                pred = process_prediction(batch_pred[i], confidence=0.0)
                if len(pred) > 0:
                    box_img = draw_bbox(data['image'][i], pred[:10])
                    box_img_ = draw_bbox(data['image'][i], label[i], pred=False)
                    img_stack.append(box_img)
                    lab_stack.append(box_img_)
        
        if len(img_stack):
            cv2.imwrite(f'{self.train_settings.image_write_path}/output_{step}.png', np.hstack(img_stack))
            cv2.imwrite(f'{self.train_settings.image_write_path}/input_{step}.png', np.hstack(lab_stack))



    # def save_images(self, real, fake, step):
    #     real = real.permute(1, 2, 0).cpu()
    #     fake = fake.permute(1, 2, 0).cpu()
    #     img = np.hstack([real, fake])
    #     img = img * 255
    #     cv2.imwrite(f'{self.train_settings.image_write_path}/step_{step}.png', img)

