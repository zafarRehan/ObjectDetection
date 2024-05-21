import torch 
from utils import create_anchors, multibox_target
from config import DatasetConfig, Config

class DetectionLoss:
    anchors = create_anchors(
                    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                    input_shapes = [[3, 20, 20], [3, 10, 10], [3, 5, 5], [3, 3, 3], [3, 1, 1]]
        )
    anchors = anchors.to(Config.DEVICE)
    
    def __init__(self):
        # self.anchors = create_anchors(
        #             sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619]],
        #             input_shapes = [[3, 20, 20], [3, 10, 10], [3, 5, 5]]
        # )
        self.box_criterion = torch.nn.MSELoss(reduction='mean')
        self.class_criterion = torch.nn.CrossEntropyLoss()
        

    def box_loss(self, label, predicted_box):
        # get the ground truth lables dimensions changed to prediction dimensions by using anchors
        # print(label[:, :, :4])
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors=DetectionLoss.anchors, labels=label)
        # bbox_masks = torch.unsqueeze(bbox_masks,dim=1)
        # bbox_labels = torch.unsqueeze(bbox_labels,dim=1)
        # print(f'anchors: {DetectionLoss.anchors.shape}')
        # print(f'bbox_labels: {bbox_labels.shape}')
        # print(f'bbox_masks: {bbox_masks.shape}')
        # print(f'cls_labels: {cls_labels.shape}')
        # print(f'predicted_box: {predicted_box.shape}')
        # print((predicted_box*bbox_masks).shape)

        loss = self.box_criterion(bbox_labels*bbox_masks, predicted_box*bbox_masks)
        # loss = loss/bbox_labels.shape[0]
        # print(predicted_box.shape)

        bbox_labels = bbox_labels[bbox_masks > 0]
        predicted_box = predicted_box[bbox_masks > 0]
        # loss = self.box_criterion(predicted_box, bbox_labels)
        # print(predicted_box.shape)



        # print(torch.abs(predicted_box-bbox_labels)[:10].view(-1, 1))
        # print(predicted_box[:10].view(-1, 1), bbox_labels[:10].view(-1, 1))

        return loss, cls_labels


    def class_loss(self, cls_labels, prediction_class):
        """
        For class loss the classes are not already softmaxed from model so they are passed
        through "softmax_cross_entropy_with_logits" which calculates the softmax inplace and calculates
        loss using it in accurate manner

        For class loss all the 2140 predictions are used for loss calculation because any class which is background
        and not classified as background needs to be punished and vice-versa

        while in the case of box loss only the masked boxes (masked boxes are those anchor boxes which has a probability that
        it contains an object) are considered for loss calculation
        because whatever value the unwanted predicted boxes have doesnt matters here and considering them will
        incur unwanted loss addition in training
        """
        # print(torch.nn.functional.softmax(prediction_class, dim=0).shape, cls_labels.shape)
        # amax = torch.argmax(cls_labels, dim=2)
        # print(amax[0][amax[0] < 15])
        loss = self.class_criterion(prediction_class.view(-1, DatasetConfig.N_CLASSES+1), cls_labels.view(-1, DatasetConfig.N_CLASSES+1))
        return loss

    def calculate_loss(self, label, prediction):
        # extract the predicted and ground truth boxes and classes
        BATCH_SIZE = prediction.shape[0]

        # label_box = label[:, :, :4]
        # label_class = label[:, :, 4:]

        # predicted_box = prediction[:, :, :4].reshape(BATCH_SIZE, -1, 8400)
        predicted_box = prediction[:, :, :4].reshape(BATCH_SIZE, -1)
        # predicted_box = torch.squeeze(predicted_box, dim=1)
        predicted_class = prediction[:, :, 4:]

        bbox_loss, cls_labels = self.box_loss(label, predicted_box)
        # print(predicted_box.shape)
        class_loss = self.class_loss(cls_labels, predicted_class)

        return bbox_loss, class_loss
    

    def __call__(self, label, prediction):
        # print(prediction.shape)
        return self.calculate_loss(label, prediction)



        
if __name__ == '__main__':

    print(DetectionLoss.anchors)
    print(DetectionLoss.anchors.shape)
        
