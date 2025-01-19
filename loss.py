import torch 
from utils import create_anchors, multibox_target
from config import DatasetConfig, Config, TrainingConfig
from torchvision.ops import complete_box_iou_loss

class DetectionLoss:
    
    def __init__(self):
        self.box_criterion = torch.nn.MSELoss(reduction='none')
        self.class_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.n_classes = DatasetConfig.N_CLASSES + 1
        self.anchors = create_anchors(
                    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                    input_shapes = [[3, 20, 20], [3, 10, 10], [3, 5, 5], [3, 3, 3], [3, 1, 1]])
        self.anchors =  self.anchors.to(Config.DEVICE).detach()
        

    def bbox_loss(self, label, predicted_box, b_size=16):
        # predicted_box.reshape(b_size, -1)
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors=self.anchors, labels=label)
        loss = self.box_criterion(bbox_labels[bbox_masks > 0], predicted_box[bbox_masks > 0]).mean(dim=0)
        return loss, cls_labels

    def bbox_loss_ciou(self, label, predicted_box):
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors=self.anchors, labels=label)
        bbox_labels = bbox_labels[bbox_masks > 0].view([-1, 4])
        predicted_box = predicted_box[bbox_masks > 0].view([-1, 4])
        loss = complete_box_iou_loss(predicted_box, bbox_labels, reduction='none')
        return loss.mean(), cls_labels


    def class_loss(self, cls_labels, prediction_class, b_size):
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
        loss = self.class_criterion(prediction_class.reshape(-1, self.n_classes), 
                                    torch.argmax(cls_labels, dim=2).reshape(-1)).reshape(b_size, -1).mean()
        return loss

    def calculate_loss(self, label, prediction):
        # extract the predicted and ground truth boxes and classes
        BATCH_SIZE = prediction.shape[0]
        predicted_box = prediction[:, :, :4].reshape(BATCH_SIZE, -1)
        predicted_class = prediction[:, :, 4:]
        # bbox_loss, cls_labels = self.bbox_loss_ciou(label, predicted_box)
        bbox_loss, cls_labels = self.bbox_loss(label, predicted_box)
        class_loss = self.class_loss(cls_labels, predicted_class, BATCH_SIZE)
        return bbox_loss, class_loss*10
    

    def __call__(self, label, prediction):
        return self.calculate_loss(label, prediction)



        
if __name__ == '__main__':
    print(DetectionLoss.anchors)
    print(DetectionLoss.anchors.shape)
        
