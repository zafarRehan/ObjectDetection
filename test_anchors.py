from dataset import data_generator
from utils import process_prediction, draw_bbox, offset_inverse
from utils import create_anchors, multibox_target
from model import Detector
import torch


if __name__ == '__main__':

    model = Detector()
    model.load(path='/home/rehan/projects/pytorch/ObjectDetection/first_experiment/checkpoints/checkpoint_1800.pt')

    anchors = create_anchors(
                    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                    input_shapes = [[3, 20, 20], [3, 10, 10], [3, 5, 5], [3, 3, 3], [3, 1, 1]]
        )
    
    anchors2 = create_anchors(
                    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                    input_shapes = [[3, 20, 20], [3, 10, 10], [3, 5, 5], [3, 3, 3], [3, 1, 1]]
        )
    
    img, label = next(data_generator(batch_size=1, nObjects=4))
    bbox_labels, bbox_masks, cls_labels = multibox_target(anchors=anchors, labels=label)
    cls_masks = bbox_masks[:, 0::4]

    bbox_labels = bbox_labels[bbox_masks > 0]
    cls_labels = cls_labels[cls_masks > 0]
    anchors = anchors[cls_masks > 0]

    print(bbox_labels)
    print('classes: \n', label[0, :, 4:])
    print('boxes: \n', label[0, :, :4])
    print(cls_labels)
    print(anchors.shape, bbox_labels.reshape(-1, 4).shape)

    inversed_pred_boxes = offset_inverse(anchors, bbox_labels.reshape(-1, 4))
    print(inversed_pred_boxes)



    # inference
    pred = model.infer(img['image'])
    pred_class, pred_bbox = pred[:, :, 4:], pred[:, :, :4]
    print(pred_class.shape, pred_bbox.shape)

    pred_class = pred_class[cls_masks > 0]
    pred_bbox = pred_bbox[cls_masks > 0]
    print(pred_class.shape, pred_bbox.shape)

    inversed_pred_boxes = offset_inverse(anchors, pred_bbox)
    print(inversed_pred_boxes.shape)
    print(inversed_pred_boxes)
    print(pred_class.shape)
    softmaxed_classes = torch.softmax(pred_class, dim=1)
    print(torch.sum(softmaxed_classes, dim=1))







