from dataset import data_generator
from utils import process_prediction, offset_inverse
from utils import create_anchors, multibox_target
from model import Detector
import torch
import cv2

colors = [
    (0, 0, 255),  # Blue
    (0, 255, 0),  # Green
    (255, 0, 0),  # Red
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 128, 128),  # Gray
    (128, 0, 0),  # Maroon
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Navy Blue
    (128, 128, 0),  # Olive Green
    (0, 128, 128),  # Teal
    (128, 0, 128),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255)  # Sky Blue
]

def draw_bbox(img, labels, pred=True):
    h,w,_ = img.shape
    img = img[:,:,::-1]*255
    for label in labels:
        cls, x1, y1, x2, y2 = label
        img = cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color=colors[int(cls)], thickness=1)
    cv2.imwrite('anchors2.png', img)


if __name__ == '__main__':

    model = Detector().cuda()
    anchors = create_anchors(
                    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                    input_shapes = [[3, 20, 20], [3, 10, 10], [3, 5, 5], [3, 3, 3], [3, 1, 1]]
        )
    
    img, label = next(data_generator(batch_size=1, nObjects=6))
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
    n_anchors = cls_labels.shape[0]

    inversed_pred_boxes = offset_inverse(anchors, torch.zeros((n_anchors, 4)))
    cls_id = cls_labels.argmax(dim=1)
    combined = torch.cat((cls_id.view(-1, 1), inversed_pred_boxes), dim=1)
    print(combined)
    draw_bbox(img['image'][0], combined, pred=True)




