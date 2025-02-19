import torch 
import cv2
import numpy as np
from config import DatasetConfig, Config

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

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  


def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    # print(max_ious[max_ious > iou_threshold], indices[max_ious > iou_threshold].shape)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    n_classes = DatasetConfig.N_CLASSES
    batch_size = labels.shape[0]
    anchors = anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    labels = torch.Tensor(labels)
    for i in range(batch_size):
        box_label, cls_label = labels[i, :, :4], labels[i, :, 4:]
        anchors_bbox_map = assign_anchor_to_bbox(
            box_label, anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros((num_anchors, n_classes+1), dtype=torch.float32,
                                   device=device)
        class_labels[:, n_classes] = torch.ones(num_anchors)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        # print(indices_true, bb_idx)
        class_labels[indices_true] = cls_label[bb_idx]
        class_labels[indices_true, n_classes] = 0 # set background class to 0
        assigned_bb[indices_true] = box_label[bb_idx]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = box_corner_to_center(anchors)
    # print(anchors.shape, offset_preds.shape)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def box_corner_to_center(cords):
    new_cords = torch.zeros_like(cords, device=cords.device)
    new_cords[:, 0] = cords[:, 0] + (cords[:, 2] - cords[:, 0]) / 2
    new_cords[:, 1] = cords[:, 1] + (cords[:, 3] - cords[:, 1]) / 2
    new_cords[:, 2] = cords[:, 2] - cords[:, 0]
    new_cords[:, 3] = cords[:, 3] - cords[:, 1]
    return new_cords


def box_center_to_corner(cords):
    new_cords = torch.zeros_like(cords, device=cords.device)
    new_cords[:, 0] = cords[:, 0] - cords[:, 2] / 2
    new_cords[:, 1] = cords[:, 1] - cords[:, 3] / 2
    new_cords[:, 2] = cords[:, 2] + new_cords[:, 0]
    new_cords[:, 3] = cords[:, 3] + new_cords[:, 1]
    return new_cords


def create_anchors(sizes, input_shapes):
    ratios = [0.5, 1, 2]
    output_anchors = multibox_prior(torch.zeros(input_shapes[0]), sizes=sizes[0], ratios=ratios)
    # print(output_anchors.shape)
    for i in range(1,len(sizes)):
        anchors_t = multibox_prior(torch.zeros(input_shapes[i]), sizes=sizes[i], ratios=[1, 2, 0.5])
        output_anchors = torch.concat([output_anchors, anchors_t], axis=1)
        
    return output_anchors 


"""
One of the most important function for post processing the multiple same class
overlapping output boxes. Check comments inside function for details

Explained here:  https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536
"""
def non_max_supression(predictions, conf_threshold, iou_threshold):
    predictions = predictions[predictions[:, 1] > 0.05]
    # sort the predictions in decreasing order of accuracy
    sortedArr = predictions[predictions[:,1].argsort()][::].flip(dims=(0,))

    # if there is no element left after filtering return the empty array
    if sortedArr.shape[0] == 0:
        return  sortedArr

    # create a final array with first element from sorted array already in place
    final_list = sortedArr[0]
    final_list = final_list.reshape(1, final_list.shape[0])
    

    for element in sortedArr[1:]:
        # if same class is already present in target list
        if element[0] in final_list[:, 0]:

            # get the indexes of all present same elements
            # indexes_present = np.where(final_list[:, 0]==element[0])[0]
            indexes_present = final_list[:, 0] == element[0]
            # calclulate iou for all present same class element
            ious = box_iou(element[2:].reshape(1, 4), final_list[indexes_present, 2:])
            # get indexes of all ious above iou_threshold
            ious_indexes = ious[0, :] > iou_threshold
            # if there is no element present which alraedy has an iou of
            # over 0.5 with current element then push it
            if ious_indexes.size == 0:
                final_list = torch.concatenate((final_list, element.reshape(1, -1)), axis=0)
        else:
            # else push the element into the final list
            final_list = torch.concatenate((final_list, element.reshape(1, -1)), axis=0)
    return final_list.cpu().numpy()



anchors_g = create_anchors(
                    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                    input_shapes = [[3, 20, 20], [3, 10, 10], [3, 5, 5], [3, 3, 3], [3, 1, 1]]
                ) 
anchors_g = anchors_g.to(Config.DEVICE).detach()
def process_prediction(prediction, confidence=0.25):
    if prediction.dim() == 2:
        prediction = torch.unsqueeze(prediction, dim=0)
    predicted_cls, predicted_box = prediction[:, :, 4:], prediction[:, :, :4]
    inversed_pred_boxes = offset_inverse(anchors_g[0], predicted_box[0])
    predicted_cls = torch.nn.functional.softmax(predicted_cls[0], dim=1)
    conf, class_ids = torch.max(predicted_cls, dim=1)
    class_prob = torch.cat((class_ids.view(-1, 1), conf.view(-1, 1)), dim=1)
    combined = torch.cat((class_prob, inversed_pred_boxes), dim=1)
    combined = combined[combined[:, 0] < DatasetConfig.N_CLASSES]  # N_CLASSES is the background class 0 to N_CLASSES-1 are actual classes
    detetctions = non_max_supression(combined, conf_threshold=confidence, iou_threshold=0.5)
    return detetctions

def draw_bbox(img, labels, pred=True):
    h,w,_ = img.shape
    img = img[:,:,::-1]*255
    display_str = ''
    for label in labels:
        if pred:
            cls, conf, x1, y1, x2, y2 = label
            display_str = str(int(cls)) + '_' + str(round(conf, 2))
            # print(cls, conf, x1, y1, x2, y2 )
        else:
            x1, y1, x2, y2 = label[:4]
            cls = np.argmax(label[4:])
            display_str = str(int(cls))
        img = cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color=colors[int(cls)], thickness=2)
        img = cv2.putText(img, display_str, org=(int(x1*w), int(y1*h)-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=colors[int(cls)], thickness=2, lineType=cv2.LINE_AA)
    return img
