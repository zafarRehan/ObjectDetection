from metadata import *
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from utils import is_notebook, process_prediction, draw_bbox
import numpy as np
from model import Detector

animals = get_animals()
backgrounds = get_backgrounds()
# The total number of classes we have
N_CLASSES = len(animals) + 1 # one extra for background

def check_dataset():
    plt.figure(figsize=(N_CLASSES - 1, 9))

    for i, (j, e) in enumerate(animals.items()):
        plt.subplot(3, 5, i + 1)
        image = cv2.imread(os.path.join('dataset' ,'animals', e['file']))
        image = Image.fromarray(image[:,:,::-1])
        draw = ImageDraw.Draw(image)
        anml = animals[i]
        draw.rectangle((anml['boxes'][0] * 72, anml['boxes'][2]*72, anml['boxes'][1]*72, anml['boxes'][3]*72), outline='black', width=1)
        plt.imshow(image)
        plt.xlabel(e['name'])
        plt.xticks([])
        plt.yticks([])

    if is_notebook():
        plt.show()
    else:
        plt.savefig('data.png') # all objects with their bounding boxes visualized


# generate random cordinates with random size given a class id
def get_random_cords(class_id):
    size = np.random.randint(50, 160)
    animal_image = animals[class_id]['image'].resize((size, size), Image.LANCZOS)
    row = np.random.randint(0, 320-size)
    col = np.random.randint(0, 320-size)

    xmin = col+(animals[class_id]['boxes'][0]*size)
    xmax = col+(animals[class_id]['boxes'][1]*size)
    ymin = row+(animals[class_id]['boxes'][2]*size)
    ymax = row+(animals[class_id]['boxes'][3]*size)
    return [xmin/320, ymin/320, xmax/320, ymax/320], row, col, animal_image


def intersection_over_smaller(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA), 0) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    
    # compute the area of both the A and B rectaangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    smallArea = min(boxAArea, boxBArea)
    iou = interArea / smallArea
    return iou

"""
Compute if a new object in an Image is overlapping with alraedy existing object's boxes.
"""
def overlap(box, boxes):
    for box_ in boxes:
        if intersection_over_smaller(box[0], box_) > 0.5:
            return True
    # print('overlap')
    return False


"""
Creates an image with n number of objects in the image
and returns the generated image along with one hot encoded class array and bbox array
"""
def create_example(n=4):
    # image = Image.new("RGB", (320, 320), (255, 255, 255))
    image = backgrounds[np.random.randint(0, 25)]['image'].copy()
    # n_objects = np.random.randint(1, 5)
    n_objects = n
    class_ids = []
    cords = []
    
    for _ in range(n_objects):
        
        class_id = np.random.randint(0, N_CLASSES-1)
        new_cord, row, col, animal_image = get_random_cords(class_id)
        
        # print(animals[class_id]['name'])
        if(len(cords) > 0):
            pasted = False
            while not pasted:
                if not overlap(np.array(new_cord).reshape(1, 4), np.array(cords)):
                    pasted = True
                else:
                    new_cord, row, col, animal_image = get_random_cords(class_id)

        image.paste(animal_image, (col, row), mask=animal_image.split()[3])
        
        one_hot_class = [0] * N_CLASSES
        one_hot_class[class_id] = 1
        class_ids.append(one_hot_class)
        
        cords.append(new_cord)
    return image, class_ids, cords


def plot_bounding_box(image, gt_bbox, norm=False):
    if norm:
        image *= 255
        image = image.astype('uint8')
    try:
        draw = ImageDraw.Draw(image)
    except:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        
    for i in range(len(gt_bbox)):
        xmin, ymin, xmax, ymax = gt_bbox[i]
        xmin *= 320
        xmax *= 320
        ymin *= 320
        ymax *= 320
        draw.rectangle((xmin, ymin, xmax, ymax), outline='blue', width=1)
    return image


"""
Generates data of given batch size and number of objects
This is the generator function used for training the model with
synthetic data
"""
def data_generator(batch_size=16, nObjects=4):
    while True:
        x_batch = np.zeros((batch_size, 320, 320, 3))
        
        bbox_batch = np.zeros((batch_size, nObjects, 4))
        y_batch = np.zeros((batch_size, nObjects, 16))

        for i in range(0, batch_size):
            image, class_ids, bboxes = create_example(nObjects)
            
            x_batch[i] = np.array(image)/255.
            y_batch[i, :len(class_ids)] = np.array(class_ids)
            # print(np.array(bboxes).shape)
            bbox_batch[i, :len(class_ids)] = np.array(bboxes)
            labels = np.concatenate((bbox_batch, y_batch), axis=2)
        yield {'image': x_batch}, labels




if __name__ == '__main__':
    model = Detector()
    example, label = next(data_generator(batch_size=5, nObjects=5))
    batch_pred = model.infer(example['image'])
    for i in range(5):
        pred = process_prediction(batch_pred[i], confidence=0.0)
        if len(pred) > 0:
            box_img = draw_bbox(example['image'][i], pred[:10])
            cv2.imwrite(f'out_{i}.jpg', box_img)


    # pred = model.infer(example['image'])
    # pred = process_prediction(pred)
    # if len(pred > 0):
    #     box_img = draw_bbox(example['image'], pred)
    # print(pred)
    # x = example['image']
    # print('image_shape: ', x.shape) # image of shape 320 X 320 X 3
    # print('label_shape: ', label.shape) # 16 classes and 4 bounding boxes