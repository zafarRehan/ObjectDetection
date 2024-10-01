import fnmatch
import os
from PIL import Image


# dictionary of animals with scaled bounding boxes 0 to 1
def get_animals():
    animals = {
        0: {'name': 'octopus',
        'file': 'octopus.png',
        'boxes': [0.09722222, 0.90277778, 0.125     , 0.81944444]},
        1: {'name': 'cat',
        'file': 'cat.png',
        'boxes': [0.04166667, 0.94444444, 0.19444444, 0.79166667]},
        2: {'name': 'cow',
        'file': 'cow.png',
        'boxes': [0.05555556, 0.91666667, 0.26388889, 0.80555556]},
        3: {'name': 'tiger',
        'file': 'tiger.png',
        'boxes': [0.04166667, 0.94444444, 0.30555556, 0.79166667]},
        4: {'name': 'cock',
        'file': 'cock.png',
        'boxes': [0.13888889, 0.84722222, 0.13888889, 0.86111111]},
        5: {'name': 'turtle',
        'file': 'turtle.png',
        'boxes': [0.05555556, 0.93055556, 0.19444444, 0.79166667]},
        6: {'name': 'monkey',
        'file': 'monkey.png',
        'boxes': [0.08333333, 0.91666667, 0.11111111, 0.875     ]},
        7: {'name': 'rat',
        'file': 'rat.png',
        'boxes': [0.05555556, 0.94444444, 0.27777778, 0.875     ]},
        8: {'name': 'elephant',
        'file': 'elephant.png',
        'boxes': [0.04166667, 0.93055556, 0.20833333, 0.81944444]},
        9: {'name': 'goat',
        'file': 'goat.png',
        'boxes': [0.05555556, 0.90277778, 0.15277778, 0.88888889]},
        10: {'name': 'dog',
        'file': 'dog.png',
        'boxes': [0.05555556, 0.88888889, 0.15277778, 0.875     ]},
        11: {'name': 'rabbit',
        'file': 'rabbit.png',
        'boxes': [0.09722222, 0.88888889, 0.125     , 0.80555556]},
        12: {'name': 'cheetah',
        'file': 'cheetah.png',
        'boxes': [0.        , 0.97222222, 0.27777778, 0.79166667]},
        13: {'name': 'fish',
        'file': 'fish.png',
        'boxes': [0.15277778, 0.90277778, 0.20833333, 0.84722222]},
        14: {'name': 'penguin',
        'file': 'penguin.png',
        'boxes': [0.16666667, 0.83333333, 0.05555556, 0.88888889]}
    }
    
    # storing image with the animals dictionary for easy image access
    for class_id, values in animals.items():
        png_file = Image.open(os.path.join('dataset/animals', values['file'])).convert('RGBA')
        animals[class_id]['image'] = png_file
    
    return animals



def get_backgrounds():
    backgrounds = {}
    filenames = fnmatch.filter(os.listdir('dataset/backgrounds'), '*.jpg')

    for idx, filename in enumerate(filenames):
        temp = {}  
        temp['name'] = filename.split('.')[0]
        temp['file'] = filename 
        backgrounds[idx] = temp
    
    # storing image with the backgrounds dictionary for easy image access
    for img_id, values in backgrounds.items():
        image = Image.open(os.path.join('dataset', 'backgrounds', values['file'])).convert('RGB')
        backgrounds[img_id]['image'] = image

    return backgrounds