import numpy as np
import torch
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random

def _is_pil_image(img):
    return isinstance(img, Image.Image)



def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


    
class RandomHorizontalFlip(object):
    
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        
    def __call__(self, sample):

        if not _is_pil_image(sample):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(sample)))
        if not _is_pil_image(sample):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(sample)))

        if self.probability < 0.5:
            print('flipped')
            sample = sample.transpose(Image.FLIP_LEFT_RIGHT)
            #depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            print('not flipped')

        return sample
    
    
class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}
