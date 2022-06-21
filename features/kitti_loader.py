import os

from torch.utils.data import DataLoader


import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose

resolution_dict = {
    'full' : (384, 1280),
    'tu_small' : (128, 416),
    'tu_big' : (228, 912),
    'half' : (192, 640)}

class KITTIDataset(Dataset):
    def __init__(self, root, split, resolution='full', augmentation='alhashim'):
        self.root = root
        self.split = split
        self.resolution = resolution_dict[resolution]
        self.augmentation = augmentation

        if split=='train':
            self.transform = self.train_transform
            self.root = os.path.join(self.root, 'train')
        elif split=='val':
            #print('is self transform check 1')
            
            self.transform = self.val_transform
            self.root = os.path.join(self.root, 'val')
        elif split=='test':
            if self.augmentation == 'alhashim':
                self.transform = None
           
            self.root = os.path.join(self.root, 'test')

        self.files = os.listdir(self.root)


    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.files[index])

        data = np.load(image_path)
        depth, image = data['depth'], data['image']
        #print(f'test_imageshape {image.shape}')

        if self.transform is not None:
            #print('is self transform check 2')
            data = self.transform(data)

        image, depth = data['image'], data['depth']
        if self.split == 'test':
            image = np.array(image)
            
            depth = np.array(depth)
            
        #print(f'test image shape here {image.shape}')
        return image, depth

    def __len__(self):
        return len(self.files)



"""
Preparation of dataloaders for Datasets
"""

def get_dataloader(dataset_name, 
                   path,
                   split='train', 
                   resolution='full',
                   augmentation='alhashim',
                   interpolation='linear', 
                   batch_size=8,
                   workers=4, 
                   uncompressed=False):
   
    dataset = KITTIDataset(path, 
                split, 
                resolution=resolution)

    dataloader = DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=(split=='train'),
            num_workers=workers, 
            pin_memory=True)
    return dataloader
