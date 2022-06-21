import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DecnetDataloader(Dataset):
    def __init__(self, args, datalist):
        #Initialization of class
        self.files = []
        self.data_file = datalist
        self.root = args.root_folder
        self.crop_w = args.val_w
        self.crop_h = args.val_h
        
        with open(os.path.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data_row in data_list:
                if len(data_row) == 0:
                    continue
                
                data_columns = data_row.split(' ')
            
                if len(data_columns) == 3:
                    self.files.append({
                        "rgb": data_columns[0],
                        "d": data_columns[1],
                        "gt": data_columns[2]
                        })
                    
                ''' # silencio
                else:
                    self.files.append({
                        "rgb": data_columns[0],
                        "sparse": data_columns[1],
                        })
                    self.no_gt = True
                '''
    def __len__(self):
        #Returns amount of samples
        return len(self.files)
    
    def data_sample(self, file_id, transformed_rgb, transformed_sparse, transformed_gt):
        #Creating a sample as dict
        sample = {'file': file_id,
                 'rgb': transformed_rgb, 
                 'd': transformed_sparse,
                 'gt': transformed_gt}
        
        return sample
    
    def completion_transform(self,sparse_data):
        tranf_data = sparse_data / 80.
        transformed_data = torch.clamp(tranf_data, 0.0, 1.0)
        return transformed_data
    
    def data_transform(self, file_id, rgb, sparse, gt):
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((self.crop_h, self.crop_w)),
        transforms.PILToTensor()
        ])
        
        
        
        transformed_rgb = transform(rgb).to('cuda') / 255.
        transformed_sparse = transform(sparse).type(torch.cuda.FloatTensor) / 256.
        transformed_sparse_again = self.completion_transform(transformed_sparse)
        transformed_gt = transform(gt).type(torch.cuda.FloatTensor) / 256.
        
        return self.data_sample(file_id, transformed_rgb, transformed_sparse_again, transformed_gt)
        

    def __getitem__(self, index):
        # Creates one sample of data 
        rgb = np.array(Image.open(self.files[index]['rgb']))
        sparse = np.array(Image.open(self.files[index]['d']))
        gt = np.array(Image.open(self.files[index]['gt']))
        file_id = self.files[index]['rgb']
        transformed_data_sample = self.data_transform(file_id, rgb, sparse, gt)
        return transformed_data_sample
    
    
