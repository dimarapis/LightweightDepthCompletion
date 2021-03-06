import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from features.decnet_sanity import torch_min_max


class DecnetDataloader(Dataset):
    def __init__(self, args, datalist):
        #Initialization of class
        self.files = []
        self.data_file = datalist
        self.root = args.root_folder
        self.crop_w = args.val_w
        self.crop_h = args.val_h
        self.dataset_type = args.dataset
        
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

                elif len(data_columns) == 2:
                    self.files.append({
                        "rgb": data_columns[0],
                        "gt": data_columns[1]
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
        
        
        if self.dataset_type == 'nn':
            transformed_rgb = transform(rgb).to('cuda') / 255.
            transformed_sparse = transform(sparse).type(torch.cuda.FloatTensor)/100.#./256.#100. #/ 256.#/ 100.# / 256.
            transformed_gt = transform(gt).type(torch.cuda.FloatTensor)/100.#/256.# 100. #256.#/100.# / 256.
            #mpourda = input(print("SANITY CHECKER, DATASET IS NN"))
        elif self.dataset_type == 'kitti':
            transformed_rgb = transform(rgb).to('cuda') / 255.
            transformed_sparse = transform(sparse).type(torch.cuda.FloatTensor)/256.#./256.#100. #/ 256.#/ 100.# / 256.
            transformed_gt = transform(gt).type(torch.cuda.FloatTensor)/256.#/256.# 100. #256.#/100.# / 256.
            #mpourda = input(print("SANITY CHECKER, DATASET IS KITTI"))
        elif self.dataset_type == 'nyuv2':
            transformed_rgb = transform(rgb).to('cuda') / 255.
            transformed_gt = transform(gt).type(torch.cuda.FloatTensor)/1000.#/256.# 100. #256.#/100.# / 256.
            transformed_sparse = self.get_sparse_depth(transformed_gt, 500)
            print(torch_min_max(transformed_rgb))
            print(torch_min_max(transformed_gt))
            print(torch_min_max(transformed_sparse))
            print(len(torch.nonzero(transformed_sparse)))
            
            #output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}

            #return output

        
        return self.data_sample(file_id, transformed_rgb, transformed_sparse, transformed_gt)
        

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp
    
    
    def __getitem__(self, index):
        # Creates one sample of data 
        rgb = np.array(Image.open(self.files[index]['rgb']))
        gt = np.array(Image.open(self.files[index]['gt']))
        
        sparse = np.array(Image.open(self.files[index]['gt']))
  
        file_id = self.files[index]['rgb']
        transformed_data_sample = self.data_transform(file_id, rgb, sparse, gt)
        return transformed_data_sample
    
    
