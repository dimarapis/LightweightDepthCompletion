import os
import xxlimited
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from features.decnet_sanity import torch_min_max
import torchvision.transforms.functional as TF

from features.decnet_transforms import RandomHorizontalFlip, RandomChannelSwap


def max_depths():
    max_depths = {
    'nyuv2' : 10.0,
    'kitti': 80.0,
    'nn' : 80.0
}
 
def datasets_resolutions():
    resolution_dict =   {
        'nyuv2': {
            'full' : (480, 640),
            'half' : (240, 320),
            'mini' : (224, 224)
        },
        'kitti_res' : {
            'full' : (384, 1280),
            'tu_small' : (128, 416),
            'tu_big' : (228, 912),
            'half' : (192, 640)
        },     
        'nn': {
            'full' : (360, 640),
            'half' : (240, 360),
            'mini' : (120, 160)
        }
    }
    return resolution_dict
    
def crops():
    crops = {
        'kitti' : [128, 381, 45, 1196],
        'nyuv2' : [20, 460, 24, 616],
        'nn' : [4, 356, 16, 624]}


class DecnetDataloader(Dataset):
    def __init__(self, args, datalist, split):
        
        #Initialization of class
        self.files = []
        self.data_file = datalist
        self.split = split
        self.root = args.root_folder
        self.dataset_type = args.dataset
        print(self.dataset_type)
        self.resolution_dict = datasets_resolutions()
        print(self.resolution_dict[self.dataset_type])
        self.resolution = self.resolution_dict[self.dataset_type][args.resolution]
        print(self.resolution)

        self.augment = args.augment
        self.mode = args.mode


        
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
    
    #def completion_transform(self,sparse_data):
    #    tranf_data = sparse_data / 80.
    #    transformed_data = torch.clamp(tranf_data, 0.0, 1.0)
    #    return transformed_data
    
    def data_transform(self, file_id, rgb, sparse, gt):
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(self.resolution),
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
            #print(f'torchnonzero transformed sparse {len(torch.nonzero(transformed_sparse))}')
            #print(f'transformed_gt transformed_gt transformed_gt {len(torch.nonzero(transformed_gt))}')

            print(torch_min_max(transformed_rgb))
            print(torch_min_max(transformed_gt))
            print(torch_min_max(transformed_sparse))
            
            #output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}

            #return output

        
        return self.data_sample(file_id, transformed_rgb, transformed_sparse, transformed_gt)


    def decnet_transform(self, file_id, rgb, sparse, gt):        #HAVE NOT IMPLEMENTED RESOLUTION + AUGMENTATIONS

        
        
        #HAVE NOT IMPLEMENTED RESOLUTION + AUGMENTATIONS
        
        
        toPILtransform = transforms.ToPILImage()#.to('cuda') / 255.
        pil_rgb = toPILtransform(rgb)
        pil_gt = toPILtransform(gt)
        pil_sparse = toPILtransform(sparse)
        

        flip_probability = random.random()
        #print(flip_probability)
        
        if self.augment and self.split == 'train':
            
            
                        #if self.split == 'train':
                
            t_rgb = transforms.Compose([
                transforms.Resize(self.resolution),
                RandomHorizontalFlip(flip_probability),
                RandomChannelSwap(0.5),
                transforms.PILToTensor()
            ])

            t_dep = transforms.Compose([
                transforms.Resize(self.resolution),
                RandomHorizontalFlip(flip_probability),
                transforms.PILToTensor()
            ])
        
            
            if self.dataset_type == 'nn':
                t_rgb_nn = transforms.Compose([
                    transforms.Resize(self.resolution),
                    RandomHorizontalFlip(flip_probability),
                    RandomChannelSwap(0.5),
                    transforms.PILToTensor()
                ])

                t_dep_nn = transforms.Compose([
                    transforms.Resize(self.resolution),
                    RandomHorizontalFlip(flip_probability),
                    transforms.PILToTensor()
                ])
                    
                
                transformed_rgb = t_rgb_nn(pil_rgb).to('cuda') / 255.
                transformed_sparse = t_dep_nn(pil_sparse).type(torch.cuda.FloatTensor)/100.#./256.#100. #/ 256.#/ 100.# / 256.
                transformed_gt = t_dep_nn(pil_gt).type(torch.cuda.FloatTensor)/100.#/256.# 100. #256.#/100.# / 256.
                #print(f'transformed rgb shape {transformed_rgb.shape}')
                #mpourda = input(print("SANITY CHECKER, DATASET IS NN"))
            elif self.dataset_type == 'kitti':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = t_dep(pil_sparse).type(torch.cuda.FloatTensor)/256.#./256.#100. #/ 256.#/ 100.# / 256.
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)/256.#/256.# 100. #256.#/100.# / 256.
                #mpourda = input(print("SANITY CHECKER, DATASET IS KITTI"))
            elif self.dataset_type == 'nyuv2':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = self.get_sparse_depth(t_dep(pil_gt).type(torch.cuda.FloatTensor), 500)
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)#/256.# 100. #256.#/100.# / 256.
            
            
            
        else:
            
                    
            #if self.split == 'train':
                
            t_rgb = transforms.Compose([
                transforms.Resize(self.resolution),
                transforms.PILToTensor()
            ])

            t_dep = transforms.Compose([
                transforms.Resize(self.resolution),
                transforms.PILToTensor()
            ])
        
            
            if self.dataset_type == 'nn':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = t_dep(pil_sparse).type(torch.cuda.FloatTensor)/100.#./256.#100. #/ 256.#/ 100.# / 256.
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)/100.#/256.# 100. #256.#/100.# / 256.
                #print(f'transformed rgb shape {transformed_rgb.shape}')
                #mpourda = input(print("SANITY CHECKER, DATASET IS NN"))
            elif self.dataset_type == 'kitti':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = t_dep(pil_sparse).type(torch.cuda.FloatTensor)/256.#./256.#100. #/ 256.#/ 100.# / 256.
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)/256.#/256.# 100. #256.#/100.# / 256.
                #mpourda = input(print("SANITY CHECKER, DATASET IS KITTI"))
            elif self.dataset_type == 'nyuv2':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = self.get_sparse_depth(t_dep(pil_gt).type(torch.cuda.FloatTensor), 500)
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)#/256.# 100. #256.#/100.# / 256.


        #print(self.split,transformed_rgb.shape,transformed_sparse.shape,transformed_gt.shape)
        
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
        if self.dataset_type == 'nn':
            sparse = np.array(Image.open(self.files[index]['d']))
        elif self.dataset_type == 'nyuv2':
            sparse = np.array(Image.open(self.files[index]['gt']))
            #print(sparse.shape)
        elif self.dataset_type == 'kitti':
            sparse = np.array(Image.open(self.files[index]['d']))

        file_id = self.files[index]['rgb']
        transformed_data_sample = self.decnet_transform(file_id, rgb, sparse, gt)
        return transformed_data_sample
    
    
