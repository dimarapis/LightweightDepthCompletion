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


class DecnetDataloader(Dataset):
    def __init__(self, args, datalist):
        #Initialization of class
        self.files = []
        self.data_file = datalist
        self.root = args.root_folder
        
        self.dataset_type = args.dataset
        self.augment = args.augment
        self.mode = args.mode
        if self.dataset_type == 'nn':
            self.width = 640
            self.height = 360
            self.crop_w = 608
            self.crop_h = 352
        elif self.dataset_type == 'kitti':
            self.height = 'needtofixthis'
            self.width = 'needtofixthis'
            self.crop_w = 'needtofixthis'
            self.crop_h = 'needtofixthis'
        elif self.dataset_type == 'nyuv2':
            self.width = 320
            self.height = 240
            self.crop_w = 304
            self.crop_h = 228

        
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
            print(f'torchnonzero transformed sparse {len(torch.nonzero(transformed_sparse))}')
            print(f'transformed_gt transformed_gt transformed_gt {len(torch.nonzero(transformed_gt))}')

            #print(torch_min_max(transformed_rgb))
            #print(torch_min_max(transformed_gt))
            #print(torch_min_max(transformed_sparse))
            #print(len(torch.nonzero(transformed_sparse)))
            
            #output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}

            #return output

        
        return self.data_sample(file_id, transformed_rgb, transformed_sparse, transformed_gt)


    def nyuv2_transform(self, file_id, rgb, sparse, gt):
        #Ensuring reproducibility using seed
        seed = 2910
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        toPILtransform = transforms.ToPILImage()#.to('cuda') / 255.
        pil_rgb = toPILtransform(rgb)
        pil_gt = toPILtransform(gt)
        pil_sparse = toPILtransform(sparse)
        #print(pil_rgb)
        
        
        
        
        #if self.dataset_type == 'nyuv2':


        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                pil_rgb = TF.hflip(pil_rgb)
                pil_gt = TF.hflip(pil_gt)
                pil_sparse = TF.hflip(pil_sparse)


            pil_rgb = TF.rotate(pil_rgb, angle=degree, resample=Image.NEAREST)
            pil_gt = TF.rotate(pil_gt, angle=degree, resample=Image.NEAREST)
            pil_sparse = TF.rotate(pil_sparse, angle=degree, resample=Image.NEAREST)


            t_rgb = transforms.Compose([
                transforms.Resize(scale),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop((self.crop_h, self.crop_w)),
                transforms.PILToTensor()
            ])

            t_dep = transforms.Compose([
                transforms.Resize(scale),
                transforms.CenterCrop((self.crop_h, self.crop_w)),
                transforms.PILToTensor()
            ])

                #transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
        
            if self.dataset_type == 'nn':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = t_dep(pil_sparse).type(torch.cuda.FloatTensor)/100.#./256.#100. #/ 256.#/ 100.# / 256.
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)/100.#/256.# 100. #256.#/100.# / 256.
                #mpourda = input(print("SANITY CHECKER, DATASET IS NN"))
            elif self.dataset_type == 'kitti':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = t_dep(pil_sparse).type(torch.cuda.FloatTensor)/256.#./256.#100. #/ 256.#/ 100.# / 256.
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)/256.#/256.# 100. #256.#/100.# / 256.
                #mpourda = input(print("SANITY CHECKER, DATASET IS KITTI"))
            elif self.dataset_type == 'nyuv2':
                transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
                transformed_sparse = self.get_sparse_depth(pil_sparse, 500)
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)/1000.#/256.# 100. #256.#/100.# / 256.
                print(f'torchnonzero transformed sparse {len(torch.nonzero(transformed_sparse))}')
                print(f'transformed_gt transformed_gt transformed_gt {len(torch.nonzero(transformed_gt))}')
            
            transformed_sparse = transformed_sparse / _scale
            transformed_gt = transformed_gt / _scale

        else: 
            self.crop_h = 352
            self.crop_w = 608

            t_rgb = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize(self.height),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop((self.crop_h, self.crop_w)),
                transforms.PILToTensor()
            ])

            t_dep = transforms.Compose([
                transforms.Resize(self.height),
                transforms.CenterCrop((self.crop_h, self.crop_w)),
                transforms.PILToTensor()
            ])

                #transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
        
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
                transformed_sparse = self.get_sparse_depth(pil_sparse, 500)
                transformed_gt = t_dep(pil_gt).type(torch.cuda.FloatTensor)/1000.#/256.# 100. #256.#/100.# / 256.
                ##print(f'torchnonzero transformed sparse {len(torch.nonzero(transformed_sparse))}')
                #print(f'transformed_gt transformed_gt transformed_gt {len(torch.nonzero(transformed_gt))}')

            #print(torch_min_max(transformed_rgb))
            #print(torch_min_max(transformed_gt))
            #print(torch_min_max(transformed_sparse))
            #print(len(torch.nonzero(transformed_sparse)))
            
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
        if self.dataset_type == 'nn':
            sparse = np.array(Image.open(self.files[index]['d']))
        elif self.dataset_type == 'nyuv2':
            sparse = np.array(Image.open(self.files[index]['gt']))
            
        file_id = self.files[index]['rgb']
        transformed_data_sample = self.nyuv2_transform(file_id, rgb, sparse, gt)
        return transformed_data_sample
    
    
