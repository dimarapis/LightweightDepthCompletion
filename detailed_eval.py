import os
import cv2
import time
import wandb
import torch
import random
import warnings
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import features.CoordConv as CoordConv

import visualizers.visualizer as visualizer
import features.deprecated_metrics as custom_metrics
import features.custom_transforms as custom_transforms
import features.kitti_loader as guided_depth_kitti_loader

from sympy import Gt
from tqdm import tqdm
from datetime import datetime
from tabulate import tabulate
from os import device_encoding
from torchvision import transforms
from matplotlib import pyplot as plt
from thop import profile,clever_format
from torch.utils.data import DataLoader
from metrics import AverageMeter, Result

import torch.nn.parallel

from models.enet_pro import ENet
from models.enet_basic import weights_init
from models.guide_depth import GuideDepth
from features.decnet_sanity import np_min_max, torch_min_max
from features.decnet_args import decnet_args_parser
from features.decnet_sanity import inverse_depth_norm
from features.decnet_losscriteria import MaskedMSELoss, SiLogLoss
from features.decnet_dataloaders import DecnetDataloader
from models.sparse_guided_depth import AuxSparseGuidedDepth, SparseGuidedDepth
from models.sparse_guided_depth import RgbGuideDepth, SparseAndRGBGuidedDepth, RefinementModule, DepthRefinement, Scaler

#Remove warning for visualization purposes (mostly due to behaviour of upsample block)
warnings.filterwarnings("ignore")

#Loading arguments and model options
print("\nSTEP 1. Loading arguments and parameters...")
decnet_args = decnet_args_parser()

#Print arguments and model options
converted_args_dict = vars(decnet_args)
print('\nParameters list: (Some may be redundant depending on the task, dataset and model chosen)')

#Defining metrics and loggers
metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']

#Initialize weights and biases logger
if decnet_args.wandblogger == True:
    if decnet_args.wandbrunname != '':
        wandb.init(project=str(decnet_args.project),entity=str(decnet_args.entity),name=decnet_args.wandbrunname)
    else:
        wandb.init(project=str(decnet_args.project),entity=str(decnet_args.entity))#)="decnet-project", entity="wandbdimar")
    wandb.config.update(decnet_args)


#Ensuring reproducibility using seed
seed = 2910
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#Loading datasets
print("\nSTEP 2. Loading datasets...")
eval_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.val_datalist),batch_size=1)

print(f'Loaded {len(eval_dl.dataset)} val files')

#Loading model
print("\nSTEP 3. Loading model and metrics...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = RgbGuideDepth(True)
model.load_state_dict(torch.load('./weights/AuxSparseGuidedDepth_26.pth', map_location=device))
model.to(device)
model.eval()

refinement_model = DepthRefinement()
refinement_model.load_state_dict(torch.load('./weights/AuxSparseGuidedDepth_26_ref.pth', map_location=device))
refinement_model.to(device)
refinement_model.eval()


eval_loss = 0.0
refined_eval_loss = 0.0
average_meter = AverageMeter()

result_metrics = {}
refined_result_metrics = {}
for metric in metric_name:
    result_metrics[metric] = 0.0
    refined_result_metrics[metric] = 0.0

def gt_and_pred_info(model, datatype, data):
    mask02 = torch.where((data>0.) & (data<=2.), torch.full_like(data, 1.0), torch.full_like(data, 0.0))
    mask25 = torch.where((data>2.) & (data<=5.), torch.full_like(data, 1.0), torch.full_like(data, 0.0))
    mask510 = torch.where((data>5.) & (data<=10.), torch.full_like(data, 1.0), torch.full_like(data, 0.0))
    mask1020 = torch.where((data>10.) & (data<=20.), torch.full_like(data, 1.0), torch.full_like(data, 0.0))
    mask20full = torch.where(data>20., torch.full_like(data, 1.0), torch.full_like(data, 0.0))

    data02 = torch.mul(data,mask02)
    data25 = torch.mul(data,mask25)
    data510 = torch.mul(data,mask510)
    data1020 = torch.mul(data,mask1020)
    data20full = torch.mul(data,mask20full)
    #print(len(data))
    resolution = torch.numel(data)
    #print(resolution)
    '''
    print(f'model: {model}')
    print(f'datatype: {datatype}')
    print(f'min: {torch.min(data.float())}')
    print(f'median: {torch.median(data.float())}')
    print(f'mean: {torch.mean(data.float())}')
    print(f'max: {torch.max(data.float())}')    
    print(f'total_valid_%: {len(torch.nonzero(data))/(resolution/100)}')
    print(f'valid_%_in_range_0-2: {len(torch.nonzero(data02))/(len(torch.nonzero(data))/100)}')#(resolution/100)}')
    print(f'valid_%_in_range_2-5: {len(torch.nonzero(data25))/(len(torch.nonzero(data))/100)}')#(resolution/100)}')
    print(f'valid_%_in_range_5-10: {len(torch.nonzero(data510))/(len(torch.nonzero(data))/100)}')#(resolution/100)}')
    print(f'valid_%_in_range_10-20: {len(torch.nonzero(data1020))/(len(torch.nonzero(data))/100)}')#(resolution/100)}')
    print(f'valid_%_in_range_20-inf: {len(torch.nonzero(data20full))/(len(torch.nonzero(data))/100)}')#(resolution/100)}')
    '''
    sanity_dict = {}
    sanity_dict['datatype'] = datatype
    sanity_dict['min'] = torch.round(torch.min(data.float()),decimals=3)
    sanity_dict['median'] = torch.round(torch.median(data.float()),decimals=3)
    sanity_dict['mean'] = torch.round(torch.mean(data.float()),decimals=3)
    sanity_dict['max'] = torch.round(torch.max(data.float()), decimals = 3)
    sanity_dict['%_range_0-2'] = np.round(len(torch.nonzero(data02))/(len(torch.nonzero(data))/100),decimals=3)
    sanity_dict['%_range_2-5'] = np.round(len(torch.nonzero(data25))/(len(torch.nonzero(data))/100),decimals=3)
    sanity_dict['%_range_5-10'] = np.round(len(torch.nonzero(data510))/(len(torch.nonzero(data))/100),decimals=3)
    sanity_dict['%_range_10-20'] = np.round(len(torch.nonzero(data1020))/(len(torch.nonzero(data))/100),decimals=3)
    sanity_dict['%_range_20-inf'] = np.round(len(torch.nonzero(data20full))/(len(torch.nonzero(data))/100),decimals=3)

    
    
    #sanity_dict['datatype'] = datatype
    #sanity_dict['datatype'] = datatype
    return sanity_dict

    

    #print(f'datatype: {datatype}')
    

def visualize_results(model, rgb, pred, refined_pred, sparse):
    img_list = []

    rgb = np.squeeze(image.cpu().detach().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb*255
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img_list.append(rgb)

    sparse_vis = np.squeeze(sparse.cpu().detach().numpy())
    sparse_vis = visualizer.depth_colorize(sparse_vis)
    sparse_vis = cv2.cvtColor(sparse_vis, cv2.COLOR_RGB2BGR)
    img_list.append(sparse_vis)
    
    depth = np.squeeze(pred.cpu().detach().numpy())
    depth = visualizer.depth_colorize(depth)
    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
    img_list.append(depth)
    
    refined_depth = np.squeeze(refined_pred.cpu().detach().numpy())
    refined_depth = visualizer.depth_colorize(refined_depth)
    refined_depth = cv2.cvtColor(refined_depth, cv2.COLOR_RGB2BGR)
    img_list.append(refined_depth)
    
    
    img_merge = np.hstack(img_list)
    #cv2.imshow('queens', )
    cv2.imwrite(f'{model}.jpg',img_merge.astype('uint8'))
    #print(img_merge)
    #cv2.waitKey()

with torch.no_grad():
    t0 = time.time()
    #data = next(iter(eval_dl))
    tabulator = []
    for i, data in enumerate(tqdm(eval_dl)):
        if i == 1:
            image_filename = data['file']
            image, gt, sparse = data['rgb'], data['gt'], data['d']
            rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)

            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            
            sanity_dict = gt_and_pred_info('basemodel', 'gt', gt)
            for key in sanity_dict:
                tabulator.append([key, sanity_dict[key]])
            #gt_and_pred_info('basemodel', 'pred', pred)
            #visualize_results('basemodel',image,pred,sparse)

            refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)

            #gt_and_pred_info('basemodel', 'gt', gt)
            #gt_and_pred_info('refine_model', 'refined_pred', refined_pred)
            visualize_results('basemodel',image,pred,refined_pred,sparse)

            pred_d, depth_gt = pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)    
            computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)


            refined_pred_d, refined_depth_gt = refined_pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            refined_pred_crop, refined_gt_crop = custom_metrics.cropping_img(decnet_args, refined_pred_d, refined_depth_gt)    
            refined_computed_result = custom_metrics.eval_depth(refined_pred_crop, refined_gt_crop)

            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
                refined_result_metrics[metric] += refined_computed_result[metric]
    
    print(tabulate(tabulator, tablefmt='orgtbl'))
    
