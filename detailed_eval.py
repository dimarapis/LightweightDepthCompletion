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
from scipy.optimize import curve_fit
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

def gt_and_pred_info(gt, pred, sparse, refined_pred):

    sanity_dict = {}
    datatypes = [gt,pred,sparse,refined_pred]
    min_list, median_list,mean_list,max_list, total_valid_list =[],[],[],[],[]
    range02_list,range25_list, range510_list,range1020_list,range20inf_list = [],[],[],[],[]
    
    for data in datatypes:
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
        resolution = torch.numel(data)
        min_list.append(torch.min(data.float()).item())
        median_list.append(torch.median(data.float()).item())
        mean_list.append(torch.mean(data.float()).item())
        max_list.append(torch.max(data.float()).item())
        total_valid_list.append(len(torch.nonzero(data))/(resolution/100))#len(torch.nonzero(data))/(resolution/100)
        if len(torch.nonzero(data)) != 0:
    
            range02_list.append(np.round(len(torch.nonzero(data02))/(len(torch.nonzero(data))/100),decimals=3))
            range25_list.append(np.round(len(torch.nonzero(data25))/(len(torch.nonzero(data))/100),decimals=3))
            range510_list.append(np.round(len(torch.nonzero(data510))/(len(torch.nonzero(data))/100),decimals=3))
            range1020_list.append(np.round(len(torch.nonzero(data1020))/(len(torch.nonzero(data))/100),decimals=3))
            range20inf_list.append(np.round(len(torch.nonzero(data20full))/(len(torch.nonzero(data))/100),decimals=3))

    sanity_dict['min'] = min_list
    sanity_dict['median'] = median_list
    sanity_dict['mean'] = mean_list
    sanity_dict['max'] = max_list
    sanity_dict['total_valid'] = total_valid_list
    sanity_dict['%_range_0-2'] = range02_list
    sanity_dict['%_range_2-5'] = range25_list
    sanity_dict['%_range_5-10'] = range510_list
    sanity_dict['%_range_10-20'] = range1020_list
    sanity_dict['%_range_20-inf'] = range20inf_list
    
    return sanity_dict


def visualize_results(model, rgb, pred, refined_pred, sparse):
    img_list = []

    rgb = np.squeeze(rgb.cpu().detach().numpy())
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



def image_level():
    with torch.no_grad():
        t0 = time.time()
        #data = next(iter(eval_dl))
        headers = ["Statistics", "GT", "Base", "Sparse", "Refined"]

        rmse_list_sing, d1_list_sing, sparse_pts_list_sing = [],[],[]
        
        for i, data in enumerate(tqdm(eval_dl)):

            image_filename = data['file'][0].split('/')[-1].rstrip('.png')
            image, gt, sparse = data['rgb'], data['gt'], data['d']
            
            rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)

            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            
            
            #gt_and_pred_info('basemodel', 'pred', pred)
            #visualize_results('basemodel',image,pred,sparse)

            refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)

            pred_d, depth_gt = pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)    
            computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)


            refined_pred_d, refined_depth_gt = refined_pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            refined_pred_crop, refined_gt_crop = custom_metrics.cropping_img(decnet_args, refined_pred_d, refined_depth_gt)    
            refined_computed_result = custom_metrics.eval_depth(refined_pred_crop, refined_gt_crop)

            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
                refined_result_metrics[metric] += refined_computed_result[metric]
                
            sanity_dict = gt_and_pred_info(gt, pred, sparse, refined_pred)

                
            d = sanity_dict
            d['---------'] = ['------------','------------','------------','------------']
            d[f'split_{i-1}'] = [image_filename]
            d['rmse'] = ['-', computed_result['rmse'], '-', refined_computed_result['rmse']]
            d['d1'] = ['-', computed_result['d1'], '-', refined_computed_result['d1']]
            d['----------'] = ['------------','------------','------------','------------']
            d['Improvement'] = ['RMSE', computed_result['rmse']-refined_computed_result['rmse'], 'D1', refined_computed_result['d1']-computed_result['d1']]
            
            rmse_list_sing.append(refined_computed_result['rmse']-computed_result['rmse'])
            d1_list_sing.append(refined_computed_result['d1']-computed_result['d1'])
            sparse_pts_list_sing.append(sanity_dict['total_valid'][2])
            
            print('\n\n')
            print(tabulate([[k,] + v for k,v in d.items()], headers = headers, tablefmt='github', numalign='right'))  
            print('\n')
            
        fig,ax = plt.subplots()
        #ax.scatter(rmse_list, sparse_pts_list, color="red", marker="*")
        ax.scatter(rmse_list_sing, sparse_pts_list_sing, color="red", marker="*")
        ax.set_ylabel("% points of sparse input")
        ax.set_xlabel("ReD: RMSE difference of refined vs basemodel (+ for better refined)")
        #ax.scatter(rmse_list, trendline, marker="*")
        ax2=ax.twiny()
        #ax2.scatter(d1_list, sparse_pts_list, color="green", marker="*")
        ax2.scatter(d1_list_sing, sparse_pts_list_sing, color="green", marker="*")
        ax2.set_xlabel("GreeN: D1 difference of refined vs basemodel (+ for better refined)")
        plt.show()

def grid_level():
    with torch.no_grad():
        t0 = time.time()
        #data = next(iter(eval_dl))
        headers = ["Statistics", "GT", "Base", "Sparse", "Refined"]

        
        for i, data in enumerate(tqdm(eval_dl)):
            tabulator,rmse_list,d1_list,sparse_pts_list= [],[],[],[]
            rmse_list_sing, d1_list_sing, sparse_pts_list_sing = [],[],[]
            
        #while True:
            #if i == 1:
            #    break
            
            image_filename = data['file'][0].split('/')[-1].rstrip('.png')
            image, gt, sparse = data['rgb'], data['gt'], data['d']
            
            rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)

            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            
            
            #gt_and_pred_info('basemodel', 'pred', pred)
            #visualize_results('basemodel',image,pred,sparse)

            refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)


            image_np = torch.squeeze(image)
            image_np = torch.permute(image_np, (1, 2, 0))
            M = image_np.shape[0]//4
            N = image_np.shape[1]//4
            
            
            gt_np = torch.squeeze(gt,dim=0)
            gt_np = torch.permute(gt_np, (1, 2, 0))

            sparse_np = torch.squeeze(sparse,dim=0)
            sparse_np = torch.permute(sparse_np, (1, 2, 0))
            
            pred_np = torch.squeeze(pred,dim=0)
            pred_np = torch.permute(pred_np, (1, 2, 0))
            
            refined_pred_np = torch.squeeze(refined_pred,dim=0)
            refined_pred_np = torch.permute(refined_pred_np, (1, 2, 0))
            
            image_tiles = [image_np[x:x+M,y:y+N] for x in range(0,image_np.shape[0],M) for y in range(0,image_np.shape[1],N)]
            gt_tiles = [gt_np[x:x+M,y:y+N] for x in range(0,gt_np.shape[0],M) for y in range(0,gt_np.shape[1],N)]
            sparse_tiles = [sparse_np[x:x+M,y:y+N] for x in range(0,sparse_np.shape[0],M) for y in range(0,sparse_np.shape[1],N)]
            pred_tiles = [pred_np[x:x+M,y:y+N] for x in range(0,pred_np.shape[0],M) for y in range(0,pred_np.shape[1],N)]
            refined_pred_tiles = [refined_pred_np[x:x+M,y:y+N] for x in range(0,refined_pred_np.shape[0],M) for y in range(0,refined_pred_np.shape[1],N)]

            for i in range(len(image_tiles)+1):
                if i == 0:
                    pass
                else: 
                    #print(i)
                    image = image_tiles[i-1]
                    gt = gt_tiles[i-1]
                    sparse = sparse_tiles[i-1]
                    pred = pred_tiles[i-1]
                    refined_pred = refined_pred_tiles[i-1]

                pred_d, depth_gt = pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
                pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)    
                computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)


                refined_pred_d, refined_depth_gt = refined_pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
                refined_pred_crop, refined_gt_crop = custom_metrics.cropping_img(decnet_args, refined_pred_d, refined_depth_gt)    
                refined_computed_result = custom_metrics.eval_depth(refined_pred_crop, refined_gt_crop)

                for metric in metric_name:
                    result_metrics[metric] += computed_result[metric]
                    refined_result_metrics[metric] += refined_computed_result[metric]
                    
                sanity_dict = gt_and_pred_info(gt, pred, sparse, refined_pred)

                if i == 0:
                    d = sanity_dict
                    d['---------'] = ['------------','------------','------------','------------']
                    d[f'split_{i-1}'] = [image_filename]
                    d['rmse'] = ['-', computed_result['rmse'], '-', refined_computed_result['rmse']]
                    d['d1'] = ['-', computed_result['d1'], '-', refined_computed_result['d1']]
                    d['----------'] = ['------------','------------','------------','------------']
                    d['Improvement'] = ['RMSE', computed_result['rmse']-refined_computed_result['rmse'], 'D1', refined_computed_result['d1']-computed_result['d1']]
                    
                    rmse_list_sing.append(refined_computed_result['rmse']-computed_result['rmse'])
                    d1_list_sing.append(refined_computed_result['d1']-computed_result['d1'])
                    sparse_pts_list_sing.append(sanity_dict['total_valid'][2])
                    
                    print('\n\n')
                    print(tabulate([[k,] + v for k,v in d.items()], headers = headers, tablefmt='github', numalign='right'))  
                    print('\n')
                    
                else: 
                #for key in sanity_dict:
                #    tabulator.append([key, sanity_dict[key]])
                    d = sanity_dict
                    d['---------'] = ['------------','------------','------------','------------']
                    d[f'split_{i-1}'] = [image_filename]
                    d['rmse'] = ['-', computed_result['rmse'], '-', refined_computed_result['rmse']]
                    d['d1'] = ['-', computed_result['d1'], '-', refined_computed_result['d1']]
                    d['----------'] = ['------------','------------','------------','------------']
                    d['Improvement'] = ['RMSE', computed_result['rmse']-refined_computed_result['rmse'], 'D1', refined_computed_result['d1']-computed_result['d1']]
                    
                    rmse_list.append(refined_computed_result['rmse']-computed_result['rmse'])
                    d1_list.append(refined_computed_result['d1']-computed_result['d1'])
                    sparse_pts_list.append(sanity_dict['total_valid'][2])
                
                    print('\n\n')
                    print(tabulate([[k,] + v for k,v in d.items()], headers = headers, tablefmt='github', numalign='right'))  
                    print('\n')
                
            fig,ax = plt.subplots()
            ax.scatter(rmse_list, sparse_pts_list, color="red", marker="o")
            ax.scatter(rmse_list_sing, sparse_pts_list_sing, color="red", marker="*")
            ax.set_ylabel("% points of sparse input")
            ax.set_xlabel("ReD: RMSE difference of refined vs basemodel (+ for better refined)")
            #ax.scatter(rmse_list, trendline, marker="*")
            ax2=ax.twiny()
            ax2.scatter(d1_list, sparse_pts_list, color="green", marker="^")
            ax2.scatter(d1_list_sing, sparse_pts_list_sing, color="green", marker="*")
            ax2.set_xlabel("GreeN: D1 difference of refined vs basemodel (+ for better refined)")
            plt.show()


image_level()
grid_level()