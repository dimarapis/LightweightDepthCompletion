import os
from termios import TIOCSERGETMULTI
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
from nlspnconfig import args
from models.nlspnmodel import NLSPNModel

from features.decnet_args import decnet_args_parser

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
from models.s2d import ResNet
import torch.nn.parallel

from models.enet_pro import ENet
from models.enet_basic import weights_init
from models.guide_depth import GuideDepth
from features.decnet_sanity import np_min_max, torch_min_max
from features.decnet_args import decnet_args_parser
from features.decnet_sanity import inverse_depth_norm
from features.decnet_losscriteria import MaskedMSELoss, SiLogLoss
from features.decnet_dataloaders import DecnetDataloader
from models.sparse_guided_depth import AuxSparseGuidedDepth, DecnetNLSPN, DecnetNLSPN_sharedDecoder, SparseGuidedDepth, DecnetDepthRefinement
from models.sparse_guided_depth import RgbGuideDepth, SparseAndRGBGuidedDepth, RefinementModule, DepthRefinement, Scaler
from models.sparse_guided_depth import DecnetSparseIncorporated

torch.cuda.empty_cache()
decnet_args = decnet_args_parser()


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
eval_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.val_datalist,split='eval'),batch_size=1)

print(f'Loaded {len(eval_dl.dataset)} val files')

#Loading model
print("\nSTEP 3. Loading model and metrics...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GuideDepth()
model.load_state_dict(torch.load('./weights/GuideDepthOriginal.pth', map_location=device))




model.to(device)
model.eval()

#refinement_model = DepthRefinement()
#refinement_model.load_state_dict(torch.load('./weights/nn_final_ref.pth', map_location=device))
#refinement_model.to(device)
#refinement_model.eval()


#refinement_model = DecnetDepthRefinement()
#refinement_model.load_state_dict(torch.load('./weights/2022_08_19-03_03_48_PM/DecnetModule_99_ref.pth', map_location=device))
#refinement_model.load_state_dict(torch.load('./weights/DecnetModule_19_ref.pth', map_location=device))

refinement_model = DecnetSparseIncorporated()
#refinement_model.load_state_dict(torch.load('./weights/DecnetModule_50k_500.pth', map_location='cpu'))
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

print("\nSTEP 4. Test time...\n")


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


def visualize_results(image, rgb, pred, refined_pred, sparse, gt):
    img_list_row_1 = []
    img_list_row_2 = []
    img_list_row_3 = []
    
    rgb = np.squeeze(rgb.cpu().detach().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb*255
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img_list_row_1.append(rgb)
    
    depth = np.squeeze(pred.cpu().detach().numpy())
    depth_colorize_min_depth = 0
    depth_colorize_max_depth = np.max(depth) * 1.2
    depth_2 = visualizer.depth_colorize_fixed_ranges(depth,depth_colorize_min_depth,depth_colorize_max_depth)
    depth_2 = cv2.cvtColor(depth_2, cv2.COLOR_RGB2BGR)
    
    sparse_vis = np.squeeze(sparse.cpu().detach().numpy())
    sparse_vis_2 = visualizer.depth_colorize_fixed_ranges(sparse_vis,depth_colorize_min_depth,depth_colorize_max_depth)
    sparse_vis_2 = cv2.cvtColor(sparse_vis_2, cv2.COLOR_RGB2BGR)
    img_list_row_1.append(sparse_vis_2)  
    
    gt_vis = np.squeeze(gt.cpu().detach().numpy())
    gt_vis_2 = visualizer.depth_colorize_fixed_ranges(gt_vis,depth_colorize_min_depth,depth_colorize_max_depth)
    gt_vis_2 = cv2.cvtColor(gt_vis_2, cv2.COLOR_RGB2BGR)
    img_list_row_1.append(gt_vis_2)
    img_list_row_2.append(depth_2)
    
    refined_depth = np.squeeze(refined_pred.cpu().detach().numpy())
    refined_depth_2 = visualizer.depth_colorize_fixed_ranges(refined_depth,depth_colorize_min_depth,depth_colorize_max_depth)
    refined_depth_2 = cv2.cvtColor(refined_depth_2, cv2.COLOR_RGB2BGR)
    img_list_row_2.append(refined_depth_2)
    #img_list_row_2.append(np.zeros(refined_depth_2.shape))
    
    gt_vis = np.squeeze(gt.cpu().detach().numpy())
    error_map_total_basemodel = np.where(gt_vis, gt_vis-depth, gt_vis)
    error_map_total_refined = np.where(gt_vis, gt_vis-refined_depth, gt_vis)#np.where(refined_depth, refined_depth-depth, refined_depth)# / np.where(gt_vis, gt_vis-refined_depth, gt_vis)
    error_map_sparse_present = np.where(np.logical_and(gt_vis > 0,sparse_vis > 0), gt_vis-refined_depth, 0)#np.where(refined_depth, refined_depth-depth, refined_depth)# / np.where(gt_vis, gt_vis-refined_depth, gt_vis)
    error_map_sparse_absent = np.where(np.logical_and(gt_vis > 0,sparse_vis == 0), gt_vis-refined_depth, 0)#np.where(refined_depth, refined_depth-depth, refined_depth)# / np.where(gt_vis, gt_vis-refined_depth, gt_vis)
    sensor_error = np.where(np.logical_and(gt_vis > 0,sparse_vis > 0), gt_vis-sparse_vis, 0)#np.where(refined_depth, refined_depth-depth, refined_depth)# / np.where(gt_vis, gt_vis-refined_depth, gt_vis)
    if decnet_args.show_sensor_error == True:
        error_map_sparse_present = sensor_error
        #print(f'error between sensors valid datapoints!  np.min: {np.min(sensor_error[np.nonzero(sensor_error)])}, np.max: {np.max(sensor_error[np.nonzero(sensor_error)])}, np.mean: {np.mean(sensor_error[np.nonzero(sensor_error)])} and np.median: {np.median(sensor_error[np.nonzero(sensor_error)])} ')
        
    #print(f'gt_vis {np.min(gt_vis)} and {np.max(gt_vis)}')
    #print(f'gt_vis {np.min(sparse_vis)} and {np.max(sparse_vis)}')
    #print(f'gt_vis {np.min(gt_vis)} and {np.max(gt_vis)}')
    
    
    
    #print(f'test_error_map_absent {np.min(error_map_sparse_absent)} and {np.max(error_map_sparse_absent)}')

    #print(error_map)
    
    error_colorize_min = np.min(error_map_total_basemodel)
    error_colorize_max = np.max(error_map_total_basemodel)
    #print(error_colorize_min, error_colorize_max)

    if error_colorize_min < decnet_args.error_vis_min:
        error_colorize_min = decnet_args.error_vis_min
        filtered_error_map_total_basemodel = np.where(error_map_total_basemodel < decnet_args.error_vis_min, decnet_args.error_vis_min, error_map_total_basemodel)
        filtered_error_map_total_refined = np.where(error_map_total_refined < decnet_args.error_vis_min, decnet_args.error_vis_min, error_map_total_refined)
        filtered_error_map_sparse_present = np.where(error_map_sparse_present < decnet_args.error_vis_min, decnet_args.error_vis_min, error_map_sparse_present)
        filtered_error_map_sparse_absent = np.where(error_map_sparse_absent < decnet_args.error_vis_min, decnet_args.error_vis_min, error_map_sparse_absent)
        
        #print(f'test1 {np.min(filtered_error_map_sparse_absent)} and {np.max(filtered_error_map_total_basemodel)}')
        #error_colorize_min = -8.0
        
    else: 
        filtered_error_map_total_basemodel = error_map_total_basemodel
        filtered_error_map_total_refined = error_map_total_refined
        filtered_error_map_sparse_present = error_map_sparse_present
        filtered_error_map_sparse_absent = error_map_sparse_absent
    
    if error_colorize_max > decnet_args.error_vis_max:
        filtered_error_map_total_basemodel = np.where(filtered_error_map_total_basemodel > decnet_args.error_vis_max, decnet_args.error_vis_max, filtered_error_map_total_basemodel)
        filtered_error_map_total_refined = np.where(filtered_error_map_total_refined > decnet_args.error_vis_max, decnet_args.error_vis_max, filtered_error_map_total_refined)
        filtered_error_map_sparse_present = np.where(filtered_error_map_sparse_present > decnet_args.error_vis_max, decnet_args.error_vis_max, filtered_error_map_sparse_present)
        filtered_error_map_sparse_absent = np.where(filtered_error_map_sparse_absent > decnet_args.error_vis_max, decnet_args.error_vis_max, filtered_error_map_sparse_absent)
        
        
        error_colorize_max = decnet_args.error_vis_max
    
    #if decnet_args.show_sensor_error == True:
        #print(filtered_error_map_sparse_present.shape)
        #error_map_sparse_present = sensor_error
    #    print(f'error between sensors valid datapoints!  np.min: {np.min(filtered_error_map_sparse_present[np.nonzero(filtered_error_map_sparse_present)])}, np.max: {np.max(filtered_error_map_sparse_present[np.nonzero(filtered_error_map_sparse_present)])}, np.mean: {np.mean(filtered_error_map_sparse_present[np.nonzero(filtered_error_map_sparse_present)])} and np.median: {np.median(filtered_error_map_sparse_present[np.nonzero(filtered_error_map_sparse_present)])} ')
    #error_colorize_mean = np.mean(error_map)
    #print(f'basemodel min: {np.min(filtered_error_map_total_basemodel)}, basemodel max: {np.max(filtered_error_map_total_basemodel)}')
    #print(f'refined min: {np.min(filtered_error_map_total_refined)}, refined max: {np.max(filtered_error_map_total_refined)}')
    #print(f'sparse_present min: {np.min(filtered_error_map_sparse_present)}, sparse_present max: {np.max(filtered_error_map_sparse_present)}')
    #print(f'sparse_absent min: {np.min(filtered_error_map_sparse_absent)}, sparse_absent max: {np.max(filtered_error_map_sparse_absent)}')
    
    #print(np.min(filtered_error_map_total_refined), np.max(filtered_error_map_total_refined))
    
    #print(error_colorize_max)
    error_map_col_basemodel = visualizer.error_map_colorizer(filtered_error_map_total_basemodel,error_colorize_min,error_colorize_max)
    error_map_col_basemodel = cv2.cvtColor(error_map_col_basemodel, cv2.COLOR_RGB2BGR)
    img_list_row_3.append(error_map_col_basemodel)
    #img_list_row_3.append(np.where(error_map_col_basemodel==246, 0, error_map_col_basemodel))
    #print(np.mean(error_map_col_basemodel))
    
    error_map_col_refined = visualizer.error_map_colorizer(filtered_error_map_total_refined,error_colorize_min,error_colorize_max)
    error_map_col_refined = cv2.cvtColor(error_map_col_refined, cv2.COLOR_RGB2BGR)
    img_list_row_3.append(error_map_col_refined)
    #img_list_row_3.append(np.where(error_map_col_refined==246, 0, error_map_col_refined))
    
    error_map_col_sparse_present = visualizer.error_map_colorizer(filtered_error_map_sparse_present,error_colorize_min,error_colorize_max)
    error_map_col_sparse_present = cv2.cvtColor(error_map_col_sparse_present, cv2.COLOR_RGB2BGR)
    img_list_row_2.append(error_map_col_sparse_present)
    #img_list_row_3.append(np.zeros(error_map_col_refined.shape))
    #print(np.mean(error_map_col_refined))


    error_map_col_sparse_absent = visualizer.error_map_colorizer(filtered_error_map_sparse_absent,error_colorize_min,error_colorize_max)
    error_map_col_sparse_absent = cv2.cvtColor(error_map_col_sparse_absent, cv2.COLOR_RGB2BGR)
    img_list_row_3.append(error_map_col_sparse_absent)
    #img_list_row_3.append(np.zeros(error_map_col_refined.shape))
    #print(np.mean(error_map_col_refined))

    #print(refined_depth.shape)
    #print(error_map.shape)
    img_merge_row_1 = np.hstack(img_list_row_1)
    img_merge_row_2 = np.hstack(img_list_row_2)
    img_merge_row_3 = np.hstack(img_list_row_3)
    
    
    img_merge = np.vstack((img_merge_row_1,img_merge_row_2,img_merge_row_3))
    #cv2.imshow('queens', )
    cv2.imwrite(f'results/{image}.jpg',img_merge.astype('uint8'))
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
            print(f'image: {torch_min_max(image)}')    
            max_depth  = decnet_args.max_depth_eval
            
            if decnet_args.dataset == 'nyuv2':
               
                gt = gt * 0.001
                sparse = sparse * 0.001
                
                max_depth = 10

            flipped_evaluation = True
            raise error
            if flipped_evaluation:
                image_flip = torch.flip(image, [3])
                gt_flip = torch.flip(gt, [3])
            
            #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
            inv_pred = model(image)
            #print(inv_pred)
            
            pred = inverse_depth_norm(max_depth,inv_pred)
            
                        
            print(f'pred: {torch_min_max(pred)}')    
            print(f'gt: {torch_min_max(gt)}')
            print(f'sparse: {torch_min_max(sparse)}')    
            #gt_and_pred_info('basemodel', 'pred', pred)
            #visualize_results('basemodel',image,pred,sparse)
            #refined_pred = pred
            refined_inv_pred = refinement_model(image,sparse)
            #refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)
            refined_pred = inverse_depth_norm(max_depth,refined_inv_pred)
            print(f'refined_pred: {torch_min_max(refined_pred)}')    
            
                
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
            visualize_results(image_filename,image,pred,refined_pred,sparse,gt)


                
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
            #break
            
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
            
            
            if decnet_args.dataset == 'nyuv2':
               
                gt = gt * 0.001
                sparse = sparse * 0.001
                
                max_depth = 10
            
            #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)

            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            
            
            #gt_and_pred_info('basemodel', 'pred', pred)

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


def gpu_timings(models):
    for modelo in models:              
        if modelo == 's2d':
            print(f'GPU timings for model {modelo}')
            model = ResNet(layers=50, decoder='deconv2', output_size=(480,640),
                in_channels=4, pretrained=False)
            model.to(device)
            model.eval()
            
        elif modelo == 'GuideDepth':
            print(f'GPU timings for model {modelo}')
            model = GuideDepth()
            #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location=device))
            model.to(device)
            model.eval()
        elif modelo == 'GuideDepth-small':
            print(f'GPU timings for model {modelo}')
            model = GuideDepth(up_features=[32, 8, 4], inner_features=[32, 8, 4])
            #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location=device))
            model.to(device)
            model.eval()
        elif modelo == 'DecnetModule':
            print(f'GPU timings for model {modelo}')
            model = DecnetSparseIncorporated()
            model.to(device)
            model.eval()
            
        elif modelo == 'DecnetModule-small':
            print(f'GPU timings for model {modelo}')
            model = DecnetSparseIncorporated(up_features=[32, 8, 4], inner_features=[32, 8, 4])
            model.to(device)
            model.eval()
            
        elif modelo == 'nlspn':
            model = NLSPNModel(args)
            model.to(device)
            model.eval()
            
        elif modelo == 'decnetnlspn':
            model = DecnetNLSPN(decnet_args)
            model.to(device)
            model.eval()
            
        elif modelo == 'decnetnlspn_encoshared':
            model = DecnetNLSPN_sharedDecoder(decnet_args)
            model.to(device)
            model.eval()
        
        
        print("Calculating inference for model...")
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings = np.zeros((repetitions, 1))

        # GPU warm-up
        for _ in range(20):
            test_data_rgb = torch.randint(0, 256, (1, 3, 480, 640)).to(device)
            test_data_sparse = torch.randint(0, 256, (1, 1, 480, 640)).to(device)
            test_data_rgb = test_data_rgb.to(torch.float32)
            test_data_sparse = test_data_sparse.to(torch.float32)
            
            if modelo == 's2d':
                parse = model(torch.cat((test_data_rgb,test_data_sparse),dim=1))
            elif modelo == 'GuideDepth' or modelo == 'GuideDepth-small':
                parse = model(test_data_rgb)
            elif modelo == 'DecnetModule' or modelo == 'DecnetModule-small' or modelo == 'decnetnlspn' or modelo == 'decnetnlspn_encoshared':
                parse = model(test_data_rgb,test_data_sparse)
            elif modelo == 'nlspn':
                sample = dict()
                sample['rgb'] = test_data_rgb
                sample['dep'] = test_data_sparse
                parse = model(sample)
            
                
            #pred = inverse_depth_norm(80.0,inv_pred)


        # Measure performance 
        with torch.no_grad():
            for rep in range(repetitions):
                
                test_data_rgb = torch.randint(0, 256, (1, 3, 480, 640)).to(device)
                test_data_sparse = torch.randint(0, 256, (1, 1, 480, 640)).to(device)
                test_data_rgb = test_data_rgb.to(torch.float32)
                test_data_sparse = test_data_sparse.to(torch.float32)
                
                starter.record()
                    
                if modelo == 's2d':
                    parse = model(torch.cat((test_data_rgb,test_data_sparse),dim=1))                
                elif modelo == 'GuideDepth' or modelo == 'GuideDepth-small':
                    parse = model(test_data_rgb)
                elif modelo == 'DecnetModule' or modelo == 'DecnetModule-small' or modelo == 'decnetnlspn' or modelo == 'decnetnlspn_encoshared':
                    parse = model(test_data_rgb,test_data_sparse)
                elif modelo == 'nlspn':
                    sample = dict()
                    sample['rgb'] = test_data_rgb
                    sample['dep'] = test_data_sparse
                    parse = model(sample)
                ender.record()

                # Wait for GPU to sync
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        # Calculate mean and std
        mean_time = np.sum(timings) / repetitions
        std_time = np.std(timings)
        #print(f'{modelo} model timings calculation...\n')
        
        print(f'{modelo} model timings calculation\nMean time to process {repetitions} frames: {mean_time}, with std_deviation of: {std_time}')


def model_summary(model):
    from pytorch_model_summary import summary
    architecture = summary(model, torch.zeros((1, 1, 360, 480)), show_input=False, show_hierarchical=True)
    print(architecture)
    with open('architecture_summary.txt', 'w') as f:
        f.write(architecture)

    

gpu_timings(['decnetnlspn_encoshared','decnetnlspn','nlspn','s2d', 'GuideDepth', 'GuideDepth-small', 'DecnetModule', 'DecnetModule-small'])
#image_level()
#grid_level()

#model_summary(refinement_model)
