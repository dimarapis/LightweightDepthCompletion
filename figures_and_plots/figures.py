from unittest.main import MODULE_EXAMPLES
import cv2
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath('../'))

#models
from models.s2d import ResNet
from models.nlspnmodel import NLSPNModel
from models.sparse_guided_depth import DecnetNLSPN
from models.sparse_guided_depth import DecnetNLSPNSmall

from torch.utils.data import DataLoader
import visualizers.visualizer as visualizer
from features.decnet_args import decnet_args_parser
from features.decnet_sanity import inverse_depth_norm
from features.decnet_dataloaders import DecnetDataloader




torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

decnet_args = decnet_args_parser()

figure_dl = DataLoader(DecnetDataloader(decnet_args,'nyu_figures.list',split='eval'),batch_size=1)

samples = list()
upscale_to_full_resolution = torchvision.transforms.Resize((480,640))
downsample_to_half_resolution = torchvision.transforms.Resize((240,320))

#models = ['S2D','DecnetNLSPNsmall','DecnetNLSPN','NLSPN']
models = ['DecnetNLSPNsmall','DecnetNLSPN']

sample_id = 0




color = [255, 255, 255] # 'cause purple!

# border widths; I set them all to 150
top, bottom, left, right = [5]*4


for sample in figure_dl:

    image_filename = sample['file'][0].split('/')[-1].rstrip('.png')
    image, gt, sparse = sample['rgb'], sample['gt'], sample['d']
    
    gt = gt * 0.001
    sparse = sparse * 0.001
    max_depth = 10
    
    img_list_row = []
    
    #image, gt, sparse= downsample_to_half_resolution(image_orig), downsample_to_half_resolution(gt_orig),downsample_to_half_resolution(sparse_orig)
   
    #gt_resized, rgb_resized, sparse_resized = downsample_to_half_resolution(image_orig).squeeze(), image_orig.squeeze(),sparse_orig.squeeze()
    
    depth_colorize_min_depth = np.min(gt.cpu().detach().numpy())
    depth_colorize_max_depth = np.max(gt.cpu().detach().numpy())
    
    rgb = np.squeeze(image.cpu().detach().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb*255
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rgb_with_border = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    
    img_list_row.append(rgb_with_border)
    
    sparse_vis = np.squeeze(sparse.cpu().detach().numpy())
    
    depth_colorize_min_depth =np.min(sparse_vis) #np.min(gt.cpu().detach().numpy())
    depth_colorize_max_depth = np.max(sparse_vis)# np.max(gt.cpu().detach().numpy())
    
    
    sparse_vis_2 = visualizer.depth_colorize_fixed_ranges(sparse_vis,depth_colorize_min_depth,depth_colorize_max_depth)
    sparse_vis_2 = cv2.cvtColor(sparse_vis_2, cv2.COLOR_RGB2BGR)
    sparse_vis_2_with_border = cv2.copyMakeBorder(sparse_vis_2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    img_list_row.append(sparse_vis_2_with_border)  
    
    
    
    gt_vis = np.squeeze(gt.cpu().detach().numpy())

    depth_colorize_min_depth =np.min(gt_vis) #np.min(gt.cpu().detach().numpy())
    depth_colorize_max_depth = np.max(gt_vis)# np.max(gt.cpu().detach().numpy())
    
    gt_vis_2 = visualizer.depth_colorize_fixed_ranges(gt_vis,depth_colorize_min_depth,depth_colorize_max_depth)
    gt_vis_2 = cv2.cvtColor(gt_vis_2, cv2.COLOR_RGB2BGR)

    
    
    
    for model in models:
        
        if model == 'S2D':
            net = ResNet(layers=50, decoder='deconv2', output_size=(480,640),
                    in_channels=4, pretrained=False)
            net.to(device)
            net.eval()
            output = net(torch.cat((image, sparse),dim=1))
            
            pred = torch.zeros(1,1,240,320)
            pred_resized = upscale_to_full_resolution(pred).squeeze()
            pred_resized = np.squeeze(pred_resized.cpu().detach().numpy())
            
    
            pred_resized = visualizer.depth_colorize_fixed_ranges(pred_resized,depth_colorize_min_depth,depth_colorize_max_depth)
            pred_resized = cv2.cvtColor(pred_resized, cv2.COLOR_RGB2BGR)
            pred_resized = cv2.resize(pred_resized, (320,240), interpolation = cv2.INTER_AREA)
            pred_resized_with_border = cv2.copyMakeBorder(pred_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            
            img_list_row.append(pred_resized_with_border)
                 
        elif model == 'NLSPN':
            from nlspnconfig import args
            net = NLSPNModel(args)
            net.to(device)
            #net.load_state_dict(torch.load('../weights/NLSPN_NYU.pt', map_location=device))
            net.eval()
            output = net(image, sparse)
            #pred = output['pred']
            #pred = inverse_depth_norm(max_depth,inv_pred)
            pred = torch.zeros(1,1,240,320)
            pred_resized = upscale_to_full_resolution(pred).squeeze()
            pred_resized = np.squeeze(pred_resized.cpu().detach().numpy())
            pred_resized = visualizer.depth_colorize_fixed_ranges(pred_resized,depth_colorize_min_depth,depth_colorize_max_depth)
            pred_resized = cv2.cvtColor(pred_resized, cv2.COLOR_RGB2BGR)
            pred_resized = cv2.resize(pred_resized, (320,240), interpolation = cv2.INTER_AREA)
            
            pred_resized_with_border = cv2.copyMakeBorder(pred_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

            img_list_row.append(pred_resized_with_border)  
                    
        elif  model == 'DecnetNLSPNsmall':
            net = DecnetNLSPNSmall(decnet_args)
            net.to(device)
            net.load_state_dict(torch.load('../weights/DecnetNLSPNsmall_best.pth', map_location=device))
            net.eval() 
            output = net(image, sparse)
            inv_pred = output['pred']
            pred = inverse_depth_norm(max_depth,inv_pred)
            pred_resized = upscale_to_full_resolution(pred).squeeze()
            pred_resized = np.squeeze(pred_resized.cpu().detach().numpy())
            
            depth_colorize_min_depth =np.min(pred_resized) #np.min(gt.cpu().detach().numpy())
            depth_colorize_max_depth = np.max(pred_resized)# np.max(gt.cpu().detach().numpy())
            
            pred_resized = visualizer.depth_colorize_fixed_ranges(pred_resized,depth_colorize_min_depth,depth_colorize_max_depth)
            pred_resized = cv2.cvtColor(pred_resized, cv2.COLOR_RGB2BGR)
            pred_resized = cv2.resize(pred_resized, (320,240), interpolation = cv2.INTER_AREA)
            
            pred_resized_with_border = cv2.copyMakeBorder(pred_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

            img_list_row.append(pred_resized_with_border)  
  
        elif  model == 'DecnetNLSPN':
            net = DecnetNLSPN(decnet_args)
            net.to(device)
            net.load_state_dict(torch.load('../weights/DecnetNLSPN_best.pth', map_location=device))
            net.eval()
            output = net(image, sparse)
            inv_pred = output['pred']
            pred = inverse_depth_norm(max_depth,inv_pred)
            pred_resized = upscale_to_full_resolution(pred).squeeze()
            pred_resized = np.squeeze(pred_resized.cpu().detach().numpy())
            
            depth_colorize_min_depth =np.min(pred_resized) #np.min(gt.cpu().detach().numpy())
            depth_colorize_max_depth = np.max(pred_resized)# np.max(gt.cpu().detach().numpy())
            
            pred_resized = visualizer.depth_colorize_fixed_ranges(pred_resized,depth_colorize_min_depth,depth_colorize_max_depth)
            pred_resized = cv2.cvtColor(pred_resized, cv2.COLOR_RGB2BGR)
            pred_resized = cv2.resize(pred_resized, (320,240), interpolation = cv2.INTER_AREA)
            
            pred_resized_with_border = cv2.copyMakeBorder(pred_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

            img_list_row.append(pred_resized_with_border)  

             
        else:
            pass# inv_pred = model(image, sparse)
    
    
    

    
    #print(error_map.shape)
    gt_vis_2_with_border = cv2.copyMakeBorder(gt_vis_2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img_list_row.append(gt_vis_2_with_border)  
    sample_id += 1
    
    
    if sample_id == 1:
        sample_1 = np.hstack(img_list_row)
    elif sample_id == 2:
        sample_2 = np.hstack(img_list_row)
    elif sample_id == 3:
        sample_3 = np.hstack(img_list_row)
    elif sample_id == 4:
        sample_4 = np.hstack(img_list_row) 

    
img_merge = np.vstack((sample_1,sample_2,sample_3,sample_4))
#cv2.imshow('queens', )
cv2.imwrite(f'all_model_results.png',img_merge.astype('uint8'))