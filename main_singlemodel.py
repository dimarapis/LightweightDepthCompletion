import os
import torch
import time
import wandb
import random
import warnings
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import features.CoordConv as CoordConv

#from nlspnconfig import args

from models.s2d import ResNet

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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset


from models.enet_pro import ENet
from models.enet_basic import weights_init
from models.guide_depth import GuideDepth
from features.decnet_sanity import np_min_max, torch_min_max
from features.decnet_args import decnet_args_parser
from features.decnet_sanity import inverse_depth_norm
from features.decnet_losscriteria import MaskedMSELoss, SiLogLoss, MaskedL1Loss
from features.decnet_dataloaders import DecnetDataloader
#from models.sparse_guided_depth import AuxSparseGuidedDepth, DecnetLateBase, DecnetNLSPN_sharedDecoder, DecnetNLSPNSmall, SparseGuidedDepth, DecnetModule, DecnetNLSPN, DecnetEarlyBase
#from models.sparse_guided_depth import SparseAndRGBGuidedDepth, RefinementModule, DecnetSparseIncorporated
from features.decnet_dataloaders import cspn_nyu_input_crop

#from models.nlspnmodel import NLSPNModel
from models.twise_model import MultiRes_network_avgpool_diffspatialsizes
from models.common import *
#from models.modulated_deform_conv_func import ModulatedDeformConvFunction
import torch
import torch.nn as nn



#Saving weights and log files locally
grabtime = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
if not os.path.exists(os.path.join('weights',grabtime)):
    os.makedirs(os.path.join('weights',grabtime))
else:
    os.mkdir(os.path.join('weights',grabtime+'_'+str(np.randint(0,10))))

#Finding were gradients became nans - DONT USE IT IN TRAINING AS IT SLOWS IT DOWN
#torch.autograd.set_detect_anomaly(True)

#Remove warning for visualization purposes (mostly due to behaviour of upsample block)
warnings.filterwarnings("ignore")

#Loading arguments and model options
print("\nSTEP 1. Loading arguments and parameters...")
decnet_args = decnet_args_parser()
#Print arguments and model options
converted_args_dict = vars(decnet_args)
print('\nParameters list: (Some may be redundant depending on the task, dataset and model chosen)')
print(converted_args_dict)

#Defining metrics and loggers
metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']

#Initialize weights and biases logger
if decnet_args.wandblogger == True:
    if decnet_args.wandbrunname != '':
        wandb.init(project=str(decnet_args.project),entity=str(decnet_args.entity),name=decnet_args.wandbrunname)
    else:
        wandb.init(project=str(decnet_args.project),entity=str(decnet_args.entity))#)="decnet-project", entity="wandbdimar")
    wandb.config.update(decnet_args)
    
def smooth2chandep(chan_deps, params = None, device = None):
    if device is None:
        device = torch.device("cpu")
    split_deps = torch.split(chan_deps, 1, 1)        
    split_deps = list(split_deps)

    alpha = torch.sigmoid(split_deps[2])
    final_dep = alpha*F.relu(split_deps[0])*params['depth_maxrange'] + (1 - alpha)*F.relu(split_deps[1])*params['depth_maxrange']

    return final_dep


#Ensuring reproducibility using seed
seed = 2910
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#Loading datasets
print("\nSTEP 2. Loading datasets...")
train_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.train_datalist, split='train'),batch_size=decnet_args.batch_size, shuffle=True)
eval_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.val_datalist, split='eval'),batch_size=1)

subset = False
if subset == True:
    len_dataset = list(range(10))#len(train_dl.dataset)))
    indices = random.sample(len_dataset,6)
    #indices = list(range(129))
    print(indices)
    #indices = list(range(len(train_dl.dataset)))
    #train_sampler = SubsetRandomSampler(indices)
    #rint(train_sampler)
    #random_index = random.randint(0,len(letters)-1)
    #train_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.train_datalist),batch_size=decnet_args.batch_size, sampler = train_sampler)
    #mini_batch_dl, valid_ds = torch.utils.data.random_split(train_dl.dataset, (129, len(train_dl.dataset)-129))
    train_dataset = Subset(train_dl, indices)
    print(train_dataset)
    print(f'Loaded {len(train_dataset.dataset)} training files')
    
 
    

print(f'Loaded {len(train_dl.dataset)} training files')
print(f'Loaded {len(eval_dl.dataset)} val files')

'''
#FIX - Need to load intrinsics in data loaders
#Constructing CoordConv and K matrix if needed for the model
transform_to_tensor = transforms.ToTensor()
pil_to_tensor = transforms.PILToTensor()
#NN_K
new_K = transform_to_tensor(np.array([[599.9778442382812, 0.0000, 318.6040344238281],
        [0.0000, 600.5001220703125, 247.7696533203125],
        [0.0000, 0.0000, 1.0000]])).to(dtype=torch.float32).to(device)
#KITTI_K
new_K = transform_to_tensor(np.array([[721.5377, 0.0, 596.5593],
        [0.0, 721.5377, 149.854],
        [0.0000, 0.0000, 1.0000]])).to(dtype=torch.float32).to(device)
position = CoordConv.AddCoordsNp(decnet_args.val_h, decnet_args.val_w)
position = position.call()
position = transform_to_tensor(position).unsqueeze(0).to(device)
#print(position.shape)
'''

#Loading model
print("\nSTEP 3. Loading model and metrics...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if decnet_args.networkmodel == "GuideDepth":
    #print(decnet_args.pretrained)
    model = GuideDepth(True)

    if decnet_args.pretrained == True:
        model.load_state_dict(torch.load('./weights/GuideDepth_full_original.pth', map_location=device))  
        #model.load_state_dict(torch.load('./weights/2022_08_21-10_23_53_PM/GuideDepth_99.pth', map_location='cpu'))  
        #2022_08_21-10_23_53_PM
      
elif decnet_args.networkmodel == "SparseGuidedDepth":
    model = SparseGuidedDepth(True)
    #if decnet_args.pretrained
        #model.load_state_dict(torch.load('./weights/guide.pth', map_location='cpu'))     
elif decnet_args.networkmodel == "SparseAndRGBGuidedDepth":
    model = SparseAndRGBGuidedDepth(False)
    #if decnet_args.pretrained
        #model.load_state_dict(torch.load('./weights/guide.pth', map_location='cpu'))     
elif decnet_args.networkmodel == "ENET2021":
    model = ENet(decnet_args)

elif decnet_args.networkmodel == "DecnetModule":
    model = DecnetSparseIncorporated()
        
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/DecnetModule_50k_500.pth', map_location=device), strict=False)

        #model.load_state_dict(torch.load('./weights/2022_08_21-10_10_22_PM/DecnetModule_99.pth', map_location='cpu'))#, strict=False)

elif decnet_args.networkmodel == "DecnetModuleSmall":
    #model = DecnetSparseIncorporated()
    model = DecnetSparseIncorporated(up_features=[32, 8, 4], inner_features=[32, 8, 4])    
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/2022_09_01-05_42_36_PM/DecnetModuleSmall_1.pth', map_location=device), strict=False)

        #model.load_state_dict(torch.load('./weights/2022_08_21-10_10_22_PM/DecnetModule_99.pth', map_location='cpu'))#, strict=False)

elif decnet_args.networkmodel == "AuxSparseGuidedDepth":
    model = GuideDepth(True)
    #0209refinement_model = RefinementModule()
    if decnet_args.pretrained == True:
        model.load_state_dict(torch.load('./weights/KITTI_Full_GuideDepth.pth', map_location='cpu'))  

elif decnet_args.networkmodel == "DecnetNLSPN":
    model = DecnetNLSPN(decnet_args)
        
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/2022_08_29-10_04_58_AM/DecnetNLSPN_2.pth', map_location=device),strict=False)
        
    #if decnet_args.pretrained == True:
    #    #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
    #    model.load_state_dict(torch.load('./weights/DecnetModule_50k_500.pth', map_location=device), strict=False)

        #model.load_state_dict(torch.load('./weights/2022_08_21-10_10_22_PM/DecnetModule_99.pth', map_location='cpu'))#, strict=False)

elif decnet_args.networkmodel == "DecnetNLSPNSmall":
    model = DecnetNLSPNSmall(decnet_args)
        
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/2022_08_30-02_53_53_AM/DecnetNLSPNSmall_8.pth', map_location=device))
        
    #if decnet_args.pretrained == True:
    #    #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
    #    model.load_state_dict(torch.load('./weights/DecnetModule_50k_500.pth', map_location=device), strict=False)

        #model.load_state_dict(torch.load('./weights/2022_08_21-10_10_22_PM/DecnetModule_99.pth', map_location='cpu'))#, strict=False)
elif decnet_args.networkmodel == "DecnetNLSPN_decoshared":
    model = DecnetNLSPN_sharedDecoder(decnet_args)
        
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/2022_08_29-07_15_25_PM/DecnetNLSPN_decoshared_6.pth', map_location=device))
        
    #if decnet_args.pretrained == True:
    #    #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
    #    model.load_state_dict(torch.load('./weights/DecnetModule_50k_500.pth', map_location=device), strict=False)

        #model.load_state_dict(torch.load('./weights/2022_08_21-10_10_22_PM/DecnetModule_99.pth', map_location='cpu'))#, strict=False)
        #model.load_state_dict(torch.load('./weights/2022_08_21-10_10_22_PM/DecnetModule_99.pth', map_location='cpu'))#, strict=False)
elif decnet_args.networkmodel == "DecnetLateBase":
    model = DecnetLateBase(decnet_args)
    #model.load_state_dict(torch.load('./weights/2022_08_29-10_04_58_AM/DecnetNLSPN_2.pth', map_location=device),strict=False)
elif decnet_args.networkmodel == "DecnetEarlyBase":
    model = DecnetEarlyBase(decnet_args)   
    
elif decnet_args.networkmodel == "nlspn":
    print("here")
    from features.all_args import all_args_parser
    all_args = all_args_parser()
    model = NLSPNModel(all_args)

elif decnet_args.networkmodel == 's2d':
    model = ResNet(layers=50, decoder='deconv2', output_size=(240,320),
    in_channels=4, pretrained=False)
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/2022_09_07-02_25_40_AM/s2d_9.pth', map_location=device))
    
elif decnet_args.networkmodel == 'cspn':
    import models.torch_resnet_cspn_nyu as cspn_model
    cspn_config = {'step': 24, 'norm_type': '8sum'}
    model = cspn_model.resnet50(pretrained = True,
                cspn_config=cspn_config)
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/2022_09_07-01_22_09_AM/cspn_1.pth', map_location=device))
    
elif decnet_args.networkmodel == 'twise':
    model = MultiRes_network_avgpool_diffspatialsizes()

    
else:
    print("Can't seem to find the model configuration. Make sure you choose a model by --network-model argument. Integrated options are: [GuideDepth,SparseGuidedDepth,SparseAndRGBGuidedDepth,ENET2021]") 

model.to(device)
#0209refinement_model.to(device)

'''
# Calculating macs and parameters of model to assess how heavy the model is
rgb_shape = torch.randn(1, 3, decnet_args.train_height, decnet_args.train_width).to(device)
d_shape = torch.randn(1, 1, decnet_args.train_height, decnet_args.train_width).to(device)
macs, params = profile(model, inputs=(rgb_shape, ))
macs, params = clever_format([macs, params], "%.3f")
print(f'model macs: {macs} and params: {params}')
if decnet_args.wandblogger == True:
    wandb.config.update({"macs": macs,  "params": params})
'''


'''
#Convering model to tensorrt
if decnet_args.torch_mode == 'tensorrt':
    from torch2trt import torch2trt
    model.eval()GuideDepth
    model = model_trt
'''


optimizer = optim.Adam(model.parameters(), lr=decnet_args.learning_rate)#, eps=1e-3, amsgrad=True)#, momentum=0.9) 
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[15,20,25], gamma=0.1)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer,20,gamma=0.1)

tabulator_args = []

for key in converted_args_dict:
    tabulator_args.append([key,converted_args_dict[key]]) 

with open("txt_logging/"+grabtime+".txt", "a") as txt_log:
#Printing args for checking
    txt_log.write(tabulate(tabulator_args, tablefmt='orgtbl'))
    #txt_log.write('\nScheduler settings: ' + str(lr_scheduler.state_dict()))


#lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

depth_criterion = MaskedL1Loss()
#depth_criterion = SiLogLoss()

#defining index (epoch)
epoch = 0
prev_loss = 0.0
#print(f"Loaded model {converted_args_dict['networkmodel']}'# for {converted_args_dict['task']}")



def print_torch_min_max_rgbpredgt(rgb,pred,gt):
    print('\n')
    print(f'torch_min_max rgb {torch_min_max(rgb)}')
    print(f'torch_min_max pred {torch_min_max(pred)}')
    print(f'torch_min_max gt {torch_min_max(gt)}')
    print('\n')
    
def print_torch_min_max_rgbsparsepredgt(rgb,sparse,pred,gt):
    print('\n')
    print(f'torch_min_max rgb {torch_min_max(rgb)}')
    print(f'torch_min_max sparse {torch_min_max(sparse)}')
    print(f'torch_min_max pred {torch_min_max(pred)}')
    print(f'torch_min_max gt {torch_min_max(gt)}')
    print('\n')
    
def unpack_and_move(data):
    if isinstance(data, (tuple, list)):
        #print('here1')
        image = data[0].to(device, non_blocking=True)
        gt = data[1].to(device, non_blocking=True)
        sparse = data[2].to(device, non_blocking=True)
        return image, gt, sparse
    if isinstance(data, dict):
        #print('here2')
        keys = data.keys()
        image = data['image'].to(device, non_blocking=True)
        gt = data['gt'].to(device, non_blocking=True)
        sparse = data['sparse'].to(device, non_blocking=True)
        return image, gt, sparse
    print('Type not supported')


#Iterate images  
print("\nSTEP 4. Training or eval stage...")
'''
def metric_block(pred,gt,metric_name,decnet_args):
    model.eval()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

        pred_d, depth_gt, = pred[i].squeeze(), gt[i].squeeze()#, data['d'].squeeze()# / 1000.0
        pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)
        computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)

        for metric in metric_name:
            result_metrics[metric] += computed_result[metric]        
    
    #calculating total metrics by averaging  
    for metric in metric_name:
        result_metrics[metric] = result_metrics[metric] / float((i+1))
    print(f'batch average {float(i+1)}')
    # printing result 
    print("Results:")
    for key in result_metrics:
        print(key, ' = ', result_metrics[key])
'''
if decnet_args.dataset == 'nyuv2':
    
    upscale_to_full_resolution = torchvision.transforms.Resize((480,640))
elif decnet_args.dataset == 'nn':

    upscale_to_full_resolution = torchvision.transforms.Resize((360,640))
    

def evaluation_block(epoch):
    print(f"\nSTEP. Testing block... Epoch no: {epoch}")
    torch.cuda.empty_cache()
    model.eval()
    #0209refinement_model.eval()
    global best_rmse
    if epoch == 0:
        best_rmse = np.inf

    eval_loss = 0.0
    refined_eval_loss = 0.0
    average_meter = AverageMeter()

    result_metrics = {}
    flipped_result_metrics = {}
    refined_result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
        flipped_result_metrics[metric] = 0.0
        refined_result_metrics[metric] = 0.0

    with torch.no_grad():
        t0 = time.time()
        for i, data in enumerate(tqdm(eval_dl)):
            
            #model.eval()
            image_filename = data['file']
            #print(image_filename)
            image, gt, sparse = data['rgb'], data['gt'], data['d']#.permute(0,2,3,1), data['gt'], data['d']
            
            if decnet_args.dataset == 'nyuv2':
                #image = image * 255
                #print(f'before {torch_min_max(sparse)}')
                #sparse = sparse/100.
                #print(f'after {torch_min_max(sparse)}')
                #gt = gt/100.
               
               
                gt = gt * 0.001
                sparse = sparse * 0.001
                
                #sparse = sparse /255.0 * 10.0 #From 8bit to range [0, 10] (meter)
                #gt = gt /255.0 * 10.0
                
                max_depth = 10
                
            
            elif decnet_args.dataset == 'nn':
                max_depth = 80
                

            if decnet_args.networkmodel == 'GuideDepth':
                inv_pred = model(image)
            elif decnet_args.networkmodel == 'DecnetNLSPN' or decnet_args.networkmodel == 'DecnetNLSPN_decoshared' \
                or decnet_args.networkmodel == 'DecnetNLSPNSmall' or decnet_args.networkmodel == 'nlspn':
            
                output = model(image, sparse)
                #print(output)
                inv_pred = output['pred']
                
            elif decnet_args.networkmodel == 's2d':
                input = torch.cat((image,sparse),dim=1)
                inv_pred = model(input) 
            elif decnet_args.networkmodel == 'cspn':
                input = torch.cat((image,sparse),dim=1)
                resized_input = cspn_nyu_input_crop(input)
                #print(input.shape)
                inv_pred = model(resized_input) 
                gt = cspn_nyu_input_crop(gt)

            elif decnet_args.networkmodel == 'twise':
                valpred_dc,_,_ = model(sparse,image)
                device = 'cuda'
                params = {'depth_maxrange': 10.0,                                                
                    'threshold': 100, 
                    }
                pred_datavalid = smooth2chandep(valpred_dc, params = params, device = device)


                #print(pred_datavalid.shape)
                #print()   
                
            else:    
            #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
                inv_pred = model(image, sparse)
                

            #inv_pred = model(image)
            #print(inv_pred)
            #print(pred.shape,image.shape)
            if decnet_args.networkmodel != 'twise':
                pred = inverse_depth_norm(max_depth,inv_pred)
            else:
                pred = pred_datavalid#.unsqueeze(dim=0)
                
            #print(pred.shape)
            #print(f'pred {torch_min_max(pred)}')
            #print(f'gt {torch_min_max(gt)}')
            #print(f'eval pred  {pred.shape} and image shapes {image.shape}')
            
            #print(f'pred {torch_min_max(pred)}')
            #print_torch_min_max_rgbsparsepredgt(image, sparse, pred, gt)   
            #print(image_filename)         
            #ipnut = input()

            #print_torch_min_max_rgbpredgt(image,pred,gt) 
            #            



            loss = depth_criterion(pred, gt)
            #print(loss.item())
            #print(torch_min_max(pred))
            #pred = 
            #0209refined_inv_pred = refinement_model(inv_pred,sparse)
            #0209refined_pred = inverse_depth_norm(decnet_args.max_depth_eval,refined_inv_pred)   

            #print_torch_min_max_rgbpredgt(image,pred,gt)            
            
            eval_loss += loss.item()
            #print(loss)
            #0209refined_loss = depth_criterion(refined_pred,gt)
            #0209refined_eval_loss += refined_loss.item()

            #upscaling depth to compare (if needed)
            #upscale_depth = transforms.Resize(gt.shape[-2:]) #To GT res
            #prediction = upscale_depth(pred)

            
            
            #print('half_reso',pred.shape,gt.shape)
            pred_resized, gt_resized = upscale_to_full_resolution(pred).squeeze(), upscale_to_full_resolution(gt).squeeze()
            #print('full_reso',pred_resized.shape,gt_resized.shape)

            
            pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_resized, gt_resized)    
            computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)

            flipped_evaluation = True
            if flipped_evaluation:
                image_flip = torch.flip(image, [3])
                gt_flip = torch.flip(gt, [3])
                sparse_flip = torch.flip(sparse, [3])

                if decnet_args.networkmodel == 'GuideDepth':
                    flipped_inv_pred = model(image_flip)
                elif decnet_args.networkmodel == 'DecnetNLSPN' or decnet_args.networkmodel == 'DecnetNLSPN_decoshared'\
                    or decnet_args.networkmodel == 'DecnetNLSPNSmall' or decnet_args.networkmodel == 'nlspn':
                    output = model(image_flip, sparse_flip)
                    flipped_inv_pred = output['pred']
                elif decnet_args.networkmodel == 's2d':
                    flipped_input = torch.cat((image_flip,sparse_flip),dim=1)
                    flipped_inv_pred = model(flipped_input)

                elif decnet_args.networkmodel == 'cspn':
                    flipped_input = torch.cat((image_flip,sparse_flip),dim=1)
                    resized_flipped_input = cspn_nyu_input_crop(flipped_input)
                    flipped_inv_pred = model(resized_flipped_input)
                    gt_flip = cspn_nyu_input_crop(gt_flip)
                    
                elif decnet_args.networkmodel == 'twise':

                    flipped_valpred_dc,_,_  = model(sparse_flip,image_flip)
                    device = 'cuda'
                    params = {'depth_maxrange': 10.0,                                                
                            'threshold': 100, 
                            }
                    flipped_pred_datavalid = smooth2chandep(flipped_valpred_dc, params = params, device = device)
                
                else:    
                #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
                    flipped_inv_pred = model(image_flip, sparse_flip)
                
                if decnet_args.networkmodel != 'twise':
                    flipped_pred = inverse_depth_norm(max_depth,flipped_inv_pred)

                else:
                
                    flipped_pred = flipped_pred_datavalid
                    #print(f'pred {torch_min_max(pred)}')
                #print_torch_min_max_rgbsparsepredgt(image, sparse, pred, gt)   
                #print(image_filename)         
                #ipnut = input()

                flipped_pred_d, flipped_depth_gt, = flipped_pred.squeeze(), gt_flip.squeeze()#, data['d'].squeeze()# / 1000.0
                flipped_pred_crop, flipped_gt_crop = custom_metrics.cropping_img(decnet_args, flipped_pred_d, flipped_depth_gt)    
                flipped_computed_result = custom_metrics.eval_depth(flipped_pred_crop, flipped_gt_crop)

            #0209refined_pred_d, refined_depth_gt, = refined_pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            #0209refined_pred_crop, refined_gt_crop = custom_metrics.cropping_img(decnet_args, refined_pred_d, refined_depth_gt)    
            #0209refined_computed_result = custom_metrics.eval_depth(refined_pred_crop, refined_gt_crop)


                for metric in metric_name:
                    result_metrics[metric] += computed_result[metric]
                    flipped_result_metrics[metric] += flipped_computed_result[metric]
                    #0209refined_result_metrics[metric] += refined_computed_result[metric]

            else:
                for metric in metric_name:
                    result_metrics[metric] += computed_result[metric]
                    #0209refined_result_metrics[metric] += refined_computed_result[metric]

            #result = Result()
            #result.evaluate(pred_d.data, depth_gt.data)
            #average_meter.update(result, gpu_time, data_time, image.size(0))
        

        #avg = average_meter.average()
        #print('\n*\n'
        #    'RMSE={average.rmse:.3f}\n'
        #    'MAE={average.mae:.3f}\n'
        #    'Delta1={average.delta1:.3f}\n'
        #    'Delta2={average.delta2:.3f}\n'
        #    'Delta3={average.delta3:.3f}\n'
        #    'REL={average.absrel:.3f}\n'
        #    'Lg10={average.lg10:.3f}\n'
        #    't_GPU={time:.3f}\n'.format(
        #    average=avg, time=avg.gpu_time))

            
        average_loss = eval_loss / (len(eval_dl.dataset) + 1)
        print(f'Evaluation Loss: {average_loss}')  
        print(f'Learning rate: {optimizer}')  
        #VISUALIZE BLOCK
        #Saving depth prediciton data along with original image
        #visualizer.save_depth_prediction(prediction,data['rgb']*255)

        #Showing plots, results original image, etc
        #visualizer.plotter(pred_d,sparse_depth,depth_gt,pred,data['rgb'])
            
        #calculating total metrics by averaging  
        print(i+1)
        for metric in metric_name:
            result_metrics[metric] = result_metrics[metric] / float((i+1))

            if flipped_evaluation:
                flipped_result_metrics[metric] = flipped_result_metrics[metric] / float((i+1))


        tabulator, flipped_tabulator = [],[]
        for key in result_metrics:
            tabulator.append([key,result_metrics[key]]) 
            flipped_tabulator.append([key, flipped_result_metrics[key]])

        if epoch == decnet_args.epochs:
            print(f"Results on epoch: {epoch}")
            print("Base model results")
            print(tabulate(tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished evaluation block")
            print("Flipped model results")
            print(tabulate(flipped_tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished training..")
            print(f"Average time for parsing images {time.time - t0}")


        else:
            print(f"Results on epoch: {epoch}")
            print("Base model results")
            print(tabulate(tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished evaluation block")
            print("Flipped model results")
            print(tabulate(flipped_tabulator, tablefmt='orgtbl'))
            #0209print("Refined model results")
            #0209print(tabulate(refined_tabulator, tablefmt='orgtbl'))
            #0209print(f"\n\nFinished training..")
            print(f"Average time for parsing images {time.time() - t0}")
            if result_metrics['rmse'] < best_rmse:
                best_rmse = result_metrics['rmse']
                #remove all previous weights to save space
                filelist = [ f for f in os.listdir(os.path.join('weights',grabtime)) if f.endswith(".pth") ]
                for f in filelist:
                    os.remove(os.path.join((os.path.join('weights',grabtime)), f))
                

                path = f"weights/{grabtime}/{decnet_args.networkmodel}_{epoch}.pth"
                torch.save(model.state_dict(), path)
                with open("txt_logging/"+grabtime+".txt", "a") as txt_log:
                # Append 'hello' at the end of file
                #file_object.write("hello")
                    txt_log.write(f'\n\nNew model saved: {path} \n')
                    txt_log.write(tabulate(tabulator, tablefmt='orgtbl'))
                print(f"\nSaved model and logfile {path} with last rmse {best_rmse}")
            
        
    if decnet_args.wandblogger == True:
        #if epoch != 0:
            #epoch = epoch[1]
        wandb.log(result_metrics, step = epoch)
        #Wandb save sample image
        #0209wandb_image, wandb_depth_colorized, wandb_refined_depth_colorized = visualizer.wandb_image_prep(image, pred, refined_pred) 
        #0209wandb.log({"Samples": [wandb.Image(wandb_image,caption="RGB sample"), wandb.Image(wandb_depth_colorized, caption="Colorized base prediction"), wandb.Image(wandb_refined_depth_colorized, caption="Colorized refined prediction")]},step = epoch)
        wandb_image, wandb_depth_colorized, wandb_gt_colorized = visualizer.wandb_image_prep(image, pred, gt) 
        wandb.log({"Samples": [wandb.Image(wandb_image,caption="RGB sample"), wandb.Image(wandb_depth_colorized, caption="Prediction"), wandb.Image(wandb_gt_colorized, caption="Groundtruth")]},step = epoch)
    
    #model.train()




def training_block(model):
    
    print("\nSTEP. Training block...")
    global best_rmse
    best_rmse = np.inf

    #for epoch in enumerate(tqdm(range(1,int(decnet_args.epochs)+1))):
    for epoch in range(1,int(decnet_args.epochs)+1):
        
        iteration = 0
        model.train()
        #0209refinement_model.train()
        #for param in model.feature_extractor.parameters():
           #param.requires_grad = False

        epoch_loss = 0.0

        for data in train_dl:
            image_filename = data['file']

            iteration += 1
            image, gt, sparse = data['rgb'], data['gt'], data['d']#.permute(0,2,3,1), data['gt'], data['d']
           
            if decnet_args.dataset == 'nyuv2':
                #image = image * 255
                #print(f'before {torch_min_max(sparse)}')
                #sparse = sparse/100.
                #print(f'after {torch_min_max(sparse)}')
                #gt = gt/100.
                
                sparse = sparse /255.0 * 10.0 #From 8bit to range [0, 10] (meter)
                gt = gt /255.0 * 10.0

                max_depth = 10
                
            elif decnet_args.dataset == 'nn':
                max_depth = 80
                

            if decnet_args.networkmodel == 'GuideDepth':
                inv_pred = model(image)
            elif decnet_args.networkmodel == 'DecnetNLSPN' or decnet_args.networkmodel == 'DecnetNLSPN_decoshared' \
                or decnet_args.networkmodel == 'DecnetNLSPNSmall' or decnet_args.networkmodel == 'nlspn':
                output = model(image, sparse)
                inv_pred = output['pred']
                
                            
            elif decnet_args.networkmodel == 's2d':
                input = torch.cat((image,sparse),dim=1)
                inv_pred = model(input) 
            elif decnet_args.networkmodel == 'cspn':
                input = torch.cat((image,sparse),dim=1)
                resized_input = cspn_nyu_input_crop(input)
                #print(input.shape)
                inv_pred = model(resized_input) 
                gt = cspn_nyu_input_crop(gt)

                #print()
            elif decnet_args.networkmodel == 'twise':
                valpred_dc,_,_ = model(sparse,image)
                device = 'cuda'
                params = {'depth_maxrange': 10.0,                                                
                    'threshold': 100, 
                    }
                pred_datavalid = smooth2chandep(valpred_dc, params = params, device = device)


                #print(pred_datavalid.shape)
                #print()   
                
            else:    
            #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
                inv_pred = model(image, sparse)
                

            #inv_pred = model(image)
            #print(inv_pred)
            #print(pred.shape,image.shape)
            if decnet_args.networkmodel != 'twise':
                pred = inverse_depth_norm(max_depth,inv_pred)
            else:
                pred = pred_datavalid#.unsqueeze(dim=0)
            #print(f'train pred  {pred.shape} and image shapes {image.shape}')
            
        
            #print(pred.shape,image.shape)
            
            #inv_pred = model(image)        
            #0209refined_inv_pred = refinement_model(inv_pred,sparse)
            #0209refined_pred = inverse_depth_norm(decnet_args.max_depth_eval,refined_inv_pred)            
            #print(f'pred {torch_min_max(pred)}')
            #print(f'gt {torch_min_max(gt)}')
            

            #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
            #pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            #print(f'pred {torch_min_max(pred)}')
            #print_torch_min_max_rgbpredgt(image,  pred, gt)            
            #print_torch_min_max_rgbsparsepredgt(image[0], sparse[0], pred[0], gt[0])            
            #print(image_filename[0])
            #ipnut = input()
            
            #print_torch_min_max_rgbpredgt(image,pred,gt)            
            
            loss = depth_criterion(pred, gt)
            #print(loss.item())
            
            #0209refined_loss = depth_criterion(refined_pred,gt)

            a = list(model.parameters())[0].clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            #print(a == b)
            epoch_loss += loss.item() 
            #print(loss.item())
            if iteration == 1 or iteration % 100 == 0:

                print(f'Iteration {iteration} out of {int(np.ceil(len(train_dl.dataset) / decnet_args.batch_size))}. Loss: {loss.item()}')
            

        average_loss = epoch_loss / (len(train_dl.dataset) / decnet_args.batch_size)
        print(f'Training Loss: {average_loss}. Epoch {epoch} of {decnet_args.epochs}')
        lr_scheduler.step()

        evaluation_block(epoch)
        
if converted_args_dict['mode'] == 'eval':
    #pass
    epoch = 0
    evaluation_block(epoch)
elif converted_args_dict['mode'] == 'train':
    epoch = 0
    evaluation_block(epoch)
    training_block(model)
    #evaluation_block()
    