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
from torch.utils.data import Subset

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

#Saving weights and log files locally
grabtime = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
os.mkdir(os.path.join('weights',grabtime))

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
train_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.train_datalist),batch_size=decnet_args.batch_size)
eval_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.val_datalist),batch_size=1)

#if decnet_args.training_subset != 0:
    #train1_dl = DecnetDataloader(decnet_args,decnet_args.train_datalist)
    #test_mask = np.random.choice(len(train1_dl), decnet_args.training_subset, replace=False)
    #train2_dl = train1_dl[test_mask]
    #train3_dl = DataLoader(train2_dl,batch_size=decnet_args.batch_size)
    #train_dl = train3_dl

    #indices = np.arange(len(DecnetDataloader(decnet_args,decnet_args.train_datalist)))
    #print(indices)
    #np.random.shuffle(indices)
    #print(indices)
    #print(np.shuff)
    #train_indices = indices[:decnet_args.training_subset]
    #print(len(train_indices))
    ## Warp into Subsets and DataLoaders
    #train_dataset = Subset(train_dl, train_indices)
    ##print(train_dl.dataset)
    ##test_dataset = Subset(dataset, test_indices)

    #print(f'Loaded {len(train_dataset.dataset)} training files')

    #train_dl = DataLoader(train_dataset,batch_size=decnet_args.batch_size)


print(f'Loaded {len(train_dl.dataset)} training files')
print(f'Loaded {len(eval_dl.dataset)} val files')

#Loading model
print("\nSTEP 3. Loading model and metrics...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if decnet_args.network_model == "GuideDepth":
    #print(decnet_args.pretrained)
    model = GuideDepth(True)

    if decnet_args.pretrained == True:
        model.load_state_dict(torch.load('./weights/KITTI_Full_GuideDepth.pth', map_location='cpu'))  
      
elif decnet_args.network_model == "SparseGuidedDepth":
    model = SparseGuidedDepth(True)
    #if decnet_args.pretrained
        #model.load_state_dict(torch.load('./weights/guide.pth', map_location='cpu'))     
elif decnet_args.network_model == "SparseAndRGBGuidedDepth":
    model = SparseAndRGBGuidedDepth(False)
    #if decnet_args.pretrained
        #model.load_state_dict(torch.load('./weights/guide.pth', map_location='cpu'))     
elif decnet_args.network_model == "ENET2021":
    model = ENet(decnet_args)

elif decnet_args.network_model == "AuxSparseGuidedDepth":
    model = RgbGuideDepth(True)
    refinement_model = DepthRefinement()
    #refinement_model = RefinementModule()
    #refinement_model = Scaler()
    if decnet_args.pretrained == True:
        model.load_state_dict(torch.load('./weights/NYU_Full_GuideDepth.pth', map_location='cpu'))
        #model.load_state_dict(torch.load('./weights/2022_07_11-11_31_51_PM/AuxSparseGuidedDepth_26.pth', map_location='cuda'))
        #model.load_state_dict(torch.load('./weights/2022_07_06-10_06_37_AM/AuxSparseGuidedDepth_99.pth', map_location='cuda'), strict=False)
        #refinement_model.load_state_dict(torch.load('./weights/2022_07_11-11_31_51_PM/AuxSparseGuidedDepth_26_ref.pth', map_location='cuda'))
        

else:
    print("Can't seem to find the model configuration. Make sure you choose a model by --network-model argument. Integrated options are: [GuideDepth,SparseGuidedDepth,SparseAndRGBGuidedDepth,ENET2021]") 
    
#model = torch.nn.DataParallel(model)
#refinement_model = torch.nn.DataParallel(refinement_model)

model.to(device)
refinement_model.to(device)



#for module in model.modules():
#    module.parameters().requires_grad = False 

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

optimizer = optim.Adam(model.parameters(), lr=decnet_args.learning_rate, eps=1e-3, amsgrad=True)#, momentum=0.9) 
refinement_optimizer = optim.Adam(refinement_model.parameters(), lr=decnet_args.learning_rate, eps=1e-3, amsgrad=True)#, momentum=0.9) 
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,50,75,90], gamma=0.1)
refinement_lr_scheduler = optim.lr_scheduler.MultiStepLR(refinement_optimizer,milestones=[30,50,75,90], gamma=0.1)


tabulator_args = []

for key in converted_args_dict:
    tabulator_args.append([key,converted_args_dict[key]]) 

with open("txt_logging/"+grabtime+".txt", "a") as txt_log:
#Printing args for checking
    txt_log.write(tabulate(tabulator_args, tablefmt='orgtbl'))

depth_criterion = MaskedMSELoss()
#depth_criterion = SiLogLoss()

#defining index (epoch)
epoch = 0
prev_loss = 0.0
#print(f"Loaded model {converted_args_dict['network_model']}'# for {converted_args_dict['task']}")
'''
to_tensor_test = custom_transforms.ToTensor(test=True, maxDepth=80.0)
to_tensor = custom_transforms.ToTensor(test=False, maxDepth=80.0)

downscale_image = transforms.Resize((384,1280)) #To Model resolution


def print_torch_min_max_rgbpredgt(refined_pred,pred,gt,sparse):
    print('\n')
    print(f'torch_min_max refined_pred {torch_min_max(refined_pred)}')
    print(f'torch_min_max pred {torch_min_max(pred)}')
    print(f'torch_min_max gt {torch_min_max(gt)}')
    print(f'torch_min_max sparse {torch_min_max(sparse)}')
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

'''
#Iterate images  
print("\nSTEP 4. Training or eval stage...")


def evaluation_block(epoch):
    print(f"\nSTEP. Testing block... Epoch no: {epoch}")
    torch.cuda.empty_cache()
    model.eval()
    refinement_model.eval()
    global best_rmse
    #random_save_image = np.random.randint(0,len(eval_dl.dataset)-1)
    random_save_image = 29
    #print('random_save_image', random_save_image)
    eval_loss = 0.0
    refined_eval_loss = 0.0
    average_meter = AverageMeter()

    result_metrics = {}
    refined_result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
        refined_result_metrics[metric] = 0.0

    with torch.no_grad():
        t0 = time.time()
        for i, data in enumerate(tqdm(eval_dl)):
            
            #model.eval()
            image_filename = data['file']
            #print(image_filename)
            image, gt, sparse = data['rgb'], data['gt'], data['d']#.permute(0,2,3,1), data['gt'], data['d']
            #feat1, feat2, inv_pred = model(image, sparse)
            rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)

            #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            #feature_s1, feature_s2, coarse_depth, sparse

            #print(rgb_half.shape)
            #print(y_half.shape)
            #print(sparse_half.shape)
            #print(y.shape)

            refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)
            #refined_pred = pred
            #refined_pred = refinement_model(pred,sparse)


            loss = depth_criterion(pred, gt)
            eval_loss += loss.item()

            refined_loss = depth_criterion(refined_pred,gt)
            refined_eval_loss += refined_loss.item()

            #upscaling depth to compare (if needed)
            #upscale_depth = transforms.Resize(gt.shape[-2:]) #To GT res
            #prediction = upscale_depth(pred)

            pred_d, depth_gt = pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)    
            computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)
            #print(f'computer_result {computed_result}')

            refined_pred_d, refined_depth_gt = refined_pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            refined_pred_crop, refined_gt_crop = custom_metrics.cropping_img(decnet_args, refined_pred_d, refined_depth_gt)    
            refined_computed_result = custom_metrics.eval_depth(refined_pred_crop, refined_gt_crop)
            #print(f'refined_moufa {refined_computed_result}')

            #print(computed_result)

            #print(f'total length {len(eval_dl.dataset)}')
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
                refined_result_metrics[metric] += refined_computed_result[metric]
            #print(f'result_metrics[rmse] {result_metrics["rmse"]}')
            #print(f'refined_result_metrics[rmse] {refined_result_metrics["rmse"]}')

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
            #print(i,type(i))
            if i == random_save_image:
                #print(i,random_save_image)
                if epoch != 0:
                    temp_step = epoch[1]
                else:
                    temp_step = 0
                wandb_image, wandb_depth_colorized, wandb_refined_depth_colorized = visualizer.wandb_image_prep_refined(image, pred, refined_pred) 
                wandb.log({"Sample 1": [wandb.Image(wandb_image,caption="RGB sample"), wandb.Image(wandb_depth_colorized, caption="Colorized base prediction"), wandb.Image(wandb_refined_depth_colorized, caption="Colorized refined prediction")]},step = temp_step)
        #print(len(eval_dl.dataset))
        average_loss = eval_loss / len(eval_dl.dataset)
        refined_average_loss = refined_eval_loss / len(eval_dl.dataset)

        print(f'Evaluation Loss: {average_loss}. Refined evaluation Loss: {refined_average_loss}')    
        #VISUALIZE BLOCK
        #Saving depth prediciton data along with original image
        #visualizer.save_depth_prediction(prediction,data['rgb']*255)

        #Showing plots, results original image, etc
        #visualizer.plotter(pred_d,sparse_depth,depth_gt,pred,data['rgb'])
            
        #calculating total metrics by averaging  
        for metric in metric_name:
            result_metrics[metric] = result_metrics[metric] / len(eval_dl.dataset)
            refined_result_metrics[metric] = refined_result_metrics[metric] / len(eval_dl.dataset)
        #print(f'average result_metrics[rmse] {result_metrics["rmse"]}')
        #print(f'average refined_result_metrics[rmse] {refined_result_metrics["rmse"]}')

        #print(refined_result_metrics['rmse'])


        tabulator, refined_tabulator = [],[]
        for key in result_metrics:
            tabulator.append([key,result_metrics[key]]) 
            refined_tabulator.append([key, refined_result_metrics[key]])

        if epoch == 0:
            print(f"Results on epoch 0:")
            print("Base model results")
            print(tabulate(tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished evaluation block")
            print("Refined model results")
            print(tabulate(refined_tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished training..")
            print(f"Average time for parsing images {time.time() - t0}")
            pass

        elif epoch[1] == decnet_args.epochs:
            print(f"Results on epoch: {epoch[1]}")
            print("Base model results")
            print(tabulate(tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished evaluation block")
            print("Refined model results")
            print(tabulate(refined_tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished training..")
            print(f"Average time for parsing images {time.time() - t0}")


        else:
            print(f"Results on epoch: {epoch[1]}")
            print("Base model results")
            print(tabulate(tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished evaluation block")
            print("Refined model results")
            print(tabulate(refined_tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished training..")
            print(f"Average time for parsing images {time.time() - t0}")
            if refined_result_metrics['rmse'] < best_rmse:
                best_rmse = refined_result_metrics['rmse']
                #remove all previous weights to save space
                filelist = [ f for f in os.listdir(os.path.join('weights',grabtime)) if f.endswith(".pth") ]
                for f in filelist:
                    os.remove(os.path.join((os.path.join('weights',grabtime)), f))
                

                path = f"weights/{grabtime}/{decnet_args.network_model}_{epoch[1]}.pth"
                path_ref = f"weights/{grabtime}/{decnet_args.network_model}_{epoch[1]}_ref.pth"
                torch.save(model.state_dict(), path)
                torch.save(refinement_model.state_dict(), path_ref)
                with open("txt_logging/"+grabtime+".txt", "a") as txt_log:
                # Append 'hello' at the end of file
                #file_object.write("hello")
                    txt_log.write(f'\n\nNew model saved: {path} \n')
                    txt_log.write(tabulate(tabulator, tablefmt='orgtbl'))
                print(f"\nSaved model and logfile {path} with last rmse {best_rmse}")
            
        
    if decnet_args.wandblogger == True:
        if epoch != 0:
            temp_step = epoch[1]
        else:
            temp_step = 0
        for key in result_metrics:
            refined_result_metrics[str('refined_'+str(key))] = refined_result_metrics.pop(key)
        wandb.log(result_metrics, step = temp_step)
        wandb.log(refined_result_metrics, step = temp_step)

        #Wandb save sample image
        wandb_image, wandb_depth_colorized, wandb_refined_depth_colorized = visualizer.wandb_image_prep_refined(image, pred, refined_pred) 
        wandb.log({"Sample 2": [wandb.Image(wandb_image,caption="RGB sample"), wandb.Image(wandb_depth_colorized, caption="Colorized base prediction"), wandb.Image(wandb_refined_depth_colorized, caption="Colorized refined prediction")]},step = temp_step)
        #wandb_image, wandb_depth_colorized = visualizer.wandb_image_prep(image, pred) 
        #wandb.log({"Samples": [wandb.Image(wandb_image,caption="RGB sample"), wandb.Image(wandb_depth_colorized, caption="Colorized base prediction")]},step = epoch)
    
    #model.train()




def training_block(model):
    
    print("\nSTEP. Training block...")
    global best_rmse
    best_rmse = np.inf


    for epoch in enumerate(tqdm(range(1,int(decnet_args.epochs)+1))):
        #model.train()

        #model.train()
        optimizer.zero_grad()
        refinement_optimizer.zero_grad()
        #model.train()
        for param in model.parameters():
            param.requires_grad = False
        #model.train()
        #refinement_model.train()

            #print(param)

        epoch_loss = 0.0
        refined_epoch_loss = 0.0

        #data = next(iter(train_dl))
        for i, data in enumerate(tqdm(train_dl)):

        #EDW MESA
            image, gt, sparse = data['rgb'], data['gt'], data['d']#.permute(0,2,3,1), data['gt'], data['d']

            #feat1, feat2, inv_pred = model(image, sparse)
            #inv_pred = model(image)
            rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)

            #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            #feature_s1, feature_s2, coarse_depth, sparse
            #print(torch_min_max(pred))
            #print(torch_min_max(sparse))


            #refined_pred = pred
            
            #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
            #pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            #print_torch_min_max_rgbpredgt(refined_pred,pred,gt,sparse)            
            
            loss = depth_criterion(pred, gt)
            epoch_loss += loss.item()
            #loss.backward()
            #optimizer.step()
            
            rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)

            #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
 
            #refined_pred = pred
            #refined_pred = refinement_model(pred,sparse)

            refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)

            refined_loss = depth_criterion(refined_pred,gt)

            
            refined_epoch_loss += refined_loss.item()
            print(f'refined_loss',refined_loss)
            refined_loss.backward()
            refinement_optimizer.step()

        #EDW EKSW
        average_loss = epoch_loss / len(train_dl.dataset)
        refined_average_loss = refined_epoch_loss / len(train_dl.dataset)
        
        print(f'Training Loss: {average_loss}. Refined Training Loss: {refined_average_loss}')    

        evaluation_block(epoch)
        
if converted_args_dict['mode'] == 'eval':
    #pass
    evaluation_block(epoch)
elif converted_args_dict['mode'] == 'train':
    evaluation_block(epoch)
    training_block(model)
    #evaluation_block()
    