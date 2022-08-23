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
from features.decnet_losscriteria import MaskedMSELoss, SiLogLoss
from features.decnet_dataloaders import DecnetDataloader
from models.sparse_guided_depth import AuxSparseGuidedDepth, SparseGuidedDepth, DecnetModule
from models.sparse_guided_depth import SparseAndRGBGuidedDepth, RefinementModule, DecnetSparseIncorporated

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

if decnet_args.network_model == "GuideDepth":
    #print(decnet_args.pretrained)
    model = GuideDepth(True)

    if decnet_args.pretrained == True:
        model.load_state_dict(torch.load('./weights/NYU_Full_GuideDepthOriginal.pth', map_location=device))  
        #model.load_state_dict(torch.load('./weights/2022_08_21-10_23_53_PM/GuideDepth_99.pth', map_location='cpu'))  
        #2022_08_21-10_23_53_PM
      
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

elif decnet_args.network_model == "DecnetModule":
    model = DecnetSparseIncorporated()
        
    if decnet_args.pretrained == True:
        #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location='cpu'), strict=False)
        model.load_state_dict(torch.load('./weights/NYU_Full_GuideDepthOriginal.pth', map_location=device), strict=False)

        #model.load_state_dict(torch.load('./weights/2022_08_21-10_10_22_PM/DecnetModule_99.pth', map_location='cpu'))#, strict=False)

elif decnet_args.network_model == "AuxSparseGuidedDepth":
    model = GuideDepth(True)
    #0209refinement_model = RefinementModule()
    if decnet_args.pretrained == True:
        model.load_state_dict(torch.load('./weights/KITTI_Full_GuideDepth.pth', map_location='cpu'))  

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

depth_criterion = MaskedMSELoss()
#depth_criterion = SiLogLoss()

#defining index (epoch)
epoch = 0
prev_loss = 0.0
#print(f"Loaded model {converted_args_dict['network_model']}'# for {converted_args_dict['task']}")



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

def evaluation_block(epoch):
    print(f"\nSTEP. Testing block... Epoch no: {epoch}")
    torch.cuda.empty_cache()
    model.eval()
    #0209refinement_model.eval()
    global best_rmse

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

            if decnet_args.network_model == 'GuideDepth':
                inv_pred = model(image)
            else:    
            #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
                inv_pred = model(image, sparse)
            #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
            #inv_pred = model(image)
            
            #print(inv_pred)
            
            pred = inverse_depth_norm(max_depth,inv_pred)
            #print(f'pred {torch_min_max(pred)}')
            print_torch_min_max_rgbsparsepredgt(image, sparse, pred, gt)   
            #print(image_filename)         
            #ipnut = input()

            #print_torch_min_max_rgbpredgt(image,pred,gt)            
            
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

            pred_d, depth_gt, = pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)    
            computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)

            #0209refined_pred_d, refined_depth_gt, = refined_pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
            #0209refined_pred_crop, refined_gt_crop = custom_metrics.cropping_img(decnet_args, refined_pred_d, refined_depth_gt)    
            #0209refined_computed_result = custom_metrics.eval_depth(refined_pred_crop, refined_gt_crop)


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
        #VISUALIZE BLOCK
        #Saving depth prediciton data along with original image
        #visualizer.save_depth_prediction(prediction,data['rgb']*255)

        #Showing plots, results original image, etc
        #visualizer.plotter(pred_d,sparse_depth,depth_gt,pred,data['rgb'])
            
        #calculating total metrics by averaging  
        for metric in metric_name:
            result_metrics[metric] = result_metrics[metric] / float((i+1))
            #0209refined_result_metrics[metric] = refined_computed_result[metric] / float((i+1))

        tabulator, refined_tabulator = [],[]
        for key in result_metrics:
            tabulator.append([key,result_metrics[key]]) 
            #0209refined_tabulator.append([key, refined_result_metrics[key]])

        if epoch == decnet_args.epochs:
            print(f"Results on epoch: {epoch}")
            print("Base model results")
            print(tabulate(tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished evaluation block")
            #0209print("Refined model results")
            #0209print(tabulate(refined_tabulator, tablefmt='orgtbl'))
            #0209print(f"\n\nFinished training..")
            print(f"Average time for parsing images {time.time - t0}")


        else:
            print(f"Results on epoch: {epoch}")
            print("Base model results")
            print(tabulate(tabulator, tablefmt='orgtbl'))
            print(f"\n\nFinished evaluation block")
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
                

                path = f"weights/{grabtime}/{decnet_args.network_model}_{epoch}.pth"
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

            if decnet_args.network_model == 'GuideDepth':
                inv_pred = model(image)
            else:    
            #rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
                inv_pred = model(image, sparse)
            #inv_pred = model(image)
            #print(inv_pred)
            #print(pred.shape,image.shape)
            pred = inverse_depth_norm(max_depth,inv_pred)
            #print(pred.shape,image.shape)
            
            #inv_pred = model(image)        
            #0209refined_inv_pred = refinement_model(inv_pred,sparse)
            #0209refined_pred = inverse_depth_norm(decnet_args.max_depth_eval,refined_inv_pred)            
            #print(f'inv_pred {torch_min_max(inv_pred)}')

            #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
            #pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            #print(f'pred {torch_min_max(pred)}')
            #print_torch_min_max_rgbpredgt(image,  pred, gt)            
            print_torch_min_max_rgbsparsepredgt(image[0], sparse[0], pred[0], gt[0])            
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
            print(f'Iteration {iteration} out of {int(np.ceil(len(train_dl.dataset) / decnet_args.batch_size))}. Loss: {loss.item()}')
            

        average_loss = epoch_loss / (len(train_dl.dataset) / decnet_args.batch_size)
        print(f'Training Loss: {average_loss}. Epoch {epoch} of {decnet_args.epochs}')

        evaluation_block(epoch)
        
if converted_args_dict['mode'] == 'eval':
    #pass
    evaluation_block(epoch)
elif converted_args_dict['mode'] == 'train':
    epoch = 0
    #evaluation_block(epoch)
    training_block(model)
    #evaluation_block()
    