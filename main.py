from os import device_encoding
from sympy import Gt
import torch
import wandb
import random
import metrics
import warnings
import numpy as np
import torch.optim as optim

import features.CoordConv as CoordConv
from models.enet_basic import weights_init
import visualizers.visualizer as visualizer
import features.deprecated_metrics as custom_metrics
import features.custom_transforms as custom_transforms
import features.kitti_loader as guided_depth_kitti_loader

from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
from thop import profile,clever_format
from torch.utils.data import DataLoader

from models.enet_pro import ENet
from models.guide_depth import GuideDepth
from features.decnet_sanity import np_min_max, torch_min_max
from features.decnet_args import decnet_args_parser
from features.decnet_sanity import inverse_depth_norm
from features.decnet_losscriteria import MaskedMSELoss, SiLogLoss
from features.decnet_dataloaders import DecnetDataloader
from models.sparse_guided_depth import SparseGuidedDepth
from models.sparse_guided_depth import SparseAndRGBGuidedDepth





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
    wandb.init(project=str(decnet_args.project),entity=str(decnet_args.entity))#)="decnet-project", entity="wandbdimar")
    wandb.config.update(decnet_args)

#Printing args for checking
for key in converted_args_dict:
    print(key, ' : ', converted_args_dict[key])

#Ensuring reproducibility using seed
seed = 2910
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#Loading datasets
print("\nSTEP 2. Loading datasets...")
train_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.train_datalist),batch_size=decnet_args.batch_size)
test_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.val_datalist),batch_size=1)
print(f'Loaded {len(DecnetDataloader(decnet_args,decnet_args.train_datalist))} training files')
print(f'Loaded {len(DecnetDataloader(decnet_args,decnet_args.val_datalist))} val files')

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
    model = GuideDepth(False)
    #print(decnet_args.pretrained)
    if decnet_args.pretrained == True:
        model.load_state_dict(torch.load('./weights/GuideDepth.pth', map_location='cpu'))        
elif decnet_args.network_model == "SparseGuidedDepth":
    model = SparseGuidedDepth(False)
    #if decnet_args.pretrained
        #model.load_state_dict(torch.load('./weights/guide.pth', map_location='cpu'))     
elif decnet_args.network_model == "SparseAndRGBGuidedDepth":
    model = SparseAndRGBGuidedDepth(False)
    #if decnet_args.pretrained
        #model.load_state_dict(torch.load('./weights/guide.pth', map_location='cpu'))     
elif decnet_args.network_model == "ENET2021":
    model = ENet(decnet_args)
else:
    print("Can't seem to find the model configuration. Make sure you choose a model by --network-model argument. Integrated options are: [GuideDepth,SparseGuidedDepth,SparseAndRGBGuidedDepth,ENET2021]") 

model.to(device)

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
    model.eval()
    x = torch.ones((1, 3, 384, 1280)).cuda()
    model_trt = torch2trt(model, [x])
    model = model_trt
'''


optimizer = optim.Adam(model.parameters(), lr=decnet_args.learning_rate)#, momentum=0.9) 
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,50,75,90], gamma=0.1)
#lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

depth_criterion = MaskedMSELoss()
#depth_criterion = SiLogLoss()

#defining index (epoch)
epoch = 0
prev_loss = 0.0
#print(f"Loaded model {converted_args_dict['network_model']}'# for {converted_args_dict['task']}")

to_tensor_test = custom_transforms.ToTensor(test=True, maxDepth=80.0)
to_tensor = custom_transforms.ToTensor(test=False, maxDepth=80.0)

downscale_image = transforms.Resize((384,1280)) #To Model resolution


def print_torch_min_max_rgbpredgt(rgb,pred,gt):
    print('\n')
    print(f'torch_min_max rgb {torch_min_max(rgb)}')
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
    #print(optimizer)
    #with torch.no_grad():
    model.eval()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    #for i, data in enumerate(tqdm(test_dl)):
    #data = next(iter(test_dl))
    #i = 0
    for i, data in enumerate(tqdm(test_dl)):
        model.eval()
        image_filename = data['file']
        #print(image_filename)
        image, gt, sparse = data['rgb'], data['gt'], data['d']#.permute(0,2,3,1), data['gt'], data['d']

        inv_pred = model(image)
        
        #inv_pred = model(image,sparse)
        #inv_pred = model(image)
        
        #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
        pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
        #print_torch_min_max_rgbpredgt(image,pred,gt)            

        #upscaling depth to compare (if needed)
        #upscale_depth = transforms.Resize(gt.shape[-2:]) #To GT res
        #prediction = upscale_depth(pred)

        pred_d, depth_gt, = pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0

        pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)    
        computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)

        for metric in metric_name:
            result_metrics[metric] += computed_result[metric]
        
    #VISUALIZE BLOCK
    #Saving depth prediciton data along with original image
    #visualizer.save_depth_prediction(prediction,data['rgb']*255)

    #Showing plots, results original image, etc
    #visualizer.plotter(pred_d,sparse_depth,depth_gt,pred,data['rgb'])
        
    #calculating total metrics by averaging  
    for metric in metric_name:
        result_metrics[metric] = result_metrics[metric] / float((i+1))
    #print(float(i+1))
    
    # printing result 
    print("Results:")
    for key in result_metrics:
        print(key, ' = ', result_metrics[key])
    
    if decnet_args.wandblogger == True:
        if epoch != 0:
            epoch = epoch[1]
        wandb.log(result_metrics, step = epoch)
        #Wandb save sample image
        wandb_image, wandb_depth_colorized = visualizer.wandb_image_prep(image,pred) 
        wandb.log({"Samples": [wandb.Image(wandb_image,caption="RGB sample"), wandb.Image(wandb_depth_colorized, caption="Colorized depth prediction")]},step = epoch)
    model.train()




def training_block(model):
    
    print("\nSTEP. Training block...")
    data = next(iter(train_dl))
    for epoch in enumerate(tqdm(range(1,int(decnet_args.epochs)+1))):
        for i, data in enumerate(tqdm(train_dl)):
                
            image, gt, sparse = data['rgb'], data['gt'], data['d']#.permute(0,2,3,1), data['gt'], data['d']

            inv_pred = model(image)
            #inv_pred = model(image,sparse)
            
            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            #print_torch_min_max_rgbpredgt(image,pred,gt)            
            
            #pred = F.interpolate(pred,size=(352,608),mode='bilinear')

            loss = depth_criterion(pred, gt)
            a = list(model.parameters())[0].clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            
            lr_scheduler.step()
        
        
        
        if epoch[1] == 1:
            print(f'\nStarting loss {loss.item()}')
        elif epoch[1] == decnet_args.epochs:
            path = f"weights/{decnet_args.network_model}.pth"
            torch.save(model.state_dict(), path)
            print(f"\nSaved model in {path} with last loss {loss.item()}")
        else:
            print(f"Current loss: {loss.item()}")

        if np.isnan(loss.item()):
            print("ton ipiame")
            x = list(model.parameters())[0].clone()
            print(x)#print(model.params())
            
        evaluation_block(epoch)
        
if converted_args_dict['mode'] == 'eval':
    #pass
    evaluation_block(epoch)
elif converted_args_dict['mode'] == 'train':
    evaluation_block(epoch)
    training_block(model)
    #evaluation_block()
    