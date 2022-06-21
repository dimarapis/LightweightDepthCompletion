import torch
import wandb
import random
import metrics
import warnings
import numpy as np

import features.CoordConv as CoordConv
import visualizers.visualizer as visualizer
import features.deprecated_metrics as custom_metrics

import features.kitti_loader as guided_depth_kitti_loader
import features.custom_transforms as custom_transforms

from models.enet_pro import ENet
from models.guide_depth import GuideDepth

from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from features.decnet_sanity import torch_min_max
from features.decnet_sanity import inverse_depth_norm
from features.decnet_args import decnet_args_parser
from features.decnet_losscriteria import MaskedMSELoss
from features.decnet_dataloaders import DecnetDataloader



#Remove warning for visualization purposes (mostly due to behaviour oif upsample block)
warnings.filterwarnings("ignore")

#Loading arguments and model options
print("\nSTEP 1. Loading arguments and parameters...")
decnet_args = decnet_args_parser()


#Print arguments and model options
converted_args_dict = vars(decnet_args)
print('\nParameters list: (Some may be redundant depending on the task, dataset and model chosen)')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Defining metrics and loggers
metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']

#Initialize weights and biases logger
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
files = DecnetDataloader(decnet_args,decnet_args.val_datalist)
samples_no = len(files)
test_dl = DataLoader(files,batch_size=1)
print(test_dl)
print(f'Loaded {samples_no} test files')

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


#Loading model
print("\nSTEP 3. Loading model and metrics...")

#ENET_MODEL
'''
model = ENet(decnet_args).to(device)
checkpoint = torch.load('weights/e.pth.tar', map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
'''
#GUIDEDEPTH_MODEL
model = GuideDepth(True)
state_dict = torch.load('./weights/guide.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(device)

model.eval()
depth_criterion = MaskedMSELoss()
print(f"Loaded model {converted_args_dict['network_model']} for {converted_args_dict['task']}")

#SKATA

test_loader = guided_depth_kitti_loader.get_dataloader('kitti',
                                            path='data_guidedepth',
                                            split='test',
                                            batch_size=1,
                                            augmentation='alhashim',
                                            resolution='full',
                                            workers=4)

to_tensor = custom_transforms.ToTensor(test=True, maxDepth=80.0)

downscale_image = transforms.Resize((384,1280)) #To Model resolution


def unpack_and_move(data):
    if isinstance(data, (tuple, list)):
        print('here1')
        image = data[0].to(device, non_blocking=True)
        gt = data[1].to(device, non_blocking=True)
        return image, gt
    if isinstance(data, dict):
        print('here2')
        keys = data.keys()
        image = data['image'].to(device, non_blocking=True)
        gt = data['depth'].to(device, non_blocking=True)
        return image, gt
    print('Type not supported')

#Iterate images  
print("\nSTEP 4. Training or eval stage...")
if converted_args_dict['mode'] == 'eval':
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for i, data in enumerate(tqdm(test_dl)):
        #print(f'Currently on image {i} out of {samples_no}')
        
        
        
        #data['K'] = new_K
        #data['position'] = position

        
        #ENET_MODEL
        #_, _, pred =  model(data)
        
        #GUIDEDEPTH_MODEL
        #image, gt = data
        #packed_data = {'image': image[0], 'depth':gt[0]}
        
        
        #MY DATA
        #print(data.keys())
        image_filename = data['file']
        image, gt = data['rgb'].permute(0,2,3,1), data['gt'][0] 
        packed_data = {'image': image[0].to('cpu'), 'depth': gt[0].to('cpu')}
        #image,gt = data['rgb'], data['gt']


        
        data = to_tensor(packed_data)
        print(data)
        
        #image, gt = unpack_and_move(data)
        print(image_filename)
        print(torch_min_max(gt))
        
        #print(torch_min_max(inv_pred))
        print(gt.shape)
        print('\n')
        break
        image = image.unsqueeze(0)
        #print(f'imageshape {image.shape}')
        #continue
        gt = gt.unsqueeze(0)
        image = downscale_image(image)#.permute(0,2,3,1))
        
        #GUIDEDEPTH
        #inv_pred =  model(image)#image.permute(0,2,3,1))
        #MYMODEL
        #print(image.shape)
        inv_pred =  model(image)

        #gt = downscale_image(gt)
        
        
        #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)
        
            
        #packed_data = {'image': data['rgb'].squeeze().to('cpu'), 'depth': data['gt'].squeeze().to('cpu')}
        #packed_cpu = packed_data.to('cpu')
        #data = to_tensor(packed_data)
        #image, gt = unpack_and_move(data)
        #image = image.unsqueeze(0)
        
        #gt = gt.unsqueeze(0)

        #image_flip = torch.flip(image, [3])
        #gt_flip = torch.flip(gt, [3])
        
        #rgb_input = data['rgb'].type(torch.cuda.FloatTensor) / 255.0
        #print(rgb_input.shape)
        #print(f'\n\ndtype of rgb_input: {rgb_input.dtype}')

        #print(f'image_torch_min_max {torch_min_max(rgb_input)}')
        #transformed_input = transform_to_tensor(rgb_input)
        #print(image.shape)
        
        #print(f'\n\nINVERSE_pred_torch_min_max {torch_min_max(inv_pred)}')
        pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
        #print(f'PRED_torch_min_max {torch_min_max(pred)}')
        #print(f"DATA_GT {torch_min_max(gt)}")
        
    
        
        upscale_depth = transforms.Resize(gt.shape[-2:]) #To GT res

        prediction = upscale_depth(pred)
        
        gt_height, gt_width = gt.shape[-2:] 

        crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
                                0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
        gt = gt[:,:, crop[0]:crop[1], crop[2]:crop[3]]
        prediction = prediction[:,:, crop[0]:crop[1], crop[2]:crop[3]]
        
        
        #print(f'image {torch_min_max(image)}')
        #print(f'predictiopn {torch_min_max(prediction)}')
        #print(f'gt {torch_min_max(gt)}')
        
        #prediction_flip = upscale_depth(prediction_flip)

        
        '''
        #print(type(data['rgb']))
        #print(torch_min_max(data['rgb']))
        #print(torch_min_max(rgb_input))
        #print(rgb_input.shape)
        #transformed_input = transform_to_tensor(rgb_input)
        #print(transformed_input)
        '''
        
        #pred =  model(rgb_input)
        depth_loss = depth_criterion(prediction, gt)
            
        
        pred_d, depth_gt, = prediction.squeeze(),gt.squeeze()#, data['d'].squeeze()# / 1000.0

        #print(pred_d.shape, depth_gt.shape)
        
        pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)
        computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)
        #print(f'predctop and gtcrop shapes {pred_crop.shape} asd {gt_crop.shape}')
        for metric in metric_name:
            result_metrics[metric] += computed_result[metric]
        
        #if i == 10: 
        #    break
        
        # print(result_metrics)
    
        # Using dictionary comprehension + keys()
        # Dictionary Values Division
        res = {metric: result_metrics[metric] / float(i+1)
                                for metric in result_metrics.keys()}
        

        
        #VISUALIZE BLOCK

        #Saving depth prediciton data along with original image
        #visualizer.save_depth_prediction(pred,data['rgb'])

        
        #Showing plots, results original image, etc
        #visualizer.plotter(pred_d,sparse_depth,depth_gt,pred,data['rgb'])

    #Print    
    for metric in metric_name:
        result_metrics[metric] = result_metrics[metric] / float((i+1))
        
    # printing result 
    print("Results:")
    for key in result_metrics:
        print(key, ' = ', result_metrics[key])
