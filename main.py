from sympy import Gt
import torch
import wandb
import random
import metrics
import warnings
import numpy as np
import torch.optim as optim

import features.CoordConv as CoordConv
import visualizers.visualizer as visualizer
import features.deprecated_metrics as custom_metrics
import features.custom_transforms as custom_transforms
import features.kitti_loader as guided_depth_kitti_loader

from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from thop import profile,clever_format
from torch.utils.data import DataLoader

from models.enet_pro import ENet
from models.guide_depth import GuideDepth
from features.decnet_sanity import np_min_max, torch_min_max
from features.decnet_args import decnet_args_parser
from features.decnet_sanity import inverse_depth_norm
from features.decnet_losscriteria import MaskedMSELoss
from features.decnet_dataloaders import DecnetDataloader
from models.sparse_guided_depth import SparseGuidedDepth
from models.sparse_guided_depth import SparseAndRGBGuidedDepth




epoch = 0
#Remove warning for visualization purposes (mostly due to behaviour oif upsample block)
#warnings.filterwarnings("ignore")

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

test_files = DecnetDataloader(decnet_args,decnet_args.val_datalist)
test_samples_no = len(test_files)
test_dl = DataLoader(test_files,batch_size=1)

train_files = DecnetDataloader(decnet_args,decnet_args.train_datalist)
train_samples_no = len(train_files)
train_dl = DataLoader(train_files,batch_size=8)



#print(test_dl)
print(f'Loaded {train_samples_no} training files')
print(f'Loaded {test_samples_no} val files')

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
#model = SparseGuidedDepth(False)#
#model = SparseAndRGBGuidedDepth(False)
#model = torch.nn.Sequential(
#          torch.nn.Conv2d(1,20,5),
#          torch.nn.ReLU(),
#          torch.nn.Conv2d(20,64,5),
#          torch.nn.ReLU()
#        )
#state_dict = torch.load('./weights/guide.pth', map_location='cpu')
#model.load_state_dict(state_dict, strict=False)
model.to(device)

rgb_shape = torch.randn(1, 3, decnet_args.train_height, decnet_args.train_width).to(device)
d_shape = torch.randn(1, 1, decnet_args.train_height, decnet_args.train_width).to(device)

#macs, params = profile(model, inputs=(rgb_shape, d_shape))
macs, params = profile(model, inputs=(rgb_shape, ))

macs, params = clever_format([macs, params], "%.3f")
print(f'model macs: {macs} and params: {params}')


wandb.config.update({"macs": macs, "params": params})



if decnet_args.torch_mode == 'tensorrt':
    from torch2trt import torch2trt

    #import tensorrt
    #import torch_tensorrt
    model.eval()
    x = torch.ones((1, 3, 384, 1280)).cuda()
    model_trt = torch2trt(model, [x])
    model = model_trt
    #trt_module = torch_tensorrt.compile(model,
    ##inputs = [torch_tensorrt.Input((1, 1, 384, 1280))], # input shape   
    #enabled_precisions = {torch_tensorrt.dtype.half} # Run with FP16
    #)# save the TensorRT embedded Torchscript
    #torch.jit.save(trt_module, "trt_torchscript_module.ts")
    #model = trt_module.to(device)
    

optimizer = optim.Adam(model.parameters(), lr=decnet_args.learning_rate) 
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#print(optimizer)

depth_criterion = MaskedMSELoss()
print(f"Loaded model {converted_args_dict['network_model']} for {converted_args_dict['task']}")

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

def evaluation_block(epoch):
    print("\nSTEP. Testing block...")
    print(optimizer)
    
 
    #with torch.no_grad():
    model.eval()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for i, data in enumerate(tqdm(test_dl)):
        #print(data.keys())
        
        #print(f'Currently on image {i} out of {samples_no}')

        
        #ENET_MODEL
        #_, _, pred =  model(data)
        
        #GUIDEDEPTH_MODEL
        #image, gt = data
        #packed_data = {'image': image[0], 'depth':gt[0]}
        
        
        #MY DATA
        image_filename = data['file']
        
        #image, gt = data['rgb'].permute(0,2,3,1), data['gt'][0] 
        image, gt, sparse = data['rgb'].permute(0,2,3,1), data['gt'][0], data['d'][0] 
        #packed_data = {'image': image[0].to('cpu'), 'depth': gt[0].to('cpu')}
        packed_data = {'image': image[0].to('cpu'), 'gt': gt[0].to('cpu'), 'sparse' : sparse[0].to('cpu')}
        
        #image,gt = data['rgb'], data['gt']

        
        data = to_tensor_test(packed_data)

        
        image, gt, sparse  = unpack_and_move(data)
        #mage = data['image'].to(device)#, non_blocking=True)
        #gt = data['depth'].to(device)#, non_blocking=True)
    
        image = image.unsqueeze(0)
        #print(f'imageshape {image.shape}')
        #continue
        gt = gt.unsqueeze(0)
        #print(f'sparse_shape_before {sparse.shape}')
        sparse = sparse.unsqueeze(0)
        
        #print(f'torch_minmax sparse {torch_min_max(sparse)}')
        #print(f'sparse_shape_afta {torch_min_max(gt)}')
        #image = downscale_image(image)#.permute(0,2,3,1))
        #rint(f'rgbshape {image.shape}')
        
        
        #GUIDEDEPTH
        #inv_pred =  model(image)#image.permute(0,2,3,1))
        #MYMODEL
        #print(image.shape)
        #inv_pred = model(image,sparse)
        
        inv_pred = model(image)
        

        

        
        #ALSO NEED TO BUILD EVALUATION ON FLIPPED IMAGE (LIKE  GUIDENDEPTH)

        pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
        print_torch_min_max_rgbpredgt(image,pred,gt)
    
        
        upscale_depth = transforms.Resize(gt.shape[-2:]) #To GT res
        #print('notupscaled', pred.shape)
        #print(prediction.shape)
        prediction = upscale_depth(pred)

        #print('upscaled', prediction.shape)
        #print(f'torchminmax pred {torch_min_max(pred)}')
        #print(f'torchminmax gt {torch_min_max(gt)}')
        #gt_height, gt_width = gt.shape[-2:] 

        #crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
        #                        0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
        #gt = gt[:,:, crop[0]:crop[1], crop[2]:crop[3]]
        #prediction = prediction[:,:, crop[0]:crop[1], crop[2]:crop[3]]
        #print(f'torchminmax pred cropped{torch_min_max(prediction)}')
        #print(f'torchminmax gt cropped{torch_min_max(gt)}')
        
        #print(f'image {torch_min_max(image)}')
        #print(f'predictiopn {torch_min_max(prediction)}')
        #print(f'gt {torch_min_max(gt)}')
        
        #prediction_flip = upscale_depth(prediction_flip)

        #print(torch_min_max(pred))
        #print(torch_min_max(gt))
        #loss = depth_criterion(pred, gt)
        #print(loss)
        '''
        #print(type(data['rgb']))
        #print(torch_min_max(data['rgb']))
        #print(torch_min_max(rgb_input))
        #print(rgb_input.shape)
        #transformed_input = transform_to_tensor(rgb_input)
        #print(transformed_input)
        '''
        
        #pred =  model(rgb_input)
        #print(prediction.shape, gt.shape)
        #print(f' torchminmax prediction {torch_min_max(prediction)}')
        #print(f' torchminmax gt {torch_min_max(gt)}')
        
        #depth_loss = depth_criterion(prediction, gt)
        #print(depth_loss)
        
        pred_d, depth_gt, = pred.squeeze(), gt.squeeze()#, data['d'].squeeze()# / 1000.0
        
        #print(pred_d.shape, depth_gt.shape)
        
        pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)
        #print(pred_crop.shape, gt_crop.shape)
        #print(torch_min_max(pred_crop), torch_min_max(gt_crop))
        
        
        computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)
        #print(computer_result)
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
        #print(res)
        #Saving depth prediciton data along with original image
        #visualizer.save_depth_prediction(prediction,data['rgb']*255)

        
        #Showing plots, results original image, etc
        #visualizer.plotter(pred_d,sparse_depth,depth_gt,pred,data['rgb'])
    

    
    #calculating total metrics by averaging  
    for metric in metric_name:
        result_metrics[metric] = result_metrics[metric] / float((i+1))
    print(float(i+1))
    # printing result 
    print("Results:")
    for key in result_metrics:
        print(key, ' = ', result_metrics[key])
    
    print(epoch)
    wandb.log(result_metrics, step = epoch)
    
    #Wandb save sample image
    wandb_image, wandb_depth_colorized = visualizer.wandb_image_prep(image,pred) 
    #print(np.min(wandb_image))
    #print(np.max(wandb_image))
    wandb.log({"Samples": [wandb.Image(wandb_image,caption="RGB sample"), wandb.Image(wandb_depth_colorized, caption="Colorized depth prediction")]},step = epoch)


    #images = wandb.Image(wandb_image, caption="Image_sample")

    #images = wandb.Image(wandb_image.squeeze().to)
    
    #print(pred_crop.shape)
    #print(image.shape)
    #print(sparse.shape)
    #print(gt.shape)
    #print('\n\n\n')



def training_block():
    print("\nSTEP. Training block...")
    for epoch in range(1,decnet_args.epochs):
        #model.train()
        
        for i, data in enumerate(tqdm(train_dl)):
            #print(data.keys())
            #image, gt = data['rgb'].permute(to(dtype=torch.float32), data['gt'].to(dtype=torch.float32)
            #packed_data = {'image': image[0].to('cpu'), 'depth': gt[0].to('cpu')}        
            #data_step = to_tensor(packed_data)
            #image, gt = unpack_and_move(data_step)
            #image = image.unsqueeze(0)
            #gt = gt.unsqueeze(0)
            #image = downscale_image(image)
            
    
            #packed_data = {'image': image[0].to('cpu'), 'depth': gt[0].to('cpu')}        
            #data['K'] = new_K
            #data['position'] = position
            
            image, gt, sparse = data['rgb'], data['gt'], data['d']#.permute(0,2,3,1), data['gt'], data['d']
            
            #image = np.array(image).astype(np.float32) / 255.0
            #gt = np.array(gt).astype(np.float32) #/ self.maxDepth #Why / maxDepth?
            #sparse = np.array(sparse).astype(np.float32)# / decnet_args.max_depth_eval #Why / maxDepth?
            #print(image.shape)     

            #image, gt, sparse = transform_to_tensor(image), transform_to_tensor(gt), transform_to_tensor(sparse)
            #image.permute(0,3,1,2)
            

            #print("???")
            #print(torch_min_max(image))  
            #print(image.shape)     
            
            #image = image / 255.
            
            #print(f'torchminmax image {torch_min_max(image)}')
            #print(f'torchminmax gt {torch_min_max(gt)}')
            #print(f'torchminmax sparse {torch_min_max(sparse)}')
            
            #print(torch_min_max(image))
            #packed_data = {'image': image.to('cpu'), 'depth': gt.to('cpu')}    

            #data_step = to_tensor(packed_data)
            #image, gt = unpack_and_move(data_step)
            
            #print(torch_min_max(image))
            #image = image.permute(0,3,1,2)     
            #inv_pred =  model(image)
            
            #print(f'sparse_shape_before {sparse.shape}')
            #print(f'rgbshape {image.shape}')
            #sparse = sparse.unsqueeze(0)
            #print(f'sparse_shape_afta {sparse.shape}')
            inv_pred = model(image)
            #inv_pred = model(image,sparse)
            
            #print(f'inv_pred_shape {inv_pred.shape}')
            #print(f'gt_shape {gt.shape}')
            
            #print(gt.shape)
            #print(f'torchminmax predbef {torch_min_max(inv_pred)}')
            #print(f'torchminmax gtbef {torch_min_max(gt)}')
            
            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            print_torch_min_max_rgbpredgt(image,pred,gt)
            
            #print(f'torchminmax pred {torch_min_max(pred)}')
            #print(f'torchminmax gt {torch_min_max(gt)}')
            
            #upscale_depth = transforms.Resize(gt.shape[-2:]) #To GT res
            #prediction = upscale_depth(pred)
            
            #gt_height, gt_width = gt.shape[-2:] 

            #crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
            #               0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
            
            #gt = gt[:,:, crop[0]:crop[1], crop[2]:crop[3]]
            #prediction = prediction[:,:, crop[0]:crop[1], crop[2]:crop[3]]
            #print(torch_min_max(pred))
            #print(torch_min_max(gt))
            loss = depth_criterion(pred, gt)
            #print(loss)
            #print(loss)
            #a = list(model.parameters())
            #print(a)
            
            #loss.backward()
            #self.optimizer.step()
            
            
            a = list(model.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            
            #print(torch.equal(a.data, b.data))
          
            #loss.backward()
            
            #optimizer.step()
            
            
            #print(loss)

            pred_d, depth_gt, sparse_depth = pred.squeeze(),gt.squeeze(), sparse.squeeze()

            
            #pred_crop, gt_crop = custom_metrics.cropping_img(decnet_args, pred_d, depth_gt)
            #computed_result = custom_metrics.eval_depth(pred_crop, gt_crop)
            #for metric in metric_name:
            #    result_metrics[metric] += computed_result[metric]
        
            # Using dictionary comprehension + keys()
            # Dictionary Values Division
            #res = {metric: result_metrics[metric] / float(i+1)
            #                        for metric in result_metrics.keys()}
            
            #VISUALIZE BLOCK

            #Saving depth prediciton data along with original image
            #visualizer.save_depth_prediction(prediction,data['rgb']*255)

            
            #Showing plots, results original image, etc
            #visualizer.plotter(pred_d,sparse_depth,depth_gt,pred,data['rgb'])
        
        lr_scheduler.step()
        evaluation_block(epoch)
        #asdd


    
if converted_args_dict['mode'] == 'eval':
    #pass
    evaluation_block(epoch)
elif converted_args_dict['mode'] == 'train':
    evaluation_block(epoch)
    training_block()
    #evaluation_block()
    