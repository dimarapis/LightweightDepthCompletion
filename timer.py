from cgi import test
from configparser import Interpolation
from torch.utils.data import DataLoader
from features.decnet_dataloaders import DecnetDataloader
from features.decnet_args import decnet_args_parser
from features.decnet_sanity import inverse_depth_norm

import torch
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from models.sparse_guided_depth import RgbGuideDepth
from models.sparse_guided_depth import DepthRefinement 
from models.sparse_guided_depth import DepthRefinement_p1
from models.sparse_guided_depth import Scaler



""" Script for testing various networks at inference time. All timings are given in milliseconds """


def inference_test(model_name):

    #Loading datasets
    print("\nSTEP 2. Loading datasets...")
    eval_dl = DataLoader(DecnetDataloader(decnet_args,decnet_args.val_datalist),batch_size=1)

    print(f'Loaded {len(eval_dl.dataset)} val files')

    # Load CUDA
    device = torch.device("cuda")

    # Load model
    if model_name == 'basemodel':
        model = RgbGuideDepth(True)
        model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location=device))
        model.to(device)
        model.eval()
    elif model_name == 'refinement_model':
        model = RgbGuideDepth(True)
        model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location=device))
        model.to(device)
        model.eval()
        refinement_model = DepthRefinement()
        #refinement_model.load_state_dict(torch.load('./weights/nn_final_ref.pth', map_location=device))
        refinement_model.to(device)
        refinement_model.eval()
    else:
        raise ValueError   


    # Create iteratable object
    test_dataset = iter(eval_dl)

    # Define start event, end event and number of repetitions
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50
    timings = np.zeros((repetitions, 1))

    # GPU warm-up
    for _ in range(20):
        data = test_dataset.next()
        image_filename = data['file'][0].split('/')[-1].rstrip('.png')
        image, gt, sparse = data['rgb'], data['gt'], data['d']
        
        rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
        if model_name == 'refinement_model':
            
            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            
            #gt_and_pred_info('basemodel', 'pred', pred)
            #visualize_results('basemodel',image,pred,sparse)

            refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)
            
        #_ = refi

    # Measure performance 
    with torch.no_grad():
        for rep in range(repetitions):
            data = test_dataset.next()
            image_filename = data['file'][0].split('/')[-1].rstrip('.png')
            image, gt, sparse = data['rgb'], data['gt'], data['d']
        
        

            starter.record()
            rgb_half, y_half, sparse_half, y, inv_pred = model(image,sparse)
            pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
            
            if model_name == 'refinement_model':
            
                pred = inverse_depth_norm(decnet_args.max_depth_eval,inv_pred)
                
                #gt_and_pred_info('basemodel', 'pred', pred)
                #visualize_results('basemodel',image,pred,sparse)

                refined_pred = refinement_model(rgb_half, image, y_half, y, sparse_half, sparse, pred)
            
            ender.record()

            # Wait for GPU to sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    # Calculate mean and std
    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)

    return mean_time, std_time

    
if __name__ == "__main__":
    print("\nSTEP 1. Loading arguments and parameters...")
    decnet_args = decnet_args_parser()

    #Print arguments and model options
    converted_args_dict = vars(decnet_args)
    print('\nParameters list: (Some may be redundant depending on the task, dataset and model chosen)')

    timings = {}
    
    models = ["basemodel", "refinement_model"]
    for model in models:
        model_mean, model_std = inference_test(model)
        timings[model] = [model_mean, model_std]

    
    print("Timings for all models given in milliseconds")
    print(timings)

    