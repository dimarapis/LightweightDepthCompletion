import torch
import numpy as np
from torchvision import transforms

def torch_min_max(data):
    minmax = (torch.min(data.float()).item(),torch.mean(data.float()).item(),torch.median(data.float()).item(),torch.max(data.float()).item())
    #print(minmax)
    return minmax

def np_min_max(data):
    minmax = (np.min(data.float()),np.max(data.float()),np.mean(data.float()),np.median(data.float()))
    return minmax

def inverse_depth_norm(max_depth, depth):
    depth = max_depth / depth
    depth = torch.clamp(depth, max_depth / 100, max_depth)
    return depth

def transformToTensor():
    transforms.ToTensor()