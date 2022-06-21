import torch
import torch.nn as nn
import torch.nn.functional as F

from models.guide_ddrnet import DualResNet_Backbone
from models.guide_modules import Guided_Upsampling_Block


class SparseGuidedDepth(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(SparseGuidedDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")


    def forward(self, rgb, depth):
        y = self.feature_extractor(rgb)

        x_half = F.interpolate(depth, scale_factor=.5)
        x_quarter = F.interpolate(depth, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_3(rgb, y)
        return y
    
    
class SparseAndRGBGuidedDepth(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(SparseAndRGBGuidedDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=4,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=4,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")


    def forward(self, rgb, depth):
        y = self.feature_extractor(rgb)
        print('tara')

        x_half = F.interpolate(depth, scale_factor=.5)
        x_quarter = F.interpolate(depth, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_3(rgb, y)
        return y