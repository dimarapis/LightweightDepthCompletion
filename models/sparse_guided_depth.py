from soupsieve import select
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TensorFlowBenchmarkArguments

from models.guide_ddrnet import DualResNet_Backbone
from models.guide_modules import Guided_Upsampling_Block, AuxUpsamplingBlock, AuxSparseUpsamplingBlock, DepthCorrector
from models.guide_modules import MinkoEncoder

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
                                   guide_features=4,
                                   guidance_type="full")


    def forward(self, rgb, depth):
        y = self.feature_extractor(rgb)
        #print('tara')
        rgbd = torch.cat((rgb,depth),1)

        x_half = F.interpolate(rgbd, scale_factor=.5)
        x_quarter = F.interpolate(rgbd, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_3(rgbd, y)
        return y

class AuxSparseGuidedDepth(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(AuxSparseGuidedDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = AuxUpsamplingBlock(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = AuxUpsamplingBlock(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = AuxUpsamplingBlock(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        self.ref_down_1 = DepthCorrector(in_features = 1,
                                        expand_features=inner_features[2],
                                        out_features= up_features[1],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )

        self.ref_down_2 = DepthCorrector(in_features = up_features[1],
                                        expand_features=inner_features[1],
                                        out_features= up_features[0],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )

        self.ref_up_2 = DepthCorrector(in_features = up_features[0],
                                        expand_features=inner_features[0],
                                        out_features= up_features[1],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
        self.ref_up_1 = DepthCorrector(in_features = up_features[1],
                                        expand_features=inner_features[1],
                                        out_features= up_features[2],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
        self.final = DepthCorrector(in_features = up_features[2],
                                        expand_features=inner_features[2],
                                        out_features= 1,
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
                                        


    def forward(self, rgb, sparse):
        y = self.feature_extractor(rgb)
        #print('tara')
        #rgbd = torch.cat((rgb,depth),1)

        x_half = F.interpolate(rgb, scale_factor=.5)
        x_quarter = F.interpolate(rgb, scale_factor=.25)

        

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_3(rgb, y)
        #print(y.shape)

        sparse_half = F.interpolate(sparse, scale_factor=.5)
        sparse_quarter =  F.interpolate(sparse, scale_factor=.25)
        
        y = self.ref_down_1(sparse,y)

        y = F.interpolate(y, scale_factor=.5, mode='bilinear')
        y = self.ref_down_2(sparse_half,y)

        y = F.interpolate(y, scale_factor=.5, mode='bilinear')        
        #y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.ref_up_2(sparse_quarter,y)
        
        y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.ref_up_1(sparse_half,y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.final(sparse,y)
        #print("perase apo dw, apisteuto")


        return y


class AuxGuidedDepth(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(AuxGuidedDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = AuxUpsamplingBlock(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = AuxUpsamplingBlock(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = AuxUpsamplingBlock(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")


    def forward(self, rgb):#, sparse):
        y = self.feature_extractor(rgb)
        print(y.shape)
        #print('tara')
        #rgbd = torch.cat((rgb,depth),1)

        x_half = F.interpolate(rgb, scale_factor=.5)
        x_quarter = F.interpolate(rgb, scale_factor=.25)

        #sparse_half = F.interpolate(sparse, scale_factor=.5)
        #sparse_quarter =  F.interpolate(sparse, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_3(rgb, y)
        return y


class RefinementModule(nn.Module):
    def __init__(self, 
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(RefinementModule, self).__init__()

        self.ref_down_1 = DepthCorrector(in_features = 1,
                                        expand_features=inner_features[2],
                                        out_features= up_features[1],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )

        self.ref_down_2 = DepthCorrector(in_features = up_features[1],
                                        expand_features=inner_features[1],
                                        out_features= up_features[0],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )

        self.ref_up_2 = DepthCorrector(in_features = up_features[0],
                                        expand_features=inner_features[0],
                                        out_features= up_features[1],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
        self.ref_up_1 = DepthCorrector(in_features = up_features[1],
                                        expand_features=inner_features[1],
                                        out_features= up_features[2],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
        self.final = DepthCorrector(in_features = up_features[2],
                                        expand_features=inner_features[2],
                                        out_features= 1,
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
                                        


    def forward(self, pred, sparse):

        sparse_half = F.interpolate(sparse, scale_factor=.5)
        sparse_quarter =  F.interpolate(sparse, scale_factor=.25)
        
        y = self.ref_down_1(sparse,pred)

        y = F.interpolate(y, scale_factor=.5, mode='bilinear')
        y = self.ref_down_2(sparse_half,y)

        y = F.interpolate(y, scale_factor=.5, mode='bilinear')        
        #y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.ref_up_2(sparse_quarter,y)
        
        y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.ref_up_1(sparse_half,y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.final(sparse,y)
        #print("perase apo dw, apisteuto")


        return y


class MinkoRefinement(nn.Module):
    def __init__(self, 
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(MinkoRefinement, self).__init__()

        self.ref_down_1 = MinkoEncoder(in_features = 1,
                                        expand_features=inner_features[2],
                                        out_features= up_features[1],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )

        self.ref_down_2 = DepthCorrector(in_features = up_features[1],
                                        expand_features=inner_features[1],
                                        out_features= up_features[0],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )

        self.ref_up_2 = DepthCorrector(in_features = up_features[0],
                                        expand_features=inner_features[0],
                                        out_features= up_features[1],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
        self.ref_up_1 = DepthCorrector(in_features = up_features[1],
                                        expand_features=inner_features[1],
                                        out_features= up_features[2],
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
        self.final = DepthCorrector(in_features = up_features[2],
                                        expand_features=inner_features[2],
                                        out_features= 1,
                                        kernel_size=3,
                                        channel_attention=True,
                                        guide_features=1
                                        )
                                        


    def forward(self, pred, sparse):

        sparse_half = F.interpolate(sparse, scale_factor=.5)
        sparse_quarter =  F.interpolate(sparse, scale_factor=.25)
        
        y = self.ref_down_1(sparse,pred)

        y = F.interpolate(y, scale_factor=.5, mode='bilinear')
        y = self.ref_down_2(sparse_half,y)

        y = F.interpolate(y, scale_factor=.5, mode='bilinear')        
        #y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.ref_up_2(sparse_quarter,y)
        
        y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.ref_up_1(sparse_half,y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')
        y = self.final(sparse,y)
        #print("perase apo dw, apisteuto")


        return y
