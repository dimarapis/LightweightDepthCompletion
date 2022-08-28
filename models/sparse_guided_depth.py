from soupsieve import select
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TensorFlowBenchmarkArguments

from models.guide_ddrnet import DualResNet_Backbone
from models.guide_modules import Decnet_Guided_Upsampling_Block, Guided_Upsampling_Block,AuxUpsamplingBlock, AuxSparseUpsamplingBlock, DepthCorrector
from models.guide_modules import MinkoEncoder

from models.enet_basic import *
from .common import *


from features.decnet_sanity import torch_min_max


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


class RgbGuideDepth(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(RgbGuideDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        
    def forward(self, rgb, sparse):
        y_eighth = self.feature_extractor(rgb)
        #print('y.shape', y_eighth.shape) # [B, 64, 44, 76]
        #print(y_eighth,shape)
        rgb_half_for_cspn = F.interpolate(rgb, scale_factor=.5)
        rgb_quarter = F.interpolate(rgb, scale_factor=.25)

        sparse_half = F.interpolate(sparse, scale_factor=.5)
        #sparse_quarter = F.interpolate(sparse, scale_factor=.25)


        y_quarter = F.interpolate(y_eighth, scale_factor=2., mode='bilinear')#, align_corners=True)
        y_quarter = self.up_1(rgb_quarter, y_quarter)
        #print('y_quarter.shape', y_quarter.shape) # [B, 32, 88, 152]


        y_half_for_cspn = F.interpolate(y_quarter, scale_factor=2., mode='bilinear')#,align_corners=True)
        #print(f'y_half_for_cspn {y_half_for_cspn.shape}')

        
        y_half = self.up_2(rgb_half_for_cspn, y_half_for_cspn)
        #print('y_half.shape', y_half.shape) # [B, 16, 176, 304]


        y_for_cspn = F.interpolate(y_half, scale_factor=2., mode='bilinear')#, align_corners=True)
        #print(f'y_for_cspn {y_for_cspn.shape}')
        #print('y_for_cspn.shape', y_for_cspn.shape) # [B, 16, 352, 608]


        pred = self.up_3(rgb, y_for_cspn)
        #print('pred.shape', pred.shape) # [B, 1, 352, 608]
        

        return rgb_half_for_cspn, y_half_for_cspn, sparse_half, y_for_cspn, pred


#Mods
class DepthRefinement_p1(nn.Module):
    def __init__(self, 
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DepthRefinement_p1, self).__init__()

        self.rgb_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        self.rgb_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.sparse_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.mask_layer = convbn(64, 3)

        self.kernel_conf_layer = convbn(64, 3)
        self.iter_conf_layer = convbn(64, 12)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(64, 7)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(3)
        self.CSPN5 = CSPNAccelerate(5, padding=2)
        self.CSPN7 = CSPNAccelerate(7, padding=3)

        # CSPN new
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)

        weights_init(self)

    def forward(self, rgb_half, rgb, y_half, y, sparse_half, sparse, pred):#(rgb_half, image, y_half, y, sparse_half, sparse, pred)
        #rgb = input['rgb']
        rgb_cspn_half = self.rgb_cspn_input_half_reso(rgb_half, y_half)
        #print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        
        sparse_cspn_half = self.sparse_cspn_input_half_reso(sparse_half, y_half)
        #print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        
        rgb_cspn = self.rgb_cspn_input(rgb, y)
        #print(f'rgb_cspn {rgb_cspn.shape}')
        
        sparse_cspn = self.sparse_cspn_input(sparse, y)
        #print(f'sparse_cspn {sparse_cspn.shape}')
        
        
        '''
        print(f'pred {pred.shape}')
        print(f'rgb  {rgb.shape}')
        print(f'sparse {sparse.shape}')
        print(f'rgb_half {rgb_half.shape}')
        print(f'sparse_half {sparse_half.shape}')
        print(f'y {y.shape}')
        print(f'y_half {y_half.shape}')
        print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        print(f'rgb_cspn {rgb_cspn.shape}')
        print(f'sparse_cspn {sparse_cspn.shape}')
        '''


        d = sparse
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature, coarse_depth = torch.cat((rgb_cspn, sparse_cspn), 1),pred# torch.cat((rgb_cspn_half, sparse_cspn_half),1), pred
        #print(f'feature_s1 {feature_s1.shape}')
        #print(f'feature_s2 {feature_s2.shape}')
        
        depth = coarse_depth
        
        #d = input['d']
        #valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        #feature, coarse_depth= self.backbone(input)

        mask = self.mask_layer(feature)
        mask = torch.sigmoid(mask)

        mask = mask*valid_mask
        mask3 = mask[:, 0:1, :, :]
        mask5 = mask[:, 1:2, :, :]
        mask7 = mask[:, 2:3, :, :]

        kernel_conf = self.kernel_conf_layer(feature)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        conf = self.iter_conf_layer(feature)
        conf3 = conf[:, 0:4, :, :]
        conf5 = conf[:, 4:8, :, :]
        conf7 = conf[:, 8:12, :, :]
        conf3 = self.softmax(conf3)
        conf5 = self.softmax(conf5)
        conf7 = self.softmax(conf7)

        guide3 = self.iter_guide_layer3(feature)
        guide5 = self.iter_guide_layer5(feature)
        guide7 = self.iter_guide_layer7(feature)

        #init
        depth = coarse_depth
        depth3 = depth
        depth5 = depth
        depth7 = depth

        d3_list = [i for i in range(4)]
        d5_list = [i for i in range(4)]
        d7_list = [i for i in range(4)]

        #prop
        guide3 = kernel_trans(guide3, self.encoder3)
        guide5 = kernel_trans(guide5, self.encoder5)
        guide7 = kernel_trans(guide7, self.encoder7)

        for i in range(12):
            depth3 = self.CSPN3(guide3, depth3, depth)
            depth3 = mask3*d + (1-mask3)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask5*d + (1-mask5)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask7*d + (1-mask7)*depth7

            if(i==2):
                d3_list[0] = depth3
                d5_list[0] = depth5
                d7_list[0] = depth7

            if(i==5):
                d3_list[1] = depth3
                d5_list[1] = depth5
                d7_list[1] = depth7

            if(i==8):
                d3_list[2] = depth3
                d5_list[2] = depth5
                d7_list[2] = depth7

            if(i==11):
                d3_list[3] = depth3
                d5_list[3] = depth5
                d7_list[3] = depth7

        refined_depth = \
        d3_list[0] * (kernel_conf3 * conf3[:, 0:1, :, :]) + \
        d3_list[1] * (kernel_conf3 * conf3[:, 1:2, :, :]) + \
        d3_list[2] * (kernel_conf3 * conf3[:, 2:3, :, :]) + \
        d3_list[3] * (kernel_conf3 * conf3[:, 3:4, :, :]) + \
        d5_list[0] * (kernel_conf5 * conf5[:, 0:1, :, :]) + \
        d5_list[1] * (kernel_conf5 * conf5[:, 1:2, :, :]) + \
        d5_list[2] * (kernel_conf5 * conf5[:, 2:3, :, :]) + \
        d5_list[3] * (kernel_conf5 * conf5[:, 3:4, :, :]) + \
        d7_list[0] * (kernel_conf7 * conf7[:, 0:1, :, :]) + \
        d7_list[1] * (kernel_conf7 * conf7[:, 1:2, :, :]) + \
        d7_list[2] * (kernel_conf7 * conf7[:, 2:3, :, :]) + \
        d7_list[3] * (kernel_conf7 * conf7[:, 3:4, :, :])

        return refined_depth

class DepthRefinement(nn.Module):
    def __init__(self, 
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DepthRefinement, self).__init__()
        
        self.prints = False

        self.rgb_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        self.rgb_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.sparse_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.kernel_conf_layer = convbn(64, 3)
        self.mask_layer = convbn(64, 1)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(64, 7)

        self.kernel_conf_layer_s2 = convbn(128, 3)
        self.mask_layer_s2 = convbn(128, 1)
        self.iter_guide_layer3_s2 = CSPNGenerateAccelerate(128, 3)
        self.iter_guide_layer5_s2 = CSPNGenerateAccelerate(128, 5)
        self.iter_guide_layer7_s2 = CSPNGenerateAccelerate(128, 7)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.nnupsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)
        self.CSPN3_s2 = CSPNAccelerate(kernel_size=3, dilation=2, padding=2, stride=1)
        self.CSPN5_s2 = CSPNAccelerate(kernel_size=5, dilation=2, padding=4, stride=1)
        self.CSPN7_s2 = CSPNAccelerate(kernel_size=7, dilation=2, padding=6, stride=1)

        # CSPN
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)
        
        #self.initialize()

    def initialize(self):
       for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #weights_init(self)

    def forward(self, rgb_half, rgb, y_half, y, sparse_half, sparse, pred):#(rgb_half, image, y_half, y, sparse_half, sparse, pred)
       
            

        rgb_cspn_half = self.rgb_cspn_input_half_reso(rgb_half, y_half)
        #print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        
        sparse_cspn_half = self.sparse_cspn_input_half_reso(sparse_half, y_half)
        #print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        
        rgb_cspn = self.rgb_cspn_input(rgb, y)
        #print(f'rgb_cspn {rgb_cspn.shape}')
        
        sparse_cspn = self.sparse_cspn_input(sparse, y)
        #print(f'sparse_cspn {sparse_cspn.shape}')
        
        
        '''
        print(f'pred {pred.shape}')
        print(f'rgb  {rgb.shape}')
        print(f'sparse {sparse.shape}')
        print(f'rgb_half {rgb_half.shape}')
        print(f'sparse_half {sparse_half.shape}')
        print(f'y {y.shape}')
        print(f'y_half {y_half.shape}')
        print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        print(f'rgb_cspn {rgb_cspn.shape}')
        print(f'sparse_cspn {sparse_cspn.shape}')
        '''


        d = sparse
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature_s1, feature_s2, coarse_depth = torch.cat((rgb_cspn, sparse_cspn), 1), torch.cat((rgb_cspn_half, sparse_cspn_half),1), pred
        #print(f'feature_s1 {feature_s1.shape}')
        #print(f'feature_s2 {feature_s2.shape}')
        
        depth = coarse_depth
        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
        mask_s2 = self.mask_layer_s2(feature_s2)
        if self.prints == True:
            print(torch_min_max(d), d.shape, torch_min_max(valid_mask), valid_mask.shape)  
            print(torch_min_max(d_s2), d_s2.shape, torch_min_max(valid_mask_s2), valid_mask_s2.shape)
            print(torch_min_max(mask_s2), mask_s2.shape)

        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2
        
        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)#LATHOS
        #print(torch_min_max(kernel_conf_s2), kernel_conf_s2.shape)
        
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        #print(torch_min_max(kernel_conf_s2), kernel_conf_s2.shape)
        
        kernel_conf3_s2 = self.nnupsample(kernel_conf_s2[:, 0:1, :, :])
        #print(torch_min_max(kernel_conf3_s2), kernel_conf3_s2.shape)
        
        kernel_conf5_s2 = self.nnupsample(kernel_conf_s2[:, 1:2, :, :])
        #print(torch_min_max(kernel_conf5_s2), kernel_conf5_s2.shape)
        
        kernel_conf7_s2 = self.nnupsample(kernel_conf_s2[:, 2:3, :, :])
        #print(torch_min_max(kernel_conf7_s2), kernel_conf7_s2.shape)

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)

        depth_s2 = self.nnupsample(d_s2)
        mask_s2 = self.nnupsample(mask_s2)
        depth3 = depth5 = depth7 = depth

        mask = self.mask_layer(feature_s1)
        mask = torch.sigmoid(mask)
        mask = mask * valid_mask

        kernel_conf = self.kernel_conf_layer(feature_s1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        guide3 = self.iter_guide_layer3(feature_s1)
        guide5 = self.iter_guide_layer5(feature_s1)
        guide7 = self.iter_guide_layer7(feature_s1)
        
        if self.prints == True:
            print(f'guide3 {torch_min_max(guide3)} {guide3.shape}')
            print(f'guide5 {torch_min_max(guide5)} {guide5.shape}')
            print(f'guide7 {torch_min_max(guide7)} {guide7.shape}')

        guide3 = kernel_trans(guide3, self.encoder3)
        #print(f'guide3_afta {torch_min_max(guide3)} {guide3.shape}')
        
        guide5 = kernel_trans(guide5, self.encoder5)
        #print(f'guide5_afta {torch_min_max(guide5)} {guide5.shape}')
        
        guide7 = kernel_trans(guide7, self.encoder7)
        
        if self.prints == True:
            
            print(f'guide3_afta {torch_min_max(guide3)} {guide3.shape}')
            print(f'guide5_afta {torch_min_max(guide5)} {guide5.shape}')
            print(f'guide7_afta {torch_min_max(guide7)} {guide7.shape}')

        guide3_s2 = kernel_trans(guide3_s2, self.encoder3)
        guide5_s2 = kernel_trans(guide5_s2, self.encoder5)
        guide7_s2 = kernel_trans(guide7_s2, self.encoder7)

        guide3_s2 = self.nnupsample(guide3_s2)
        guide5_s2 = self.nnupsample(guide5_s2)
        guide7_s2 = self.nnupsample(guide7_s2)

        for i in range(1):
            #print(f'guide3_s2 and other shapes {guide3_s2.shape} {depth3.shape} {coarse_depth.shape}')
            depth3 = self.CSPN3_s2(guide3_s2, depth3, coarse_depth)
            depth3 = mask_s2*depth_s2 + (1-mask_s2)*depth3
            depth5 = self.CSPN5_s2(guide5_s2, depth5, coarse_depth)
            depth5 = mask_s2*depth_s2 + (1-mask_s2)*depth5
            depth7 = self.CSPN7_s2(guide7_s2, depth7, coarse_depth)
            depth7 = mask_s2*depth_s2 + (1-mask_s2)*depth7

        depth_s2 = kernel_conf3_s2*depth3 + kernel_conf5_s2*depth5 + kernel_conf7_s2*depth7
        refined_depth_s2 = depth_s2

        depth3 = depth5 = depth7 = refined_depth_s2

        #prop
        for i in range(1):
            #print(f'guide3 and other shapes {guide3.shape} {depth3.shape} {depth_s2.shape}')
            
            depth3 = self.CSPN3(guide3, depth3, depth_s2)
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth_s2)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth_s2)
            depth7 = mask*d + (1-mask)*depth7

        

        
        
        refined_depth = kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7
        #print(f'refined_depth.shape {refined_depth.shape}'
        if self.prints == True:
        
            print(f'kernel_conf3 {kernel_conf3[0][0][0][0]}')
            print(f'depth3 {depth3[0][0][0][0]}')
            print(f'kernel_conf5 {kernel_conf5[0][0][0][0]}')
            print(f'depth5 {depth5[0][0][0][0]}')
            print(f'kernel_conf7 {kernel_conf7[0][0][0][0]}')
            print(f'depth7 {depth7[0][0][0][0]}')
        
            print(f'refined_depth {refined_depth[0][0][0][0]}')
        
        return refined_depth
    
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



class Scaler(nn.Module):
    def __init__(self):
        super(Scaler, self).__init__()

        self.feature_conv = nn.Sequential(
            nn.Conv2d(1, 16,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2))
        
        #self.initialize()

    def initialize(self):
       for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #weights_init(self)

    def forward(self, rgb_half, image, y_half, y, sparse_half, sparse, pred):
        
        x = self.feature_conv(pred)
        

        return x




class DepthRefinement_p1(nn.Module):
    def __init__(self, 
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DepthRefinement_p1, self).__init__()

        self.rgb_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        self.rgb_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.sparse_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.mask_layer = convbn(64, 3)

        self.kernel_conf_layer = convbn(64, 3)
        self.iter_conf_layer = convbn(64, 12)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(64, 7)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(3)
        self.CSPN5 = CSPNAccelerate(5, padding=2)
        self.CSPN7 = CSPNAccelerate(7, padding=3)

        # CSPN new
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)

        weights_init(self)

    def forward(self, rgb_half, rgb, y_half, y, sparse_half, sparse, pred):#(rgb_half, image, y_half, y, sparse_half, sparse, pred)
        #rgb = input['rgb']
        rgb_cspn_half = self.rgb_cspn_input_half_reso(rgb_half, y_half)
        #print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        
        sparse_cspn_half = self.sparse_cspn_input_half_reso(sparse_half, y_half)
        #print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        
        rgb_cspn = self.rgb_cspn_input(rgb, y)
        #print(f'rgb_cspn {rgb_cspn.shape}')
        
        sparse_cspn = self.sparse_cspn_input(sparse, y)
        #print(f'sparse_cspn {sparse_cspn.shape}')
        
        
        '''
        print(f'pred {pred.shape}')
        print(f'rgb  {rgb.shape}')
        print(f'sparse {sparse.shape}')
        print(f'rgb_half {rgb_half.shape}')
        print(f'sparse_half {sparse_half.shape}')
        print(f'y {y.shape}')
        print(f'y_half {y_half.shape}')
        print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        print(f'rgb_cspn {rgb_cspn.shape}')
        print(f'sparse_cspn {sparse_cspn.shape}')
        '''


        d = sparse
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature, coarse_depth = torch.cat((rgb_cspn, sparse_cspn), 1),pred# torch.cat((rgb_cspn_half, sparse_cspn_half),1), pred
        #print(f'feature_s1 {feature_s1.shape}')
        #print(f'feature_s2 {feature_s2.shape}')
        
        depth = coarse_depth
        
        #d = input['d']
        #valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        #feature, coarse_depth= self.backbone(input)

        mask = self.mask_layer(feature)
        mask = torch.sigmoid(mask)

        mask = mask*valid_mask
        mask3 = mask[:, 0:1, :, :]
        mask5 = mask[:, 1:2, :, :]
        mask7 = mask[:, 2:3, :, :]

        kernel_conf = self.kernel_conf_layer(feature)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        conf = self.iter_conf_layer(feature)
        conf3 = conf[:, 0:4, :, :]
        conf5 = conf[:, 4:8, :, :]
        conf7 = conf[:, 8:12, :, :]
        conf3 = self.softmax(conf3)
        conf5 = self.softmax(conf5)
        conf7 = self.softmax(conf7)

        guide3 = self.iter_guide_layer3(feature)
        guide5 = self.iter_guide_layer5(feature)
        guide7 = self.iter_guide_layer7(feature)

        #init
        depth = coarse_depth
        depth3 = depth
        depth5 = depth
        depth7 = depth

        d3_list = [i for i in range(4)]
        d5_list = [i for i in range(4)]
        d7_list = [i for i in range(4)]

        #prop
        guide3 = kernel_trans(guide3, self.encoder3)
        guide5 = kernel_trans(guide5, self.encoder5)
        guide7 = kernel_trans(guide7, self.encoder7)

        for i in range(12):
            depth3 = self.CSPN3(guide3, depth3, depth)
            depth3 = mask3*d + (1-mask3)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask5*d + (1-mask5)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask7*d + (1-mask7)*depth7

            if(i==2):
                d3_list[0] = depth3
                d5_list[0] = depth5
                d7_list[0] = depth7

            if(i==5):
                d3_list[1] = depth3
                d5_list[1] = depth5
                d7_list[1] = depth7

            if(i==8):
                d3_list[2] = depth3
                d5_list[2] = depth5
                d7_list[2] = depth7

            if(i==11):
                d3_list[3] = depth3
                d5_list[3] = depth5
                d7_list[3] = depth7

        refined_depth = \
        d3_list[0] * (kernel_conf3 * conf3[:, 0:1, :, :]) + \
        d3_list[1] * (kernel_conf3 * conf3[:, 1:2, :, :]) + \
        d3_list[2] * (kernel_conf3 * conf3[:, 2:3, :, :]) + \
        d3_list[3] * (kernel_conf3 * conf3[:, 3:4, :, :]) + \
        d5_list[0] * (kernel_conf5 * conf5[:, 0:1, :, :]) + \
        d5_list[1] * (kernel_conf5 * conf5[:, 1:2, :, :]) + \
        d5_list[2] * (kernel_conf5 * conf5[:, 2:3, :, :]) + \
        d5_list[3] * (kernel_conf5 * conf5[:, 3:4, :, :]) + \
        d7_list[0] * (kernel_conf7 * conf7[:, 0:1, :, :]) + \
        d7_list[1] * (kernel_conf7 * conf7[:, 1:2, :, :]) + \
        d7_list[2] * (kernel_conf7 * conf7[:, 2:3, :, :]) + \
        d7_list[3] * (kernel_conf7 * conf7[:, 3:4, :, :])

        return refined_depth
    
    
class DepthRefinementSmall(nn.Module):
    def __init__(self, 
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DepthRefinementSmall, self).__init__()

        self.rgb_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        self.rgb_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_cspn_input = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=32,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.sparse_cspn_input_half_reso = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=64,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=1,
                                   guidance_type="full")

        self.kernel_conf_layer = convbn(64, 3)
        self.mask_layer = convbn(64, 1)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(64, 7)

        self.kernel_conf_layer_s2 = convbn(128, 3)
        self.mask_layer_s2 = convbn(128, 1)
        self.iter_guide_layer3_s2 = CSPNGenerateAccelerate(128, 3)
        self.iter_guide_layer5_s2 = CSPNGenerateAccelerate(128, 5)
        self.iter_guide_layer7_s2 = CSPNGenerateAccelerate(128, 7)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.nnupsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)
        self.CSPN3_s2 = CSPNAccelerate(kernel_size=3, dilation=2, padding=2, stride=1)
        self.CSPN5_s2 = CSPNAccelerate(kernel_size=5, dilation=2, padding=4, stride=1)
        self.CSPN7_s2 = CSPNAccelerate(kernel_size=7, dilation=2, padding=6, stride=1)

        # CSPN
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)
        
        #self.initialize()

    def initialize(self):
       for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #weights_init(self)

    def forward(self, rgb_half, rgb, y_half, y, sparse_half, sparse, pred):#(rgb_half, image, y_half, y, sparse_half, sparse, pred)


        rgb_cspn_half = self.rgb_cspn_input_half_reso(rgb_half, y_half)
        #print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        
        sparse_cspn_half = self.sparse_cspn_input_half_reso(sparse_half, y_half)
        #print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        
        rgb_cspn = self.rgb_cspn_input(rgb, y)
        #print(f'rgb_cspn {rgb_cspn.shape}')
        
        sparse_cspn = self.sparse_cspn_input(sparse, y)
        #print(f'sparse_cspn {sparse_cspn.shape}')
        
        
        '''
        print(f'pred {pred.shape}')
        print(f'rgb  {rgb.shape}')
        print(f'sparse {sparse.shape}')
        print(f'rgb_half {rgb_half.shape}')
        print(f'sparse_half {sparse_half.shape}')
        print(f'y {y.shape}')
        print(f'y_half {y_half.shape}')
        print(f'rgb_cspn_half {rgb_cspn_half.shape}')
        print(f'sparse_cspn_half {sparse_cspn_half.shape}')
        print(f'rgb_cspn {rgb_cspn.shape}')
        print(f'sparse_cspn {sparse_cspn.shape}')
        '''


        d = sparse
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature_s1, feature_s2, coarse_depth = torch.cat((rgb_cspn, sparse_cspn), 1), torch.cat((rgb_cspn_half, sparse_cspn_half),1), pred
        #print(f'feature_s1 {feature_s1.shape}')
        #print(f'feature_s2 {feature_s2.shape}')
        
        depth = coarse_depth
        print(torch_min_max(d), d.shape, torch_min_max(valid_mask), valid_mask.shape)
        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
        print(torch_min_max(d_s2), d_s2.shape, torch_min_max(valid_mask_s2), valid_mask_s2.shape)
        
        mask_s2 = self.mask_layer_s2(feature_s2)
        print(torch_min_max(mask_s2), mask_s2.shape)
       #print(f'mask_s2 {mask_s2.shape}')

        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2
        
        #print(f'mask_s2 {mask_s2.shape}')
        print(torch_min_max(mask_s2), mask_s2.shape)
        
        

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)#LATHOS
        print(torch_min_max(kernel_conf_s2), kernel_conf_s2.shape)
        
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        print(torch_min_max(kernel_conf_s2), kernel_conf_s2.shape)
        
        kernel_conf3_s2 = self.nnupsample(kernel_conf_s2[:, 0:1, :, :])
        print('conf3s2',torch_min_max(kernel_conf3_s2), kernel_conf3_s2.shape)
        
        kernel_conf5_s2 = self.nnupsample(kernel_conf_s2[:, 1:2, :, :])
        print(torch_min_max(kernel_conf5_s2), kernel_conf5_s2.shape)
        
        kernel_conf7_s2 = self.nnupsample(kernel_conf_s2[:, 2:3, :, :])
        print(torch_min_max(kernel_conf7_s2), kernel_conf7_s2.shape)

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)

        depth_s2 = self.nnupsample(d_s2)
        mask_s2 = self.nnupsample(mask_s2)
        depth3 = depth5 = depth7 = depth

        mask = self.mask_layer(feature_s1)
        mask = torch.sigmoid(mask)
        mask = mask * valid_mask

        kernel_conf = self.kernel_conf_layer(feature_s1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        guide3 = self.iter_guide_layer3(feature_s1)
        guide5 = self.iter_guide_layer5(feature_s1)
        guide7 = self.iter_guide_layer7(feature_s1)
       
        print(f'guide3 {torch_min_max(guide3)} {guide3.shape}')
        print(f'guide5 {torch_min_max(guide5)} {guide5.shape}')
        print(f'guide7 {torch_min_max(guide7)} {guide7.shape}')

        guide3 = kernel_trans(guide3, self.encoder3)
        print(f'guide3_afta {torch_min_max(guide3)} {guide3.shape}')
        
        guide5 = kernel_trans(guide5, self.encoder5)
        print(f'guide5_afta {torch_min_max(guide5)} {guide5.shape}')
        
        guide7 = kernel_trans(guide7, self.encoder7)
        print(f'guide7_afta {torch_min_max(guide7)} {guide7.shape}')

        guide3_s2 = kernel_trans(guide3_s2, self.encoder3)
        guide5_s2 = kernel_trans(guide5_s2, self.encoder5)
        guide7_s2 = kernel_trans(guide7_s2, self.encoder7)

        guide3_s2 = self.nnupsample(guide3_s2)
        guide5_s2 = self.nnupsample(guide5_s2)
        guide7_s2 = self.nnupsample(guide7_s2)

        for i in range(1):
            print(f'guide3_s2 and other shapes {guide3_s2.shape} {depth3.shape} {coarse_depth.shape}')
            depth3 = self.CSPN3_s2(guide3_s2, depth3, coarse_depth)
            depth3 = mask_s2*depth_s2 + (1-mask_s2)*depth3
            depth5 = self.CSPN5_s2(guide5_s2, depth5, coarse_depth)
            depth5 = mask_s2*depth_s2 + (1-mask_s2)*depth5
            depth7 = self.CSPN7_s2(guide7_s2, depth7, coarse_depth)
            depth7 = mask_s2*depth_s2 + (1-mask_s2)*depth7

        depth_s2 = kernel_conf3_s2*depth3 + kernel_conf5_s2*depth5 + kernel_conf7_s2*depth7
        refined_depth_s2 = depth_s2

        depth3 = depth5 = depth7 = refined_depth_s2

        #prop
        for i in range(1):
            print(f'guide3 and other shapes {guide3.shape} {depth3.shape} {depth_s2.shape}')
            
            depth3 = self.CSPN3(guide3, depth3, depth_s2)
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth_s2)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth_s2)
            depth7 = mask*d + (1-mask)*depth7

        
        print(f'kernel_conf3 {kernel_conf3[0][0][0][0]}')
        print(f'depth3 {depth3[0][0][0][0]}')
        print(f'kernel_conf5 {kernel_conf5[0][0][0][0]}')
        print(f'depth5 {depth5[0][0][0][0]}')
        print(f'kernel_conf7 {kernel_conf7[0][0][0][0]}')
        print(f'depth7 {depth7[0][0][0][0]}')
        
        
        
        refined_depth = kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7
        #print(f'refined_depth.shape {refined_depth.shape}')
        print(f'refined_depth {refined_depth[0][0][0][0]}')
        
        return refined_depth
    
    
    
class DecnetModule(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DecnetModule, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = Decnet_Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   rgb_guide_features=3,
                                   sparse_guide_features=1,
                                   guidance_type="full")
        self.up_2 = Decnet_Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   rgb_guide_features=3,
                                   sparse_guide_features=1,
                                   guidance_type="full")
        self.up_3 = Decnet_Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   rgb_guide_features=3,
                                   sparse_guide_features=1,
                                   guidance_type="full")


    def forward(self, rgb, sparse):
        y = self.feature_extractor(rgb)
        #print('tara')
        
        #rgbd = torch.cat((rgb,depth),1)

        x_half = F.interpolate(rgb, scale_factor=.5)
        x_quarter = F.interpolate(rgb, scale_factor=.25)
        
        sparse_half = F.interpolate(sparse, scale_factor=.5)
        sparse_quarter = F.interpolate(sparse, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter,sparse_quarter, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half,sparse_half, y)

        y = F.interpolate(y, scale_factor=2., mode='bilinear')#, align_corners=True)
        y = self.up_3(rgb, sparse, y)
        return y


class DecnetDepthRefinement(nn.Module):
    def __init__(self, 
            pretrained=False,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DecnetDepthRefinement, self).__init__()

        #self.conv1 = dilated_conv3x3bn(1,16)
        
        #self.conv2 = dilated_conv3x3bn(16,32)
        
        #self.conv3 = convbnrelu(32,64)
        
        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained,
                features_n = 1, 
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
                                   guide_features=1,
                                   guidance_type="full")
        
    

    def forward(self, basepred, sparse):
        #print(f'basepred {basepred.shape}')
        #print(f'sparse {sparse.shape}')
        y = self.feature_extractor(sparse)
        y2 = F.interpolate(y, scale_factor=2)
        # y1 = self.conv1(sparse)
        #print(f'y1 {y1.shape}')
                
        #y2 = self.conv2(y1)
        #print(f'y2 {y2.shape}')
        
        #y2 = self.conv3(y2)

        
        
        #print(y2.shape)
        
        
        pred_half = F.interpolate(basepred, scale_factor=.5)
        pred_quarter = F.interpolate(basepred, scale_factor=.25)

        #y3 = F.interpolate(y2, scale_factor=2., mode='bilinear')
        #print('y3', y3.shape)
        
        y3 = self.up_1(pred_quarter, y2)
        #print('y4', y3.shape)
        
        #print('self.up_1.shape', y.shape)


        y4 = F.interpolate(y3, scale_factor=2., mode='bilinear')
        y5 = self.up_2(pred_half, y4)
        
        
        y6 = F.interpolate(y5, scale_factor=2., mode='bilinear')
        y5 = self.up_3(basepred, y6)

        return y5


class DecnetSparseIncorporated(nn.Module):
    def __init__(self, 
            pretrained=False,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DecnetSparseIncorporated, self).__init__()
        

        self.feature_extractor = DualResNet_Backbone(
                pretrained=True,
                features_n = 3, 
                features=up_features[0])
        
        self.sparse_feature_extractor = DualResNet_Backbone(                
                pretrained=False,
                features_n = 1, 
                features=up_features[0])
        
        



        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_up_1 =  Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        
        self.fusion_up_half = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.fusion_up_final = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        
        
        
    

    def forward(self, rgb, sparse):
        #rgb = torch.zeros(1,3,480,640).to('cuda')
        #sparse = torch.zeros(1,1,480,640).to('cuda')
        #print(f'basepred {basepred.shape}')
        #print(f'sparse {sparse.shape}')
        y = self.feature_extractor(rgb)
        y2 = F.interpolate(y, scale_factor=2)
        
        x = self.sparse_feature_extractor(sparse)
        x2 = F.interpolate(x,scale_factor=2)
        # y1 = self.conv1(sparse)
        #print(f'y1 {y1.shape}')
                
        #y2 = self.conv2(y1)
        #print(f'y2 {y2.shape}')
        
        #y2 = self.conv3(y2)

        
        
        #print(y2.shape)
        
        
        rgb_half = F.interpolate(rgb, scale_factor=.5)
        rgb_quarter = F.interpolate(rgb, scale_factor=.25)
        
        #sparse_half = F.interpolate(sparse, scale_factor=.5)
        #sparse_quarter =  F.interpolate(sparse, scale_factor=.25)

        #y3 = F.interpolate(y2, scale_factor=2., mode='bilinear')
        #print('y3', y3.shape)
        
        y3 = self.up_1(rgb_quarter, y2)
        x3 = self.sparse_up_1(rgb_quarter, x2)
        
        #print('y4', y3.shape)
        
        #print('self.up_1.shape', y.shape)


        y4 = F.interpolate(y3, scale_factor=2., mode='bilinear')
        x4 = F.interpolate(x3, scale_factor=2., mode='bilinear')
        
        y5 = self.up_2(rgb_half, y4)
        x5 = self.sparse_up_2(rgb_half,x4)
        
        xyhalf = self.fusion_up_half(rgb_half, torch.cat((y5,x5),dim=1))
        xy = self.fusion_up_final(rgb, F.interpolate(xyhalf, scale_factor=2., mode='bilinear'))
        
        
        #y6 = F.interpolate(y5, scale_factor=2., mode='bilinear')
        #y5 = self.up_3(rgb, y6)

        return xy


class DecnetNLSPN(nn.Module):
    def __init__(self, 
            pretrained=False,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(DecnetNLSPN, self).__init__()
        

        
                
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)
        
        self.feature_extractor = DualResNet_Backbone(
                pretrained=True,
                features_n = 48, 
                features=up_features[0])
        
        self.sparse_feature_extractor = DualResNet_Backbone(                
                pretrained=False,
                features_n = 16, 
                features=up_features[0])
        
        



        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_up_1 =  Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.sparse_up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        
        self.fusion_up_half = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        self.fusion_up_final = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        
        
        
        
    

    def forward(self, rgb, sparse):
        #rgb = torch.zeros(1,3,480,640).to('cuda')
        #sparse = torch.zeros(1,1,480,640).to('cuda')
        #print(f'basepred {basepred.shape}')
        #print(f'sparse {sparse.shape}')
        

        
        y = self.feature_extractor(rgb)
        print(f'y {y.shape}')
        y2 = F.interpolate(y, scale_factor=2)
        print(f'y2 {y2.shape}')
        
        x = self.sparse_feature_extractor(sparse)
        x2 = F.interpolate(x,scale_factor=2)
        # y1 = self.conv1(sparse)
        #print(f'y1 {y1.shape}')
                
        #y2 = self.conv2(y1)
        #print(f'y2 {y2.shape}')
        
        #y2 = self.conv3(y2)

        
        
        #print(y2.shape)
        
        
        rgb_half = F.interpolate(rgb, scale_factor=.5)
        rgb_quarter = F.interpolate(rgb, scale_factor=.25)
        
        #sparse_half = F.interpolate(sparse, scale_factor=.5)
        #sparse_quarter =  F.interpolate(sparse, scale_factor=.25)

        #y3 = F.interpolate(y2, scale_factor=2., mode='bilinear')
        #print('y3', y3.shape)
        
        y3 = self.up_1(rgb_quarter, y2)
        x3 = self.sparse_up_1(rgb_quarter, x2)
        
        #print('y4', y3.shape)
        
        #print('self.up_1.shape', y.shape)


        y4 = F.interpolate(y3, scale_factor=2., mode='bilinear')
        x4 = F.interpolate(x3, scale_factor=2., mode='bilinear')
        
        y5 = self.up_2(rgb_half, y4)
        x5 = self.sparse_up_2(rgb_half,x4)
        
        xyhalf = self.fusion_up_half(rgb_half, torch.cat((y5,x5),dim=1))
        xy = self.fusion_up_final(rgb, F.interpolate(xyhalf, scale_factor=2., mode='bilinear'))
        
        
        #y6 = F.interpolate(y5, scale_factor=2., mode='bilinear')
        #y5 = self.up_3(rgb, y6)

        return xy
