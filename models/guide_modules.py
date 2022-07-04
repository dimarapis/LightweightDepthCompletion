import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

class SELayer(nn.Module):
    """
    Taken from:
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2,3]) # Replacement of avgPool for large kernels for trt
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand(x.shape)



class Guided_Upsampling_Block(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guidance_type='full', guide_features=3):
        super(Guided_Upsampling_Block, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.guide_features = guide_features
        self.in_features = in_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),  
            nn.ReLU(inplace=True))

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                nn.Conv2d(self.guide_features, expand_features,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True))

            comb_features = (expand_features // 2) * 2
        elif self.guidance_type =='raw':
            comb_features = expand_features // 2 + guide_features
        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        self.reduce = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=1)

        if self.channel_attention:
            self.SE_block = SELayer(comb_features,
                                    reduction=1)
            
            
        #self.initialize()

    #def initialize(self):
    #   for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            nn.init.kaiming_normal_(m.weight)
    #        elif isinstance(m, nn.BatchNorm2d):
    #            nn.init.constant_(m.weight, 1)
    #            nn.init.constant_(m.bias, 0)
        

    def forward(self, guide, depth):
        x = self.feature_conv(depth)

        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            xy = torch.cat([x, y], dim=1)
        elif self.guidance_type == 'raw':
            xy = torch.cat([x, guide], dim=1)
        else:
            xy = x

        if self.channel_attention:
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + depth)



class AuxSparseUpsamplingBlock(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guidance_type='full', guide_features=3):
        super(AuxSparseUpsamplingBlock, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.guide_features = guide_features
        self.in_features = in_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),  
            nn.ReLU(inplace=True))

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                nn.Conv2d(self.guide_features, expand_features,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True))
        
            self.sparse_conv = nn.Sequential(
                    nn.Conv2d(1, 2*expand_features,
                            kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(2*expand_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2*expand_features, 2*expand_features // 2, kernel_size=1),
                    nn.BatchNorm2d(2*expand_features // 2),
                    nn.ReLU(inplace=True))

            comb_features = (expand_features // 2) * 2
        elif self.guidance_type =='raw':
            comb_features = expand_features // 2 + guide_features
        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        self.reduce = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=1)
        self.reduce_sparse = nn.Conv2d(2*in_features,
                                in_features,
                                kernel_size=1)

        if self.channel_attention:
            self.SE_block = SELayer(comb_features,
                                    reduction=1)
            
            
        #self.initialize()

    #def initialize(self):
    #   for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            nn.init.kaiming_normal_(m.weight)
    #        elif isinstance(m, nn.BatchNorm2d):
    #            nn.init.constant_(m.weight, 1)
    #            nn.init.constant_(m.bias, 0)
        

    def forward(self, guide, pred):
        x = self.feature_conv(pred)

        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            #z = self.sparse_conv(sparse)
            #print('z_shape', z.shape)
            xy = torch.cat([x, y], dim=1)
            
        elif self.guidance_type == 'raw':
            xy = torch.cat([x, guide], dim=1)
        else:
            xy = x

        if self.channel_attention:
            #print(xy.shape)
            #print(z.shape)
            #xy = torch.cat([xy,z], dim=1)
            #xy = self.reduce_sparse(torch.cat([xy,z], dim=1))
            #print(xy.shape)
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + pred)





class AuxUpsamplingBlock(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guidance_type='full', guide_features=3):
        super(AuxUpsamplingBlock, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.guide_features = guide_features
        self.in_features = in_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),  
            nn.ReLU(inplace=True))

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                nn.Conv2d(self.guide_features, expand_features,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True))
        
            #self.sparse_conv = nn.Sequential(
            #        nn.Conv2d(1, 2*expand_features,
            #                kernel_size=kernel_size, padding=padding),
            #        nn.BatchNorm2d(2*expand_features),
            #        nn.ReLU(inplace=True),
            #        nn.Conv2d(2*expand_features, 2*expand_features // 2, kernel_size=1),
            #        nn.BatchNorm2d(2*expand_features // 2),
            #        nn.ReLU(inplace=True))

            comb_features = (expand_features // 2) * 2
        elif self.guidance_type =='raw':
            comb_features = expand_features // 2 + guide_features
        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        self.reduce = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=1)
        #self.reduce_sparse = nn.Conv2d(2*in_features,
        #                        in_features,
        #                        kernel_size=1)

        if self.channel_attention:
            self.SE_block = SELayer(comb_features,
                                    reduction=1)
            
            
        #self.initialize()

    #def initialize(self):
    #   for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            nn.init.kaiming_normal_(m.weight)
    #        elif isinstance(m, nn.BatchNorm2d):
    #            nn.init.constant_(m.weight, 1)
    #            nn.init.constant_(m.bias, 0)
        

    def forward(self, guide, pred):
        x = self.feature_conv(pred)

        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            #z = self.sparse_conv(sparse)
            #print('z_shape', z.shape)
            xy = torch.cat([x, y], dim=1)
            
        elif self.guidance_type == 'raw':
            xy = torch.cat([x, guide], dim=1)
        else:
            xy = x

        if self.channel_attention:
            #print(xy.shape)
            #print(z.shape)
            #xy = torch.cat([xy,z], dim=1)
            #xy = self.reduce_sparse(torch.cat([xy,z], dim=1))
            #print(xy.shape)
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + pred)




class DepthCorrector(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guide_features=1):
        super(DepthCorrector, self).__init__()

        self.channel_attention = channel_attention
        self.guide_features = guide_features
        self.in_features = in_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),  
            nn.ReLU(inplace=True))


        self.guide_conv = nn.Sequential(
            nn.Conv2d(self.guide_features, expand_features,
                        kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),
            nn.ReLU(inplace=True))
    
        comb_features = (expand_features // 2) * 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        self.reduce = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=1)

        if self.channel_attention:
            self.SE_block = SELayer(comb_features,
                                    reduction=1)
            


    def forward(self, guide, main):
        x = self.feature_conv(main)
        y = self.guide_conv(guide)
        xy = torch.cat([x, y], dim=1)

        if self.channel_attention:
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + main)


class MinkoEncoder(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guide_features=1):
        super(DepthCorrector, self).__init__()

        self.channel_attention = channel_attention
        self.guide_features = guide_features
        self.in_features = in_features

        #padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_features, expand_features,
                      kernel_size=kernel_size, dimension=2),
            ME.MinkowskiBatchNorm(expand_features),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_features, expand_features,
                      kernel_size=kernel_size, dimension=2),
            ME.MinkowskiBatchNorm(expand_features),
            ME.MinkowskiELU())


        self.guide_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_features, expand_features,
                      kernel_size=kernel_size, dimension=2),
            ME.MinkowskiBatchNorm(expand_features),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_features, expand_features,
                      kernel_size=kernel_size, dimension=2),
            ME.MinkowskiBatchNorm(expand_features),
            ME.MinkowskiELU())
    
        comb_features = (expand_features // 2) * 2

        self.comb_conv = nn.Sequential(
            ME.MinkowskiConvolution(comb_features, expand_features,
                      kernel_size=kernel_size, dimension=2),
            ME.MinkowskiBatchNorm(expand_features),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(expand_features, in_features, kernel_size=1, dimension=2),
            ME.MinkowskiBatchNorm(in_features),
            ME.MinkowskiELU())



        self.reduce = ME.MinkowskiConvolution(in_features,
                                out_features,
                                kernel_size=1, 
                                dimension=2)

        #if self.channel_attention:
        #    self.SE_block = SELayer(comb_features,
        #                           reduction=1)
            


    def forward(self, guide, main):
        x = self.feature_conv(main)
        y = self.guide_conv(guide)
        xy = torch.cat([x, y], dim=1)

        #if self.channel_attention:
        #    xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + main)