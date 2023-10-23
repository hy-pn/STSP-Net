from pyexpat.errors import XML_ERROR_SYNTAX
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet50(pretrained,replace_stride_with_dilation=[False,True,True])
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + out
        out = self.relu(out)

        return out
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.se1 =SEModel(64)
        self.se2 =SEModel(64)
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=1, stride=1, padding=0)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        # feat = self.conv3(feat)
        # feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context
class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d
class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=4, activation=nn.ReLU,scale=1):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.scale = scale
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        # print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x1 = x
        x = self.pool(x)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width*self.ds,height*self.ds])
        out = out + x1

        return out
class STSPNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(STSPNet, self).__init__()
        s = 4
        self.FCN = FCN(in_channels, pretrained=True)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.sp = SpatialPath()
        self.downsampele2 = nn.Sequential(nn.Conv2d(64,128*s,3,2,1),nn.BatchNorm2d(128*s), nn.ReLU())  
        self.downsampele3 = nn.Sequential(nn.Conv2d(64,256*s,3,2,1),nn.BatchNorm2d(256*s), nn.ReLU())  
        self.downsampele4 = nn.Sequential(nn.Conv2d(128,512*s,3,2,1),nn.BatchNorm2d(512*s), nn.ReLU())  
        self.up2 = nn.Sequential(nn.Conv2d(128*s,64,1,1),nn.BatchNorm2d(64), nn.ReLU())  
        self.up3 = nn.Sequential(nn.Conv2d(256*s,64,1,1),nn.BatchNorm2d(64), nn.ReLU())  
        self.up4 = nn.Sequential(nn.Conv2d(512*s,128,1,1),nn.BatchNorm2d(128), nn.ReLU())  
        self.convblk1 = ConvBNReLU(256, 128)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.resCD1 = self._make_layer(ResBlock, 256, 128, 2, stride=1)
        self.resCD2 = self._make_layer(ResBlock, 256, 128, 4, stride=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
        # self.convblk = ConvBNReLU(256, 256, ks=1, stride=1, padding=0)
        self.head1 = nn.Sequential(nn.Conv2d(512*s, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        self.head2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())    
        self.ocr_gather_head = SpatialGather_Module(cls_num = 2)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=256,
                                                 key_channels=128,
                                                 out_channels=256,
                                                 scale=1,
                                                 dropout=0.1,
                                                 )
        self.bam = BAM(128,scale=1)
        self.bam1 = BAM(128,scale=2)
        self.convblk = nn.Conv2d(256, 128, kernel_size=1, padding=0,bias=False)
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, 128,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.aux_head,self.ocr_gather_head,self.ocr_distri_head,self.convblk,self.head1,self.head2, self.resCD1,self.resCD2,self.convblk1,self.classifier1, self.classifier2, 
                           self.classifierCD,self.downsampele2,self.downsampele3,self.downsampele4,
                        self.up2, self.up3, self.up4)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        x_s1 = self.sp.conv1(x)
        x_s2 = self.sp.conv2(x_s1)
        x = self.FCN.layer0(x) #size:1/4
        x = self.FCN.maxpool(x) #size:1/4
        x1 = self.FCN.layer1(x) #size:1/4
        x2 = self.FCN.layer2(x1) #size:1/8
        temp1 =x_s2+self.sp.se1(self.upsample(self.up2(x2)))
        x_s3 = self.sp.conv3(temp1)
        x2 = x2 + self.downsampele2(x_s2)
        x3 = self.FCN.layer3(x2) #size:1/16
        temp2 =x_s3+self.sp.se2(self.upsample(self.up3(x3)))
        x_s4 = self.sp.conv_out(temp2)
        x3 = x3 + self.downsampele3(x_s3)
        x4 = self.FCN.layer4(x3)
        x4 = x4 + self.downsampele4(x_s4)
        x = self.head1(x4)
        x_s4 = self.head2(x_s4)
        return x, x_s4
    
    def CD_forward(self, x1, x2):
        x = torch.cat([x1,x2], 1)
        coarse = self.resCD1(x)
        regionc = self.aux_head(coarse)
        context1 = self.ocr_gather_head(x, regionc)
        xf = self.ocr_distri_head(x, context1)
        change = self.resCD2(xf)
        change = self.classifierCD(change)
        return change
    
    def forward(self, x1, x2):
        x_size = x1.size()
        x1, x1_l4= self.base_forward(x1)
        x2, x2_l4= self.base_forward(x2)
        x11 = self.bam(x1)
        x21 = self.bam(x2)
        region11 = self.bam1(x1_l4)
        region21 = self.bam1(x2_l4)
        x12 = torch.cat([self.upsample(x11),region11],1)
        x22 = torch.cat([self.upsample(x21),region21],1)
        x12 = self.convblk(x12)
        x22 = self.convblk(x22)
        out1 = self.classifier1((x12))
        out2 = self.classifier2((x22))
        # windows
        # dist = F.pairwise_distance(x12,x22,eps=0.,keepdim=True)
        # Linux
        d1 = x12.permute(0, 2, 3,1)
        d2 = x22.permute(0, 2, 3,1)
        dist = F.pairwise_distance(d1,d2,eps=0.,keepdim=True)
        dist = dist.permute(0, 3, 1,2)
        change = self.CD_forward(x1, x2)
        
        return F.upsample(change, x_size[2:], mode='bilinear'), F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'), F.upsample(dist, x_size[2:], mode='bilinear')
if __name__ == '__main__':
    model = STSPNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x1 = torch.rand(1,3, 512, 512).cuda()
    x2 = torch.rand(1,3, 512, 512).cuda()
    tensor = (x1, x2)
    from torchvision.models import resnet50
    from thop import profile
    flops, params = profile(model, inputs=tensor)
    print("FLOPs: ", flops)
    print("params: ", params/1024/1024)
