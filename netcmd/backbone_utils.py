#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from .fpn import FeaturePyramidNetwork, MaxpoolOnP5, LastLevelMaxPool, LastLevelP6P7


class SwinLayerGetter(nn.Module):
    '''Swin模型的分层特征提取器'''
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
        self.out_dict = OrderedDict()

    def forward(self,x):
        x = self.model.features.get_submodule('0')(x)
        x = self.model.features.get_submodule('1')(x)
        C2 = x.permute(0,3,1,2).contiguous()
        self.out_dict["0"] = C2
        x = self.model.features.get_submodule('2')(x)
        x = self.model.features.get_submodule('3')(x)
        C3 = x.permute(0,3,1,2).contiguous()
        self.out_dict["1"] = C3
        x = self.model.features.get_submodule('4')(x)
        x = self.model.features.get_submodule('5')(x)
        C4 = x.permute(0,3,1,2).contiguous()
        self.out_dict["2"] = C4
        x = self.model.features.get_submodule('6')(x)
        x = self.model.features.get_submodule('7')(x)
        C5 = x.permute(0,3,1,2).contiguous()
        self.out_dict["3"] = C5
        return self.out_dict

class FPNForSwin(nn.Module):
    ''''''
    def __init__(self, backbone, in_channels_list, out_channel, extra_type=None):
        super().__init__()
        self.body = SwinLayerGetter(backbone)
        # faster,cascade,sparse
        if extra_type == 'maxpool':
            self.fpn = FeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=MaxpoolOnP5(),
                                             extra_type=extra_type)
        # fcos, retinanet
        elif extra_type == 'last':
            self.fpn = FeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=LastLevelP6P7(in_channels_list[-1],256),
                                             extra_type=extra_type)     
        self.out_channels = out_channel

    def forward(self,x):
        x = self.body(x)
        x = self.fpn(x)
        return x
    

def swin_fpn_backbone(extra_type='maxpool'):
    backbone = models.swin_s(weights = models.Swin_S_Weights.DEFAULT)

    # in_channels_list = [128,256,512,1024]  # swin-b
    in_channels_list = [96,192,384,768]  # swin-s
    out_channel = 256
    return FPNForSwin(backbone, in_channels_list, out_channel, extra_type=extra_type)


