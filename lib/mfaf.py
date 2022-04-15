# from __future__ import division
# import os
# import numpy as np
# import torch
import torch.nn as nn
# from torch.nn.functional import upsample,normalize
# import numpy as np
import torch


# import math
# from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
#     NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
# from torch.nn import functional as F
# from torch.autograd import Variable


class APA_Module(nn.Module):
    """ 3D Feature Alignment Position attention module"""

    def __init__(self, in_dim):
        super(APA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_RGB):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x_RGB).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_RGB).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.alpha * out + x
        return out


class ACA_Module(nn.Module):
    """  3D Feature Alignment Channel attention module"""

    def __init__(self, in_dim):
        super(ACA_Module, self).__init__()
        self.chanel_in = in_dim

        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_RGB):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x_RGB.view(m_batchsize, C, -1)
        proj_key = x_RGB.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.beta * out + x
        return out


class MFPA_Module(nn.Module):
    """ Multi-modality Fusion Position attention module"""

    def __init__(self, in_dim):
        super(MFPA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.lam = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_RGB):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x_RGB).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.lam * out + x
        return out


class MFCA_Module(nn.Module):
    """ Multi-modality Fusion Channel attention module"""

    def __init__(self, in_dim):
        super(MFCA_Module, self).__init__()
        self.chanel_in = in_dim

        self.delta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_RGB):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x_RGB.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.delta * out + x
        return out


class MFAF(nn.Module):
    def __init__(self, in_channels=256, out_channels=256 // 4, norm_layer=nn.BatchNorm2d):
        super(MFAF, self).__init__()
        inter_channels = in_channels // 4
        self.conv_ap_3d = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU(True))
        self.conv_ap_RGB = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True))
        self.conv_ac_3d = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU(True))
        self.conv_ac_RGB = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True))
        self.ap = APA_Module(inter_channels)
        self.ac = ACA_Module(inter_channels)
        self.conv_ap = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv_ac = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))

        self.conv_fp_3d = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU(True))
        self.conv_fp_RGB = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True))
        self.conv_fc_3d = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU(True))
        self.conv_fc_RGB = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True))
        self.fp = MFPA_Module(inter_channels)
        self.fc = MFCA_Module(inter_channels)
        self.conv_fp = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv_fc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))

        self.conv_a = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(True))

        self.conv_f = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(True)
                                    )

    def forward(self, x, x_rgb):
        feat_ap_3d = self.conv_ap_3d(x)
        feat_ap_rgb = self.conv_ap_RGB(x_rgb)

        feat_ap = self.ap(feat_ap_3d, feat_ap_rgb)
        feat_ap = self.conv_ap(feat_ap)

        feat_ac_3d = self.conv_ac_3d(x)
        feat_ac_rgb = self.conv_ac_RGB(x_rgb)
        feat_ac = self.ac(feat_ac_3d, feat_ac_rgb)
        feat_ac = self.conv_ac(feat_ac)

        feat_a = feat_ap + feat_ac
        feat_a = self.conv_a(feat_a)

        feat_fp_3d = self.conv_fp_3d(feat_a)
        feat_fp_rgb = self.conv_fp_RGB(x_rgb)
        feat_fp = self.fp(feat_fp_3d, feat_fp_rgb)
        feat_fp = self.conv_fp(feat_fp)

        feat_fc_3d = self.conv_fc_3d(feat_a)
        feat_fc_rgb = self.conv_fc_RGB(x_rgb)
        feat_fc = self.fc(feat_fc_3d, feat_fc_rgb)
        feat_fc = self.conv_fc(feat_fc)

        feat_f = feat_fp + feat_fc
        output = self.conv_f(feat_f)

        return output
