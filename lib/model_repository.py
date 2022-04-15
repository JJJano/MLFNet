from torch import nn
import torch
from torch.nn import functional as F
from lib.resnet import resnet18, resnet50, resnet34
from lib.resnet_depth_normal import resnet18_depth_normal
from lib.mfaf import MFAF
import pdb
import cv2


class Resnet18_8s(nn.Module):
    def __init__(self, num_keypoints=8, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18_8s, self).__init__()
        self.mfaf= MFAF()
        self.mfaf_fc = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        resnet18_8s_depth_normal = resnet18_depth_normal(fully_conv=True,
                                                         pretrained=False,
                                                         output_stride=8,
                                                         remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s_depth_normal.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s_depth_normal.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s_depth_normal = resnet18_8s_depth_normal
        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.conv8s_depth_normal = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up8sto4s_depth_normal = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )
        self.conv4s_depth_normal = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4sto2s_depth_normal = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.conv2s_depth_normal = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2storaw_depth_normal = nn.UpsamplingBilinear2d(scale_factor=2)

        self.num_edges = num_keypoints * (num_keypoints - 1) // 2
        out_channels = 2 + 1 + num_keypoints * 2 + self.num_edges * 2
        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, out_channels, 1, 1)
        )
        self.convraw_depth_normal = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, 7, 1, 1)
        )

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def masked_smooth_l1_loss(self, map_pred, map_target, mask, sigma=1.0, normalize=True, reduce=True):
        # based on: https://github.com/zju3dv/pvnet/blob/master/lib/utils/net_utils.py
        bs, c = map_pred.shape[:2]
        sigma_2 = sigma ** 2
        map_diff = map_pred - map_target
        diff = mask * map_diff
        abs_diff = torch.abs(diff)
        sign = (abs_diff < 1. / sigma_2).detach().float()
        in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * sign + \
                  (abs_diff - (0.5 / sigma_2)) * (1. - sign)
        if normalize:
            in_loss = torch.sum(in_loss.view(bs, -1), 1) / (c * torch.sum(mask.view(bs, -1), 1) + 1e-3)
        if reduce:
            in_loss = torch.mean(in_loss)
        return in_loss

    def forward(self, image, sym_cor, mask, pts2d_map, graph=None, obj_depth=None, obj_normal=None, epoch=None):
        depth_mask = (obj_depth[:, 2, :, :] != 0.).to(torch.float32).unsqueeze(1)
        depth_mask = depth_mask * mask
        normal_mask = (obj_normal[:, 2, :, :] != 0.).to(torch.float32).unsqueeze(1)
        normal_mask = normal_mask * mask

        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(image)

        fm_depth_normal = self.conv8s_depth_normal(torch.cat([xfc, x8s], 1))
        fm_depth_normal = self.up8sto4s_depth_normal(fm_depth_normal)

        fm_depth_normal = self.conv4s_depth_normal(torch.cat([fm_depth_normal, x4s], 1))
        fm_depth_normal = self.up4sto2s_depth_normal(fm_depth_normal)

        fm_depth_normal = self.conv2s_depth_normal(torch.cat([fm_depth_normal, x2s], 1))
        fm_depth_normal = self.up2storaw_depth_normal(fm_depth_normal)

        x_mask_depth_normal = self.convraw_depth_normal(torch.cat([fm_depth_normal, image], 1))

        x_mask_pred = F.sigmoid(x_mask_depth_normal[:, 0, :, :].unsqueeze(1))
        x_mask_pred_onehot = (x_mask_pred > 0.5).to(torch.float).detach()

        x_depth_normal_pred = x_mask_depth_normal[:, 1:, :, :]
        if epoch < 200:
            xfc_depth_normal = self.resnet18_8s_depth_normal(torch.cat([obj_depth, obj_normal], dim=1) * mask)[5]
        else:
            xfc_depth_normal = self.resnet18_8s_depth_normal(x_depth_normal_pred.detach() * x_mask_pred_onehot)[5]

        xfc_all = self.mfaf_fc(torch.cat([self.mfaf(xfc_depth_normal, xfc), xfc], dim=1))

        x_mask_loss = F.binary_cross_entropy(x_mask_pred, mask)
        obj_depth_loss = self.masked_smooth_l1_loss(x_depth_normal_pred[:, :3, :, :], obj_depth, depth_mask)
        obj_normal_loss = self.masked_smooth_l1_loss(x_depth_normal_pred[:, 3:, :, :], obj_normal, normal_mask)

        fm = self.conv8s(torch.cat([xfc_all, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        x = self.convraw(torch.cat([fm, image], 1))

        sym_cor_pred = x[:, :2]
        mask_pred = F.sigmoid(x[:, 2:3])

        pts2d_map_pred = x[:, 3:-(self.num_edges * 2)]
        graph_pred = x[:, -self.num_edges * 2:]

        graph_loss = self.masked_smooth_l1_loss(graph, graph_pred, mask)
        sym_cor_loss = self.masked_smooth_l1_loss(sym_cor_pred, sym_cor, mask)
        mask_loss = F.binary_cross_entropy(mask_pred, mask)
        pts2d_loss = self.masked_smooth_l1_loss(pts2d_map_pred, pts2d_map, mask)

        return \
            sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss, \
            x_mask_loss, obj_depth_loss, obj_normal_loss, x_depth_normal_pred[:, :3, :, :] * x_mask_pred_onehot, \
            x_depth_normal_pred[:, 3:, :, :] * x_mask_pred_onehot
