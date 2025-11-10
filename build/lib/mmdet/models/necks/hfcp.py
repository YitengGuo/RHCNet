# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from ..builder import NECKS


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.channel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class ScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(ScaleAttention, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        scale1 = self.conv1(x)
        scale2 = self.conv2(x)
        scale3 = self.conv3(x)
        scales = torch.stack([scale1, scale2, scale3], dim=1)
        attention_weights = self.softmax(scales.mean(dim=(3, 4), keepdim=True))
        attention = (attention_weights * scales).sum(dim=1)
        out = self.gamma * attention + x
        return out


class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.scale_attention = ScaleAttention(self.channel)
        self.sab = SA_Block(self.channel)

    def forward(self, x):
        cab = self.cab(x)
        scale_att = self.scale_attention(cab)
        sab = self.sab(scale_att)
        FM = sab
        return FM


###################################################################
# ######################## PAM Module #############################
###################################################################
class PAM(nn.Module):
    """
    位置注意力模块（Position Attention Module）：融合通道压缩与空间对比增强，强化显著区域响应。
    """
    def __init__(self, channels, reduction=4, kernel_size=3):
        super(PAM, self).__init__()
        self.reduce = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.expand = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.lcn = nn.GroupNorm(num_groups=channels // 8, num_channels=channels)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.reduce(x)
        z = F.relu(z, inplace=True)
        z = self.expand(z)
        z_norm = self.lcn(z)
        m = self.sa(x)
        fg = x * m
        bg = x * (1 - m)
        out = fg + self.alpha * (z_norm + bg)
        return out


###################################################################
# ######################## CGCA Module ############################
###################################################################
class CGCA(nn.Module):
    def __init__(self, in_channels, clusters=2, spatial_components=2):
        super(CGCA, self).__init__()
        self.in_channels = in_channels
        self.clusters = clusters
        self.components = spatial_components
        self.beta = nn.Parameter(torch.randn(clusters, spatial_components))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        x_reshaped = x.view(B, C, -1)
        out = torch.zeros_like(x)

        for b in range(B):
            features = x_reshaped[b].cpu().detach().numpy()
            kmeans = KMeans(n_clusters=self.clusters, n_init=1, random_state=42)
            labels = kmeans.fit_predict(features)

            for k in range(self.clusters):
                idx = torch.tensor((labels == k).nonzero()[0], dtype=torch.long).to(x.device)
                if idx.numel() == 0:
                    continue
                xk = x[b, idx, :, :].unsqueeze(0)
                xk_mean = torch.mean(xk, dim=1, keepdim=True)
                x_flat = xk_mean.view(-1)
                mu1 = x_flat.mean()
                mu2 = x_flat.std()
                mus = torch.stack([mu1, mu2])
                attn = self.sigmoid(torch.sum(mus * self.beta[k], dim=0)).view(1, 1, 1, 1)
                xk_attn = xk * attn
                out[b, idx, :, :] = xk_attn.squeeze(0)

        return out + x


class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.high_feature = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel1, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel1),
            nn.ReLU())
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=self.channel1)
        self.relu1 = nn.ReLU(inplace=True)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=self.channel1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_map = self.high_feature(y)
        up = F.interpolate(input_map, size=x.size()[2:], mode='bilinear', align_corners=True)
        mask = self.sa(up)
        mask = self.sigmoid(mask)
        f_feature = x * mask
        b_feature = x * (1 - mask)
        refine1 = (self.beta * b_feature) + x + up
        refine1 = self.gn1(refine1)
        refine1 = self.relu1(refine1)
        refine2 = refine1 - (self.alpha * f_feature)
        refine2 = self.gn2(refine2)
        refine2 = self.relu2(refine2)
        return refine2


@NECKS.register_module()
class HFCP(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False):
        super(HFCP, self).__init__()

        _, C3_size, C4_size, C5_size = in_channels
        feature_size = out_channels

        # 正向金字塔
        self.positioning_p6 = Positioning(512)
        self.P6_1 = nn.Sequential(nn.Conv2d(C5_size, 512, kernel_size=3, stride=2, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=512),
                                  nn.ReLU())
        self.cgca_p6 = CGCA(512)
        self.P6_2 = nn.Conv2d(512, feature_size, kernel_size=3, stride=1, padding=1)
        self.P7_1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(512, feature_size, kernel_size=3, stride=2, padding=1))
        self.P5_1 = nn.Sequential(nn.Conv2d(C5_size, 512, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=512),
                                  nn.ReLU())
        self.cgca_p5 = CGCA(512)
        self.focus3 = Focus(512, 512)
        self.P5_2 = nn.Conv2d(512, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Sequential(nn.Conv2d(C4_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=256),
                                  nn.ReLU())
        self.cgca_p4 = CGCA(256)
        self.focus2 = Focus(256, 512)
        self.P4_2 = nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Sequential(nn.Conv2d(C3_size, 128, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=128),
                                  nn.ReLU())
        self.cgca_p3 = CGCA(128)
        self.focus1 = Focus(128, 256)
        self.P3_2 = nn.Conv2d(128, feature_size, kernel_size=3, stride=1, padding=1)

        # 反向金字塔
        self.R6 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.R5 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.R4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.R3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.pam_p6 = PAM(512)
        self.pam_p5 = PAM(512)
        self.pam_p4 = PAM(256)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P6_x = self.P6_1(C5)
        P6_x = self.positioning_p6(P6_x)
        P6_x = self.cgca_p6(P6_x)
        P6_x = self.pam_p6(P6_x)
        P5_x = self.P5_1(C5)
        P5_x = self.cgca_p5(P5_x)
        P5_x = self.pam_p5(P5_x)
        P4_x = self.P4_1(C4)
        P4_x = self.cgca_p4(P4_x)
        P4_x = self.pam_p4(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = self.cgca_p3(P3_x)

        P7_x = self.P7_1(P6_x)
        P6_x = self.P6_2(P6_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_2(P3_x)

        R6_x = self.R6(F.interpolate(P7_x, size=P6_x.shape[2:], mode='bilinear', align_corners=True)) + P6_x
        R5_x = self.R5(F.interpolate(R6_x, size=P5_x.shape[2:], mode='bilinear', align_corners=True)) + P5_x
        R4_x = self.R4(F.interpolate(R5_x, size=P4_x.shape[2:], mode='bilinear', align_corners=True)) + P4_x
        R3_x = self.R3(F.interpolate(R4_x, size=P3_x.shape[2:], mode='bilinear', align_corners=True)) + P3_x

        outs = [R3_x, R4_x, R5_x, R6_x, P7_x]
        return tuple(outs)
