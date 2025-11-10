import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class LocalAdaptiveContrastEnhancement(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size,
                      padding=kernel_size // 2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x) + x