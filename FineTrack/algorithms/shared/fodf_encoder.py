import torch
import torch.nn as nn

from FineTrack.algorithms.shared.utils import ResidualBlock
from FineTrack.algorithms.shared.batch_renorm import BatchRenorm1d, BatchRenorm3d

class FodfEncoder(nn.Module):
    def __init__(self, n_coeffs=28, renorm=False, activation=nn.GELU):
        super().__init__()

        norm_layer = nn.BatchNorm3d if not renorm else BatchRenorm3d

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=n_coeffs, out_channels=n_coeffs, kernel_size=1, stride=1, padding=0),
            activation(),
            nn.BatchNorm3d(n_coeffs),  # 28x19x19x19
            nn.Conv3d(in_channels=n_coeffs, out_channels=64, kernel_size=3, stride=1, padding=1),  # 90x108x90x64
            activation(),
            norm_layer(64),

            ResidualBlock(64, norm_layer=norm_layer),  # 64x19x19x19
            ResidualBlock(64, norm_layer=norm_layer),  # 64x19x19x19
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x10x10x10
            
            ResidualBlock(128, norm_layer=norm_layer),  # 128x10x10x10
            ResidualBlock(128, norm_layer=norm_layer),  # 128x10x10x10
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # 256x5x5x5

            ResidualBlock(256, norm_layer=norm_layer),  # 256x5x5x5
            ResidualBlock(256, norm_layer=norm_layer),  # 256x5x5x5
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),  # 512x3x3x3
        )

        # self.encoder4 = nn.Sequential(
        #     ResidualBlock(512, norm_layer=norm_layer),  # 11x13x11x512
        #     ResidualBlock(512, norm_layer=norm_layer),  # 11x13x11x512
        #     nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1),  # 5x6x5x1024
        # )

    def forward(self, x):
        return self.encoder(x)