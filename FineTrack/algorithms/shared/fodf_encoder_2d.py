import torch
import torch.nn as nn
import numpy as np

class ResidualBlockV2(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        residue = x
        out = self.block(x)
        out += residue
        return out
    
def downsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def upsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

class LinLatentEncoderV22D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = (1, 32, 32)
        self.sc = 64
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3, stride=1, padding=1), # 28x32x32x32
            nn.ReLU(),

            ResidualBlockV2(in_channels=28), # 28x32x32x32
            ResidualBlockV2(in_channels=28), # 28x32x32x32
            ResidualBlockV2(in_channels=28), # 28x32x32x32
            ResidualBlockV2(in_channels=28), # 28x32x32x32

            downsampling_block(in_channels=28, out_channels=32), # 32x16x16x16

            ResidualBlockV2(in_channels=32), # 32x16x16x16
            ResidualBlockV2(in_channels=32), # 32x16x16x16
            ResidualBlockV2(in_channels=32), # 32x16x16x16
            ResidualBlockV2(in_channels=32), # 32x16x16x16

            downsampling_block(in_channels=32, out_channels=48), # 48x8x8x8

            ResidualBlockV2(in_channels=48), # 48x8x8x8
            ResidualBlockV2(in_channels=48), # 48x8x8x8
            ResidualBlockV2(in_channels=48), # 48x8x8x8
            ResidualBlockV2(in_channels=48), # 48x8x8x8

            downsampling_block(in_channels=48, out_channels=96), # 128x4x4x4

            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4
        )

        self.flat_fts = self.get_flat_fts(self.convs)
        print("Flat fts: ", self.flat_fts)
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_fts, 512)
        )

    def get_flat_fts(self, fts):
        f = fts(torch.ones(1, *self.input_size))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.convs(x)
        x = self.lin(x)
        # print("Latent space shape: ", x.shape)
        return x

class LinLatentDecoderV22D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_in_dim = 512
        self.fc_out_dim = 96*4*4
        self.sc = 64

        self.linear = nn.Sequential(
            nn.Linear(self.fc_in_dim, self.fc_out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_out_dim),
        )

        self.convs = nn.Sequential(
            nn.Unflatten(1, (96, 4, 4)),
            
            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4

            upsampling_block(in_channels=96, out_channels=64), # 64x8x8x8

            ResidualBlockV2(in_channels=64), # 64x8x8x8
            ResidualBlockV2(in_channels=64), # 64x8x8x8

            upsampling_block(in_channels=64, out_channels=48), # 48x16x16x16

            ResidualBlockV2(in_channels=48), # 48x16x16x16
            ResidualBlockV2(in_channels=48), # 48x16x16x16
            ResidualBlockV2(in_channels=48), # 48x16x16x16
            ResidualBlockV2(in_channels=48), # 48x16x16x16

            upsampling_block(in_channels=48, out_channels=32), # 32x32x32x32

            ResidualBlockV2(in_channels=32), # 32x32x32x32
            ResidualBlockV2(in_channels=32), # 32x32x32x32
            ResidualBlockV2(in_channels=32), # 32x32x32x32
            ResidualBlockV2(in_channels=32), # 32x32x32x32

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1), # 28x32x32x32
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.convs(x)
        # print("Decoded shape: ", x.shape)
        return x