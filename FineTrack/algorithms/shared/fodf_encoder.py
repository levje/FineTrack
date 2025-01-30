import torch
import torch.nn as nn

from FineTrack.algorithms.shared.utils import ResidualBlock
from FineTrack.algorithms.shared.batch_renorm import BatchRenorm1d, BatchRenorm3d
from FineTrack.utils.utils import count_parameters

class DummyFodfEncoder(nn.Module):
    """
    This should not be a bottleneck
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        print(f"{self.__class__.__name__} __init__ with {count_parameters(self)} parameters")

    def forward(self, x):
        return x[:, :64, :3, :3, :3]
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        return
    
    def state_dict(self):
        return {}

class FodfEncoder(nn.Module):
    def __init__(self, n_coeffs=28, renorm=False, activation=nn.GELU):
        super().__init__()

        norm_layer = nn.BatchNorm3d if not renorm else BatchRenorm3d

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=n_coeffs, out_channels=n_coeffs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(n_coeffs),  # 28x19x19x19
            activation(),
            nn.Conv3d(in_channels=n_coeffs, out_channels=32, kernel_size=3, stride=1, padding=1),  # 90x108x90x64
            norm_layer(32),
            activation(),

            # ResidualBlock(64, norm_layer=norm_layer),  # 64x19x19x19
            ResidualBlock(32, norm_layer=norm_layer),  # 64x19x19x19
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # 128x10x10x10

            # ResidualBlock(128, norm_layer=norm_layer),  # 128x10x10x10
            ResidualBlock(64, norm_layer=norm_layer),  # 128x10x10x10
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # 256x5x5x5

            # ResidualBlock(256, norm_layer=norm_layer),  # 256x5x5x5
            ResidualBlock(128, norm_layer=norm_layer),  # 256x5x5x5
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # 512x3x3x3

            # Reduce the number of channels, otherwise the latent space is too large to fit in memory.
            nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0),  # 256x3x3x3
            norm_layer(128),
            activation(),

            nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0),  # 128x3x3x3
            norm_layer(64),
            activation(),
        )

        self.flattener = nn.Flatten()

        # self.encoder4 = nn.Sequential(
        #     ResidualBlock(512, norm_layer=norm_layer),  # 11x13x11x512
        #     ResidualBlock(512, norm_layer=norm_layer),  # 11x13x11x512
        #     nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1),  # 5x6x5x1024
        # )

        print(f"{self.__class__.__name__} __init__ with {count_parameters(self)} parameters")

    @property
    def flat_output_size(self):
        return 64 * 3 * 3 * 3
    
    def forward(self, x, flatten=False, swap_channels=False):
        if swap_channels:
            x = x.permute(0, 4, 1, 2, 3)
            
        x = self.encoder(x)
        if flatten:
            x = self.flattener(x)
            assert x.shape[1] == self.flat_output_size, \
                "The flattened output is not the expected size of " \
                f"{self.flat_output_size}. Make sure that the input size is " \
                "in the correct order (N, C, D, H, W) as specified in the " \
                "PyTorch documentation about Conv3d layers."

        return x
    