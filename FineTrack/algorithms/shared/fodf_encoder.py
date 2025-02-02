import torch
import torch.nn as nn

from FineTrack.algorithms.shared.utils import ResidualBlock, ResNextBlock
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
    
class ExpFodfEncoder(nn.Module):
    """
    This should not be a bottleneck
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        sc = 64 # Start channels
        self.layers = nn.Sequential(
            # 28x19x19x19
            nn.Conv3d(in_channels=28, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
            nn.ReLU(),
            nn.Conv3d(in_channels=28, out_channels=sc, kernel_size=3, stride=1, padding=1),  # 64x19x19x19
            nn.ReLU(),

            nn.Conv3d(in_channels=sc, out_channels=sc*2, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
            nn.ReLU(),
            self.make_layer(in_channels=sc*2, cardinality=8, num_blocks=3, stride=1),  # 128x19x19x19

            nn.Conv3d(in_channels=sc*2, out_channels=sc*4, kernel_size=3, stride=1, padding=1),  # 256x19x19x19
            nn.ReLU(),
            self.make_layer(in_channels=sc*4, cardinality=16, num_blocks=3, stride=1),  # 256x19x19x19

            # MAXPOOL 19x19x19 -> 9x9x9
            nn.MaxPool3d(kernel_size=2, stride=2),  # 256x9x9x9
            nn.Conv3d(in_channels=sc*4, out_channels=sc*8, kernel_size=3, stride=1, padding=1),  # 512x9x9x9
            nn.ReLU(),
            self.make_layer(in_channels=sc*8, cardinality=16, num_blocks=2, stride=1),  # 512x9x9x9

            # MAXPOOL 9x9x9 -> 4x4x4
            nn.MaxPool3d(kernel_size=2, stride=2),  # 28x4x4x4
            nn.Conv3d(in_channels=sc*8, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
            nn.ReLU(),
            self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 1024x4x4x4

            nn.Conv3d(in_channels=sc*16, out_channels=sc*32, kernel_size=3, stride=1, padding=1),  # 2048x4x4x4


            # MAXPOOL 4x4x4 -> 2x2x2
            # nn.MaxPool3d(kernel_size=2, stride=2),  # 28x2x2x2
            # nn.Conv3d(in_channels=sc*8, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 2048x2x2x2
            # nn.ReLU(),
            # self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 2048x2x2x2

            # MAXPOOL 2x2x2 -> 1x1x1
            # nn.MaxPool3d(kernel_size=2, stride=2),  # 4096x1x1x1
            # nn.Conv3d(in_channels=sc*16, out_channels=sc*32, kernel_size=3, stride=1, padding=1),  # 4096x1x1x1
            # nn.ReLU(),
            # self.make_layer(in_channels=sc*32, cardinality=16, num_blocks=1, stride=1),  # 4096x1x1x1
        )

        # self.decoding_layers = nn.Sequential(
        #     self.make_layer(in_channels=sc*32, cardinality=16, num_blocks=1, stride=1),  # 4096x1x1x1

        #     # UPSAMPLE 1x1x1 -> 2x2x2
        #     nn.ConvTranspose3d(in_channels=sc*32, out_channels=sc*16, kernel_size=2, stride=2, padding=0),  # 4096x2x2x2
        #     # nn.Conv3d(in_channels=sc*32, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 2048x2x2x2
        #     # nn.ReLU(),
        #     self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 2048x2x2x2

        #     # UPSAMPLE 2x2x2 -> 4x4x4
        #     nn.ConvTranspose3d(in_channels=sc*16, out_channels=sc*8, kernel_size=2, stride=2, padding=0),  # 28x4x4x4
        #     # nn.Conv3d(in_channels=sc*16, out_channels=sc*8, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
        #     # nn.ReLU(),
        #     self.make_layer(in_channels=sc*8, cardinality=16, num_blocks=1, stride=1),  # 1024x4x4x4

        #     # UPSAMPLE 4x4x4 -> 9x9x9
        #     nn.ConvTranspose3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=2, padding=0),  # 28x9x9x9
        #     # nn.Conv3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=1, padding=1),  # 512x9x9x9
        #     # nn.ReLU(),
        #     self.make_layer(in_channels=sc*4, cardinality=16, num_blocks=2, stride=1),  # 512x9x9x9

        #     # UPSAMPLE 9x9x9 -> 19x19x19
        #     nn.ConvTranspose3d(in_channels=sc*4, out_channels=sc*2, kernel_size=3, stride=2, padding=0),  # 28x19x19x19
        #     self.make_layer(in_channels=sc*2, cardinality=8, num_blocks=3, stride=1),  # 128x19x19x19

        #     nn.Conv3d(in_channels=sc*2, out_channels=sc, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=sc, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        # )

        print(f"{self.__class__.__name__} __init__ with {count_parameters(self)} parameters")

    def forward(self, x):
        out = self.layers(x)

        # print("encoder.out.shape: ", out.shape)
        # out = self.decoding_layers(out)
        # print("decoder.out.shape: ", out.shape)

        # assert False

        return out
    
    def make_layer(self, in_channels, cardinality, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNextBlock(in_channels, in_channels//2, cardinality, stride))
        return nn.Sequential(*layers)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return
    
    def state_dict(self):
        return {}
    
class ExpFodfDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sc = 64 # Start channels
        self.decoding_layers = nn.Sequential(

            # UPSAMPLE 1x1x1 -> 2x2x2
            # nn.ConvTranspose3d(in_channels=sc*32, out_channels=sc*16, kernel_size=2, stride=2, padding=0),  # 4096x2x2x2
            # nn.Conv3d(in_channels=sc*32, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 2048x2x2x2
            # nn.ReLU(),
            # self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 2048x2x2x2

            self.make_layer(in_channels=sc*32, cardinality=16, num_blocks=1, stride=1),  # 2048x4x4x4
            nn.Conv3d(in_channels=sc*32, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
            nn.ReLU(),

            # UPSAMPLE 2x2x2 -> 4x4x4
            # nn.ConvTranspose3d(in_channels=sc*32, out_channels=sc*16, kernel_size=2, stride=2, padding=0),  # 28x4x4x4
            # nn.Conv3d(in_channels=sc*16, out_channels=sc*8, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
            # nn.ReLU(),
            self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=2, stride=1),  # 1024x4x4x4

            # UPSAMPLE 4x4x4 -> 9x9x9
            nn.ConvTranspose3d(in_channels=sc*16, out_channels=sc*8, kernel_size=3, stride=2, padding=0),  # 28x9x9x9
            # nn.Conv3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=1, padding=1),  # 512x9x9x9
            # nn.ReLU(),
            self.make_layer(in_channels=sc*8, cardinality=16, num_blocks=2, stride=1),  # 512x9x9x9

            # UPSAMPLE 9x9x9 -> 19x19x19
            nn.ConvTranspose3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=2, padding=0),  # 28x19x19x19
            self.make_layer(in_channels=sc*4, cardinality=8, num_blocks=3, stride=1),  # 128x19x19x19

            nn.Conv3d(in_channels=sc*4, out_channels=sc*2, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
            nn.ReLU(),
            nn.Conv3d(in_channels=sc*2, out_channels=sc, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
            nn.ReLU(),
            nn.Conv3d(in_channels=sc, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        )
    
    def forward(self, x):
        out = self.decoding_layers(x)
        return out
    
    def make_layer(self, in_channels, cardinality, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNextBlock(in_channels, in_channels//2, cardinality, stride))
        return nn.Sequential(*layers)

class FodfEncoder(nn.Module):
    def __init__(self, n_coeffs=28, renorm=False, activation=nn.ReLU):
        super().__init__()

        norm_layer = nn.BatchNorm3d if not renorm else BatchRenorm3d

        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels=n_coeffs, out_channels=n_coeffs, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm3d(n_coeffs),  # 28x19x19x19
        #     activation(),
        #     nn.Conv3d(in_channels=n_coeffs, out_channels=n_coeffs, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm3d(n_coeffs),  # 28x19x19x19
        #     activation(),
        #     nn.Conv3d(in_channels=n_coeffs, out_channels=32, kernel_size=3, stride=1, padding=1),  # 90x108x90x64
        #     norm_layer(32),
        #     activation(),

        #     ResidualBlock(32, norm_layer=norm_layer),  # 64x19x19x19
        #     ResidualBlock(32, norm_layer=norm_layer),  # 64x19x19x19
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # 64x10x10x10
        #     nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # 128x10x10x10

        #     ResidualBlock(64, norm_layer=norm_layer),  # 128x10x10x10
        #     ResidualBlock(64, norm_layer=norm_layer),  # 128x10x10x10
        #     nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # 256x5x5x5

        #     ResidualBlock(128, norm_layer=norm_layer),  # 256x5x5x5
        #     ResidualBlock(128, norm_layer=norm_layer),  # 256x5x5x5
        #     ResidualBlock(128, norm_layer=norm_layer),  # 256x5x5x5
        #     nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # 512x3x3x3

        #     ResidualBlock(256, norm_layer=norm_layer),  # 512x3x3x3
        #     ResidualBlock(256, norm_layer=norm_layer),  # 512x3x3x3

        #     # Reduce the number of channels, otherwise the latent space is too large to fit in memory.
        #     # nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0),  # 256x3x3x3
        #     # norm_layer(128),
        #     # activation(),

        #     # nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0),  # 128x3x3x3
        #     # norm_layer(64),
        #     # activation(),
        # )

        self.activ = activation()

        # Layers
        self.conv1x1_1 = nn.Conv3d(n_coeffs, n_coeffs, kernel_size=1) # 28x19x19x19
        self.conv1x1_2 = nn.Conv3d(n_coeffs, 64, kernel_size=1) # 64x19x19x19
        
        self.conv3x3_3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1) # 128x19x19x19
        self.bn_3 = nn.BatchNorm3d(128) # 128x19x19x19

        self.conv_4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1) # 128x19x19x19
        self.bn_4 = nn.BatchNorm3d(256) # 128x19x19x19
        self.layer_1 = self.make_layer(in_channels=256, cardinality=8, num_blocks=3, stride=1) # 128x19x19x19

        self.conv_5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1) # 256x10x10x10
        self.bn_5 = nn.BatchNorm3d(512) # 256x10x10x10
        self.layer_2 = self.make_layer(in_channels=512, cardinality=16, num_blocks=3, stride=1) # 256x10x10x10

        self.conv_6 = nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1) # 512x5x5x5
        self.bn_6 = nn.BatchNorm3d(1024) # 512x5x5x5
        self.layer_3 = self.make_layer(in_channels=1024, cardinality=32, num_blocks=3, stride=1) # 512x5x5x5

        self.conv_7 = nn.Conv3d(1024, 2048, kernel_size=3, stride=2, padding=1) # 1024x3x3x3
        self.bn_7 = nn.BatchNorm3d(2048) # 1024x3x3x3
        self.layer_4 = self.make_layer(in_channels=2048, cardinality=64, num_blocks=3, stride=1) # 1024x3x3x3

        self.flattener = nn.Flatten()

        print(f"{self.__class__.__name__} __init__ with {count_parameters(self)} parameters")

    def make_layer(self, in_channels, cardinality, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNextBlock(in_channels, in_channels//2, cardinality, stride))
        return nn.Sequential(*layers)

    @property
    def flat_output_size(self):
        return 64 * 3 * 3 * 3
    
    def forward(self, x, flatten=False, swap_channels=False):
        if swap_channels:
            x = x.permute(0, 4, 1, 2, 3)
        
        x = self.conv1x1_1(x)
        x = self.activ(x)
        x = self.conv1x1_2(x)
        x = self.activ(x)

        x = self.conv3x3_3(x)
        x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.activ(x)
        x = self.layer_1(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.activ(x)
        x = self.layer_2(x)

        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.activ(x)
        x = self.layer_3(x)

        x = self.conv_7(x)
        x = self.bn_7(x)
        x = self.activ(x)
        x = self.layer_4(x)

        if flatten:
            x = self.flattener(x)
            assert x.shape[1] == self.flat_output_size, \
                "The flattened output is not the expected size of " \
                f"{self.flat_output_size}. Make sure that the input size is " \
                "in the correct order (N, C, D, H, W) as specified in the " \
                "PyTorch documentation about Conv3d layers."

        return x
    
