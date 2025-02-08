import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import nibabel as nib
import itertools as it
from tqdm import tqdm

from FineTrack.utils.utils import count_parameters

VOLUME_PATH = '/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/ismrm2015_2mm/fodfs/ismrm2015_fodf.nii.gz'
WM_MASK_PATH = '/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/ismrm2015_2mm/masks/ismrm2015_wm_mask.nii.gz'

dim_2d = False
conv_layer = nn.Conv2d if dim_2d else nn.Conv3d
bn_layer = nn.BatchNorm2d if dim_2d else nn.BatchNorm3d
conv_t_layer = nn.ConvTranspose2d if dim_2d else nn.ConvTranspose3d
get_flat_size = lambda dim_size: dim_size**2 if dim_2d else dim_size**3
get_flat_shape = lambda dim_size: (dim_size, dim_size) if dim_2d else (dim_size, dim_size, dim_size)

class NeighborhoodDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_coefs=28, neighborhood_size=3, method='crop', torch_convention=False):
        self.is_train = train
        self.volume_nib = nib.load(VOLUME_PATH)
        self.volume_data = self.volume_nib.get_fdata()
        self.affine = self.volume_nib.affine

        self.neighborhood_size = neighborhood_size
        self.rng = np.random.RandomState(42)
        self.coords = self._get_coordinates()
        self.method = method
        self.n_coefs = n_coefs
        self.torch_convention = torch_convention

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.method == 'crop':
            return self._crop_at_coordinate(self.coords[idx])
        elif self.method == 'interpolate':
            return self._interpolate_at_coordinate(self.coords[idx])
        else:
            raise ValueError('Invalid method: {}'.format(self.method))
    
    def _crop_at_coordinate(self, coord):
        x, y, z = coord
        rad = self.neighborhood_size

        # Keep everything within bounds
        min_x = np.clip(x-rad, 0, self.volume_data.shape[0])
        max_x = np.clip(x+rad, 0, self.volume_data.shape[0])
        min_y = np.clip(y-rad, 0, self.volume_data.shape[1])
        max_y = np.clip(y+rad, 0, self.volume_data.shape[1])
        min_z = np.clip(z-rad, 0, self.volume_data.shape[2])
        max_z = np.clip(z+rad, 0, self.volume_data.shape[2])

        crop = self.volume_data[min_x:max_x, min_y:max_y, min_z:max_z]

        # We need to pad the indices where we are out of bounds
        # For example, if x-rad is negative, we need to pad the left side of the crop
        # if x+rad is greater than the volume_data.shape[0], we need to pad the right side of the crop
        # Same thing for y and z
        def get_padding(x, rad, max_value):
            pad_min = 0
            pad_max = 0
            if x-rad < 0:
                pad_min = rad - x

            if x+rad > max_value:
                pad_max = x + rad - max_value
            
            return pad_min, pad_max

        pad_left, pad_right = get_padding(x, rad, self.volume_data.shape[0])
        pad_top, pad_bottom = get_padding(y, rad, self.volume_data.shape[1])
        pad_front, pad_back = get_padding(z, rad, self.volume_data.shape[2])

        placeholder = np.zeros((2*rad, 2*rad, 2*rad, self.n_coefs), dtype=np.float32)
        placeholder[pad_left:2*rad-pad_right, pad_top:2*rad-pad_bottom, pad_front:2*rad-pad_back] = crop

        if self.torch_convention:
            placeholder = np.transpose(placeholder, (3, 0, 1, 2))

        if dim_2d:
            return placeholder[:, 0, :, :] # 3D image
        else:
            return placeholder[:, :, :, :] # 2D image

    def _interpolate_at_coordinate(self, coord):
        pass

    def _get_coordinates(self):
        all_x = np.arange(0, self.volume_data.shape[0])[self.volume_data.shape[0]//2-20:self.volume_data.shape[0]//2+20]
        all_y = np.arange(0, self.volume_data.shape[1])[self.volume_data.shape[1]//2-20:self.volume_data.shape[1]//2+20]
        all_z = np.arange(0, self.volume_data.shape[2])[self.volume_data.shape[2]//2-20:self.volume_data.shape[2]//2+20]

        all_coords = list(it.product(all_x, all_y, all_z))
        self.rng.shuffle(all_coords)
        if self.is_train:
            coords = all_coords[:int(0.8*len(all_coords))]
        else:
            coords = all_coords[int(0.8*len(all_coords)):]

        # For testing purposes, we will only use one coordinate and repeat it multiple times
        # coords = [coords[0]] * 10000
        return coords

def setup_neighborhood_datasets(neighborhood_size=3, method='crop'):
    trainset = NeighborhoodDataset(train=True, neighborhood_size=neighborhood_size, method=method, torch_convention=True)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)

    testset = NeighborhoodDataset(train=True, neighborhood_size=neighborhood_size, method=method, torch_convention=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=10)

    return trainloader, testloader

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block = nn.Sequential(
            conv_layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            bn_layer(in_channels),
            nn.ReLU(),
            conv_layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            bn_layer(in_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        residue = x
        out = self.block(x)
        out += residue
        return out

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_space_size = 4096

        self.encoder = nn.Sequential(
            # 32x32x3
            ResidualBlock(in_channels=28), # 32x32x32
            ResidualBlock(in_channels=28), # 32x32x32
            ResidualBlock(in_channels=28), # 32x32x32
            ResidualBlock(in_channels=28), # 32x32x32

            self.downsampling_block(in_channels=28, out_channels=48), # 16x16x48
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),

            self.downsampling_block(in_channels=48, out_channels=96), # 8x8x96
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            # self.downsampling_block(in_channels=96, out_channels=192), # 4x4x192
            # self.downsampling_block(in_channels=192, out_channels=96), # 2x2x96

            nn.Flatten(),
            nn.Linear(get_flat_size(8)*96, self.latent_space_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space_size, get_flat_size(8)*96),
            nn.Unflatten(1, (96, *get_flat_shape(8))),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            # self.upsampling_block(in_channels=96, out_channels=192), # 4x4x192
            # self.upsampling_block(in_channels=192, out_channels=96), # 8x8x96
            self.upsampling_block(in_channels=96, out_channels=48), # 16x16x48
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),

            conv_t_layer(in_channels=48, out_channels=48, kernel_size=2, stride=2), # 32x32x48
            nn.ReLU(),
            conv_layer(in_channels=48, out_channels=28, kernel_size=3, stride=1, padding=1),
        )

        print('Encoder: {} params'.format(count_parameters(self.encoder)))
        print('Decoder: {} params'.format(count_parameters(self.decoder)))

    def downsampling_block(self, in_channels, out_channels):
        return nn.Sequential(
            conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def upsampling_block(self, in_channels, out_channels):
        return nn.Sequential(
            conv_t_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out


def main():
    trainloader, testloader = setup_neighborhood_datasets(neighborhood_size=16, method='crop')
    model = Model()
    model.train()
    model = model.to(device='cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    last_best_loss = np.inf
    nb_epochs = 100
    for epoch in range(nb_epochs):
        for i, inputs in enumerate(tqdm(trainloader)):
            inputs = inputs.to(device='cuda')
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                loss_item = loss.item()
                print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss_item))

                if loss_item < last_best_loss:
                    last_best_loss = loss_item
                    torch.save(model.state_dict(), 'best_model.pth')



    # Vizualize some reconstructions from the testloader
    import matplotlib.pyplot as plt

    n = 5 # number of reconstructions to vizualize
    fig, axes = plt.subplots(2, n, figsize=(20, 5))
    for i, inputs in enumerate(testloader):
        inputs = inputs.to(device='cuda')
        outputs = model(inputs)
        for j in range(n):
            if dim_2d:
                target = inputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
            else:
                target = inputs[j, 0][None, 0].permute(1, 2, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None, 0].permute(1, 2, 0).cpu().detach().numpy()
            print("loss ({}): {}".format(j, criterion(outputs[j], inputs[j])))
            
            axes[0, j].imshow(target)
            axes[1, j].imshow(recons)

        break

    if dim_2d:
        plt.savefig('reconstructions_2d.png')
    else:
        plt.savefig('reconstructions_3d.png')
    # plt.show()




if __name__ == '__main__':
    main()