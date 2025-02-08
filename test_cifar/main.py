import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from FineTrack.utils.utils import count_parameters

def download_cifar_10_dataset():

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=10)

    # Download and load the test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=10)

    return trainloader, testloader

class ResidualBlock(nn.Module):
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

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_space_size = 256

        self.encoder = nn.Sequential(
            # 32x32x3
            ResidualBlock(in_channels=3), # 32x32x32
            ResidualBlock(in_channels=3), # 32x32x32
            ResidualBlock(in_channels=3), # 32x32x32
            ResidualBlock(in_channels=3), # 32x32x32

            self.downsampling_block(in_channels=3, out_channels=48), # 16x16x48
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
            nn.Linear(8*8*96, self.latent_space_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space_size, 8*8*96),
            nn.Unflatten(1, (96, 8, 8)),
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

            nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=2, stride=2), # 32x32x48
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, stride=1, padding=1),
        )

        print('Encoder: {} params'.format(count_parameters(self.encoder)))
        print('Decoder: {} params'.format(count_parameters(self.decoder)))

    def downsampling_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def upsampling_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out

def display_image(img, ax):
    zero_to_one_img = torch.clip((img + 1) / 2, 0.0, 1.0)
    img_ints = (zero_to_one_img * 255).int()
    ax.imshow(img_ints.permute(1, 2, 0).cpu().detach().numpy())


def main():
    trainloader, testloader = download_cifar_10_dataset()
    model = Model()
    model.train()
    model = model.to(device='cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    nb_epochs = 25
    for epoch in range(nb_epochs):
        for i, (inputs, _) in enumerate(trainloader):
            inputs = inputs.to(device='cuda')
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss.item()))


    # Vizualize some reconstructions from the testloader
    import matplotlib.pyplot as plt

    n = 5 # number of reconstructions to vizualize
    fig, axes = plt.subplots(2, n, figsize=(20, 5))
    for i, (inputs, _) in enumerate(testloader):
        inputs = inputs.to(device='cuda')
        outputs = model(inputs)
        for j in range(n):
            display_image(inputs[j], ax=axes[0, j])
            display_image(outputs[j], ax=axes[1, j])
            # axes[0, j].imshow(inputs[j].permute(1, 2, 0).cpu().detach().numpy())
            # axes[1, j].imshow(outputs[j].permute(1, 2, 0).cpu().detach().numpy())
        break

    plt.savefig('reconstructions.png')
    # plt.show()




if __name__ == '__main__':
    main()