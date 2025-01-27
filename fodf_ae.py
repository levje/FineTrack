import comet_ml
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
import nibabel as nib
import numpy as np
from tqdm import tqdm
from FineTrack.algorithms.shared.utils import ResidualBlock
from FineTrack.algorithms.shared.batch_renorm import BatchRenorm1d, BatchRenorm3d
from FineTrack.utils.torch_utils import get_device, get_device_str
from FineTrack.utils.logging import get_logger, setLevel
from FineTrack.algorithms.shared.fodf_encoder import FodfEncoder
from FineTrack.utils.neighborhood_interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import \
    unflatten_neighborhood, prepare_neighborhood_vectors

LOGGER = get_logger(__name__)
device = get_device_str()

class FodfAe(nn.Module):
    def __init__(self, input_shape, n_coeffs=28, renorm=False):
        super().__init__()

        self.activation = nn.GELU
        self.norm_layer = nn.BatchNorm3d if not renorm else BatchRenorm3d

        # Let's say the input image is 128x128x128x28
        self.input_shape = input_shape

        self.encoder = FodfEncoder(n_coeffs, renorm)

        self.decoder = nn.Sequential(
            # ResidualBlock(1024, norm_layer=self.norm_layer),  # 8x8x8x1024

            # nn.Upsample(scale_factor=2),  # 16x16x16x1024
            # nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1),  # 16x16x16x512

            ResidualBlock(64, norm_layer=self.norm_layer),  # 64x3x3x3

            # Add more channels before upsampling
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),  # 128x3x3x3
            self.activation(),
            self.norm_layer(128),
            ResidualBlock(128, norm_layer=self.norm_layer),  # 128x3x3x3
            
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),  # 256x3x3x3
            self.activation(),
            self.norm_layer(256),

            # Start upsampling
            nn.Upsample(scale_factor=2),  # 512x6x6x6
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),  # 256x6x6x6

            ResidualBlock(128, norm_layer=self.norm_layer),  # 256x6x6x6
            nn.Upsample(scale_factor=2),  # 256x12x12x12
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),  # 128x12x12x12

            ResidualBlock(64, norm_layer=self.norm_layer),  # 128x12x12x12
            nn.Upsample(scale_factor=2),  # 128x24x24x24

            ResidualBlock(64, norm_layer=self.norm_layer),  # 64x24x24x24
            ResidualBlock(64, norm_layer=self.norm_layer),  # 64x24x24x24
            
            nn.Conv3d(64, 64, kernel_size=5, stride=1, padding=0), # 64x20x20x20
            self.activation(),
            self.norm_layer(64),
            nn.Conv3d(64, n_coeffs, kernel_size=1, stride=1, padding=0),  # 28x20x20x20
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decode(latent)
        return output

    def decode(self, latent):
        output = self.decoder(latent)
        output = output[..., :-1, :-1, :-1]
        return output
    
    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])

class NeighborhoodManager(object):
    def __init__(self, data_volume, radius, add_neighborhood_vox, neighborhood_type='grid', resolution=1, flatten=False):
        self.data_volume = data_volume
        self.radius = radius
        self.add_neighborhood_vox = add_neighborhood_vox
        self.neighborhood_type = neighborhood_type
        self.flatten = flatten
        self.resolution = resolution

        self.neighborhood_directions = prepare_neighborhood_vectors(self.neighborhood_type,
            self.radius, self.resolution, )
    
    @property
    def common_shape(self):
        return (self.radius * 2 + 1, ) * 3

    def get(self, coords):
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=False):
            signal, _ = interpolate_volume_in_neighborhood(
                self.data_volume,
                coords,
                self.neighborhood_directions,
                clear_cache=False)

        if not self.flatten:
            # Unflatten the signal into (N, W, H, D, C) shape, where this is the
            # convention for PyTorch's Conv3d module.
            unflattened = unflatten_neighborhood(
                signal, self.neighborhood_directions, self.neighborhood_type,
                self.radius, self.add_neighborhood_vox)

            # Permute axes to fit PyTorch's convention of (N, C, D, H, W)
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
            signal = unflattened.permute(0, 4, 1, 2, 3)

        return signal

class FodfDataset(torch.utils.data.Dataset):
    def __init__(self, neighborhood_manager, coords):
        self.neigh_manager = neighborhood_manager
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coords = self.coords[idx]
        if isinstance(idx, int):
            coords = coords.unsqueeze(0)
        signal = self.neigh_manager.get(coords)
        signal = signal.squeeze(0)
        return signal
        # return coords

class FodfAeTrainer(object):
    def __init__(self, input_shape, n_coeffs=28, nb_epochs=100, neighborhood_radius=9, batch_size=1, device="cpu"):
        self.model = FodfAe(input_shape=input_shape, n_coeffs=n_coeffs, renorm=False)
        self.device = device
        self.batch_size = batch_size
        self.nb_epochs = 100

        fodf_path = "data/datasets/ismrm2015_2mm/fodfs/ismrm2015_fodf.nii.gz"
        fodf_img = nib.load(fodf_path)
        fodf_data = fodf_img.get_fdata()
        data_volume = torch.from_numpy(fodf_data)
        
        self.neigh_manager = NeighborhoodManager(
            data_volume=data_volume,
            radius=neighborhood_radius,
            add_neighborhood_vox=1,
            neighborhood_type='grid',
            resolution=1,
            flatten=False
        )

        train_ratio = 0.8
        valid_ratio = 0.1
        test_ratio = 1 - train_ratio - valid_ratio

        self.reconstruction_loss = nn.MSELoss()

        all_coords = self._get_all_coords(input_shape, shuffled=True)
        self.train_coords = all_coords[:int(train_ratio * len(all_coords))]
        self.valid_coords = all_coords[int(train_ratio * len(all_coords)):int((train_ratio + valid_ratio) * len(all_coords))]
        self.test_coords = all_coords[int((train_ratio + valid_ratio) * len(all_coords)):]

        self.train_dataset = FodfDataset(self.neigh_manager, self.train_coords)
        self.valid_dataset = FodfDataset(self.neigh_manager, self.valid_coords)
        self.test_dataset = FodfDataset(self.neigh_manager, self.test_coords)

        train_sampler = BatchSampler(SequentialSampler(
            self.train_dataset), self.batch_size,
            drop_last=False)
        valid_sampler = BatchSampler(SequentialSampler(
            self.valid_dataset), self.batch_size,
            drop_last=False)
        test_sampler = BatchSampler(SequentialSampler(
            self.test_dataset), self.batch_size,
            drop_last=False)

        self.train_loader = DataLoader(self.train_dataset, sampler=train_sampler, num_workers=8)
        self.valid_loader = DataLoader(self.valid_dataset, sampler=valid_sampler)
        self.test_loader = DataLoader(self.test_dataset, sampler=test_sampler)

        self.project_name = "fodf_ae"
        self.workspace="mrzarfir"
        self.comet_enabled = True
        self.experiment = None

        self.lr = 1e-6
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader))
        self._create_comet_experiment()

    def to(self, device):
        self.model.to(device)
        return self
    
    def _get_all_coords(self, img_shape, shuffled=False):
        grid_coords = torch.meshgrid(
            torch.arange(img_shape[0]),
            torch.arange(img_shape[1]),
            torch.arange(img_shape[2])
        )
        coords = torch.stack(grid_coords, dim=-1)
        coords = coords.reshape(-1, 3).float()

        if shuffled:
            coords = coords[torch.randperm(coords.size(0))]

        return coords
    
    def _create_comet_experiment(self):
        if self.experiment is None:
            self.experiment = comet_ml.Experiment(project_name=self.project_name,
                                        workspace=self.workspace, parse_args=False,
                                        auto_metric_logging=False,
                                        disabled=not self.comet_enabled)

    def train(self):
        self.train_losses = []
        self.valid_losses = []

        self.model = self.model.to(self.device)
        best_loss = np.inf

        for epoch in tqdm(range(self.nb_epochs), desc="Epochs"):
            self.model.train()
            with tqdm(range(len(self.train_loader)), desc="Training batches", leave=False) as train_batch_pbar:
                for i, batch in enumerate(self.train_loader):
                    # coords = coords.squeeze(0)
                    # batch = self.neigh_manager.get(coords)
                        
                    if len(batch.shape) > 5:
                        batch = batch.squeeze(0)
                    batch = batch.to(self.device)

                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=False):
                        output = self.model(batch)
                        loss = self.reconstruction_loss(output, batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_item = loss.item()
                    self.train_losses.append(loss_item)

                    if loss_item < best_loss:
                        best_loss = loss_item
                        torch.save(self.model.state_dict(), "fodf_ae/best_model.pth")
                        torch.save(self.model.encoder.state_dict(), "fodf_ae/best_encoder.pth")


                        if loss_item < best_loss:
                            best_loss = loss_item
                            torch.save(self.model.state_dict(), "fodf_ae/best_model.pth")

                        # Send to Comet.ml
                        self.experiment.log_metric("train_loss", loss_item, step=i, epoch=epoch)
                        self.experiment.log_metric("lr", self.optimizer.param_groups[0]['lr'], step=i, epoch=epoch)

                        train_batch_pbar.set_postfix({"loss": loss_item, "lr": self.optimizer.param_groups[0]['lr']})
                        train_batch_pbar.update(1)

            # avg_loss = np.mean(self.train_losses)
            self.train_losses = []


            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.valid_loader, desc="Validation", leave=False):
                    if len(batch.shape) > 5:
                        batch = batch.squeeze(0)
                    batch = batch.to(self.device)

                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=False):
                        output = self.model(batch)
                        loss = self.reconstruction_loss(output, batch)
                    self.valid_losses.append(loss.item())

            avg_loss = np.mean(self.valid_losses)
            self.valid_losses = []

            # Send to Comet.ml
            self.experiment.log_metric("valid_loss", avg_loss, epoch=epoch)

        return self.valid_losses

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.reconstruction_loss(output, batch)
                self.test_losses.append(loss.item())

        avg_loss = np.mean(self.test_losses)
        self.test_losses = []

        # Send to Comet.ml
        self.experiment.log_metric("test_loss", avg_loss)

        return avg_loss

def main():
    print("loading fodf image")
    fodf_path = "data/datasets/ismrm2015_2mm/fodfs/ismrm2015_fodf.nii.gz"
    fodf_img = nib.load(fodf_path)
    fodf_data = fodf_img.get_fdata()
    fodf_shape = fodf_data.shape
    print("fodf image has the following shape: ", fodf_shape)
    nb_coefs = fodf_shape[-1]
    img_shape = fodf_shape[:-1]
    print("nb_coefs: ", nb_coefs, " img_shape: ", img_shape)


    neighborhood_radius = 9 # 19x19x19 neighborhood
    
    trainer = FodfAeTrainer(
        input_shape=img_shape,
        n_coeffs=nb_coefs,
        nb_epochs=100,
        neighborhood_radius=neighborhood_radius,
        batch_size=64,
        device=device)
    
    trainer.train()
    trainer.test()

if '__main__' == __name__:
    main()