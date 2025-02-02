import comet_ml
import torch
import os
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
import nibabel as nib
import numpy as np
from tqdm import tqdm
from FineTrack.algorithms.shared.utils import ResidualBlock, DWSConv3d
from FineTrack.algorithms.shared.batch_renorm import BatchRenorm1d, BatchRenorm3d
from FineTrack.utils.torch_utils import get_device, get_device_str
from FineTrack.utils.logging import get_logger, setLevel
from FineTrack.algorithms.shared.fodf_encoder import FodfEncoder, ExpFodfEncoder, ExpFodfDecoder
from FineTrack.utils.neighborhood_interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import \
    unflatten_neighborhood, prepare_neighborhood_vectors
from FineTrack.utils.interpolation import neighborhood_interpolation, calc_neighborhood_grid
from FineTrack.utils.utils import TTLProfiler, count_parameters
from FineTrack.environments.neighborhood_manager import NeighborhoodManager

LOGGER = get_logger(__name__)
device = get_device_str()
USE_COMET = True

class FodfAe(nn.Module):
    def __init__(self, input_shape, n_coeffs=28, renorm=False):
        super().__init__()

        self.activation = nn.GELU
        self.norm_layer = nn.BatchNorm3d if not renorm else BatchRenorm3d

        # Let's say the input image is 128x128x128x28
        self.input_shape = input_shape

        self.encoder = ExpFodfEncoder()
        self.decoder = ExpFodfDecoder()

        # self.encoder = nn.Sequential(
        #     nn.Conv3d(n_coeffs, n_coeffs, kernel_size=1, stride=1, padding=0),  # 64x19x19x19
        #     # nn.ReLU(),

        #     # nn.Conv3d(n_coeffs, 128, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
        # )
        # self.decoder = nn.Sequential(
        #     # nn.ConvTranspose3d(128, n_coeffs, kernel_size=3, stride=1, padding=0),  # 64x24x24x24
        #     # nn.Conv3d(128, n_coeffs, kernel_size=3, stride=1, padding=1),  # 128x19x19x19


        #     nn.Conv3d(n_coeffs, n_coeffs, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        # )
        print("Encoder: {} params".format(count_parameters(self.encoder)))
        print("Decoder: {} params".format(count_parameters(self.decoder)))

        # self.encoder = FodfEncoder(n_coeffs, renorm)

        # self.decoder = nn.Sequential(
        #     # 2048x3x3x3
        #     nn.Conv3d(2048, 1024, kernel_size=3, stride=1, padding=1), # 1024x3x3x3
        #     ResidualBlock(1024, norm_layer=self.norm_layer),  # 1024x3x3x3

        #     nn.Upsample(scale_factor=2),  # 1024x6x6x6
        #     nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1),  # 512x6x6x6
        #     ResidualBlock(512, norm_layer=self.norm_layer),  # 512x6x6x6

        #     # Start upsampling
        #     nn.Upsample(scale_factor=2),  # 512x12x12x12
        #     nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),  # 256x12x12x12
        #     ResidualBlock(256, norm_layer=self.norm_layer),  # 256x12x12x12

        #     nn.Upsample(scale_factor=2),  # 256x24x24x24
        #     nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),  # 128x24x24x24
        #     ResidualBlock(128, norm_layer=self.norm_layer),  # 128x24x24x24

        #     nn.Conv3d(128, 64, kernel_size=5, stride=1, padding=0), # 64x20x20x20
        #     self.activation(),
        #     self.norm_layer(64),

        #     nn.Conv3d(64, n_coeffs, kernel_size=1, stride=1, padding=0),  # 28x20x20x20
        # )



    def forward(self, x):
        latent = self.encoder(x)
        output = self.decode(latent)
        return output

    def decode(self, latent):
        output = self.decoder(latent)
        # output = output[..., :-1, :-1, :-1]
        return output
    
    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])

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
        # signal = self.neigh_manager.get(coords.to(get_device()))
        # signal = signal.squeeze(0)
        # return signal
        return coords
    
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_warmup_steps, last_epoch=-1):
        self.n_warmup_steps = n_warmup_steps
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.n_warmup_steps:
            return [base_lr * (self.last_epoch / self.n_warmup_steps) for base_lr in self.base_lrs]
        else:
            return self.base_lrs
        

class FodfAeTrainer(object):
    def __init__(self, input_shape, n_coeffs=28, nb_epochs=100, neighborhood_radius=9, batch_size=1, device="cpu"):
        self.model = FodfAe(input_shape=input_shape, n_coeffs=n_coeffs, renorm=False)
        self.device = device
        self.batch_size = batch_size
        self.nb_epochs = 100

        fodf_path = "data/datasets/ismrm2015_2mm/fodfs/ismrm2015_fodf.nii.gz"
        fodf_img = nib.load(fodf_path)
        self.fodf_data = fodf_img.get_fdata().astype(np.float32)
        self.affine = fodf_img.affine
        
        self.neigh_manager = NeighborhoodManager(
            data_volume=self.fodf_data,
            radius=neighborhood_radius,
            add_neighborhood_vox=1,
            flatten=False,
            device=device,
            method='dwi_ml',
            neighborhood_type='grid',
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

        self.train_loader = DataLoader(self.train_dataset, sampler=train_sampler)
        self.valid_loader = DataLoader(self.valid_dataset, sampler=valid_sampler)
        self.test_loader = DataLoader(self.test_dataset, sampler=test_sampler)

        self.project_name = "fodf_ae"
        self.workspace="mrzarfir"
        self.comet_enabled = USE_COMET
        self.experiment = None

        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.warmup_scheduler = WarmupScheduler(self.optimizer, n_warmup_steps=len(self.train_loader)//10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader))
        self._create_comet_experiment()

    def to(self, device):
        self.model.to(device)
        return self
    
    def _get_all_coords(self, img_shape, shuffled=False):
        wm_mask = "data/datasets/ismrm2015_2mm/masks/ismrm2015_wm.nii.gz"
        wm_mask = torch.from_numpy(nib.load(wm_mask).get_fdata())

        grid_coords = torch.meshgrid(
            torch.arange(img_shape[0]),
            torch.arange(img_shape[1]),
            torch.arange(img_shape[2]),
            indexing='ij'
        )

        coords = torch.stack(grid_coords, dim=-1)
        coords = coords.reshape(-1, 3).float()

        all_coords_count = coords.size(0)
        # Only keep the coordinates that are within the white matter mask
        coords = coords[wm_mask[coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()] > 0]
        after_coords_count = coords.size(0)
        print(f"Kept {after_coords_count} out of {all_coords_count} coordinates")

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
        best_loss = torch.inf

        for epoch in tqdm(range(self.nb_epochs), desc="Epochs"):
            self.model.train()
            with tqdm(range(len(self.train_loader)), desc="Training batches", leave=False) as train_batch_pbar:
                for i, coords in enumerate(self.train_loader):
                    coords = coords.squeeze(0).to(self.device)

                    batch = self.neigh_manager.get(coords, torch_convention=True)
                    # if len(batch.shape) > 5:
                    #     batch = batch.squeeze(0)

                    # with torch.autocast(device_type=device, dtype=torch.float16, enabled=False):
                    output = self.model(batch)
                    loss = self.reconstruction_loss(output, batch)
                    loss = loss# * 10000

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if epoch != 0: # Naive warmup here
                        self.scheduler.step()
                    else:
                        self.warmup_scheduler.step()
                    # loss_item = loss.item()
                    # self.train_losses.append(loss_item)

                    if loss < best_loss:
                        best_loss = loss
                        torch.save(self.model.state_dict(), "fodf_ae/best_model_big.pth")
                        torch.save(self.model.encoder.state_dict(), "fodf_ae/best_encoder_big.pth")

                        # Send to Comet.ml

                    train_batch_pbar.update(1)

                    if i % 1 == 0:
                        loss_item = loss.item()
                        train_batch_pbar.set_postfix({"loss": loss_item, "real_loss": loss_item/1000, "lr": self.scheduler.get_last_lr()[0]})
                        self.experiment.log_metric("train_loss", loss_item, step=i, epoch=epoch)
                        self.experiment.log_metric("lr", self.scheduler.get_last_lr()[0], step=i, epoch=epoch)

                    if i % 100 == 0:
                        torch.cuda.empty_cache()

            # avg_loss = np.mean(self.train_losses)
            self.train_losses = []


            self.model.eval()
            with torch.no_grad():
                for coords in tqdm(self.valid_loader, desc="Validation", leave=False):
                    coords = coords.squeeze(0).to(self.device)

                    batch = self.neigh_manager.get(coords, torch_convention=True)
                    # if len(batch.shape) > 5:
                    #     batch = batch.squeeze(0)

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
    
    def predict_examples(self, n=5):
        self.model = self.model.to(self.device)
        # Create the reconst_ex directory if it doesn't exist
        fodf_ae_ex = "fodf_ae_ex"
        targets_dir = os.path.join(fodf_ae_ex, "targets")
        reconsts_dir = os.path.join(fodf_ae_ex, "reconsts")
        crops_dir = os.path.join(fodf_ae_ex, "crops")

        os.makedirs(fodf_ae_ex, exist_ok=True)
        os.makedirs(targets_dir, exist_ok=True)
        os.makedirs(reconsts_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)


        # examples of coordinates
        # coords = [
        #           [self.fodf_data.shape[0]//2, self.fodf_data.shape[1]//2, self.fodf_data.shape[2]//2],
        #           [self.fodf_data.shape[0]//3.5, self.fodf_data.shape[1]//3.5, self.fodf_data.shape[2]//3.5],
        #         #   [1, 1, 1], 
        #           ]
        # coords = torch.tensor(coords, dtype=torch.float32)
        all_coords = self._get_all_coords(self.fodf_data.shape[:-1], shuffled=False)

        torch.random.manual_seed(42)
        all_coords = all_coords[torch.randperm(all_coords.size(0))]
        coords = all_coords[:n]
        coords = coords.to(self.device)

        # Get crops of those regions
        # This is basically the real value if the coordonates are aligned with the voxels
        crops = self.neigh_manager.get_crops(coords)
        for i, crop in enumerate(crops):
            crop_path = os.path.join(crops_dir, f"crop_{i}.nii.gz")
            nib.save(nib.Nifti1Image(crop.cpu().numpy(), self.affine), crop_path)

        # Interpolate the neighborhood and save to 

        targets = self.neigh_manager.get(coords, torch_convention=False)
        rad = self.neigh_manager.radius
        for i, target in enumerate(targets):
            frame = np.zeros_like(self.fodf_data)
            coord = coords[i].long()
            frame[coord[0]-rad:coord[0]+rad+1,
                  coord[1]-rad:coord[1]+rad+1,
                  coord[2]-rad:coord[2]+rad+1] = target.cpu().numpy()
            target_path = os.path.join(targets_dir, f"target_{i}.nii.gz")
            nib.save(nib.Nifti1Image(frame, self.affine), target_path)

        self.model.eval()
        with torch.no_grad():
            for i, coord in enumerate(coords):
                coord = coord.unsqueeze(0)
                coord = coord.to(self.device)

                batch = self.neigh_manager.get(coord, torch_convention=True)
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.reconstruction_loss(output, batch)
                print(f"loss ({i}): {loss}, sum of reconst: {np.abs(output.cpu()).sum()} vs sum of target: {np.abs(batch.cpu()).sum()}")

                # output = output.permute(0, 4, 1, 2, 3)
                output = output.permute(0, 2, 3, 4, 1).squeeze(0)
                reconst = output.squeeze(0).cpu().numpy()

                frame = np.zeros_like(self.fodf_data)
                coord = coord[0].long()
                frame[coord[0]-rad:coord[0]+rad+1,
                        coord[1]-rad:coord[1]+rad+1,
                        coord[2]-rad:coord[2]+rad+1] = reconst

                print(f"sum of target: {np.abs(target.cpu()).sum()}, max of target: {target.max()}, min of target: {target.min()}")
                print(f"sum of frame: {np.abs(frame).sum()}, max of frame: {frame.max()}, min of frame: {frame.min()}")
                reconst_path = os.path.join(reconsts_dir, f"reconst_{i}.nii.gz")
                nib.save(nib.Nifti1Image(frame, self.affine), reconst_path)


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
    
    # trainer.model.load_state_dict(torch.load("fodf_ae/best_model_big.pth"))

    trainer.train()
    
    # trainer.predict_examples(n=5)


if '__main__' == __name__:
    main()
