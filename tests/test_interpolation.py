import pytest

import torch
import numpy as np
import nibabel as nib
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import \
    unflatten_neighborhood, prepare_neighborhood_vectors
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from FineTrack.utils.utils import SimpleTimer
from FineTrack.utils.interpolation import chatgpt_neighborhood_interpolation as neighborhood_interpolation, calc_neighborhood_grid, corrected_neighborhood_interpolation

nib_img_path = '/Users/jeremilevesque/Documents/uni/FineTrack/data/datasets/ismrm2015_1mm/fodfs/ismrm2015_fodf.nii.gz'

@pytest.fixture
def prepare_fodf_volume_and_target():
    nib_img = nib.load(nib_img_path)
    img = nib_img.get_fdata()
    fodf_volume = torch.tensor(img, dtype=torch.float32)
    # fodf_volume = fodf_volume.permute(3, 0, 1, 2)  # Shape: (45, D, H, W)

    coord = torch.tensor([img.shape[0]//2, img.shape[1]//2, img.shape[2]//2], dtype=torch.long)
    radius = 50

    # Crop the fodf_volume to the neighborhood
    target = fodf_volume[
        coord[0].long()-radius:coord[0].long()+radius+1,
        coord[1].long()-radius:coord[1].long()+radius+1,
        coord[2].long()-radius:coord[2].long()+radius+1]

    return fodf_volume, target, coord.float(), radius

@pytest.fixture
def prepare_fodf_volume_and_targets():
    nib_img = nib.load(nib_img_path)
    img = nib_img.get_fdata()
    fodf_volume = torch.tensor(img, dtype=torch.float32)

    coords = torch.tensor([
        [img.shape[0]//2, img.shape[1]//2, img.shape[2]//2],
        [img.shape[0]//2+5, img.shape[1]//2+5, img.shape[2]//2+5],
        [img.shape[0]//2-5, img.shape[1]//2-5, img.shape[2]//2-5],
        [img.shape[0]//2+5, img.shape[1]//2-5, img.shape[2]//2-5],
        [img.shape[0]//2+5, img.shape[1]//2+5, img.shape[2]//2-5]
    ], dtype=torch.long)
    radius = 10

    targets = []
    for coord in coords:
        target = fodf_volume[
            coord[0].long()-radius:coord[0].long()+radius+1,
            coord[1].long()-radius:coord[1].long()+radius+1,
            coord[2].long()-radius:coord[2].long()+radius+1]
        targets.append(target)

    return fodf_volume, targets, coords.float(), radius

def test_dwiml_interpolation_crop(prepare_fodf_volume_and_target):
    fodf_volume, target, coord, radius = prepare_fodf_volume_and_target

    n_coef = fodf_volume.shape[-1]
    neighborhood_type = 'grid'
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, radius, neighborhood_resolution)
    
    grid_side_size = radius*2 + 1

    # Interpolate with dwi_ml
    signal, _ = interpolate_volume_in_neighborhood(fodf_volume, coord.unsqueeze(0), neighborhood_vectors)
    assert signal.shape == (1, grid_side_size*grid_side_size*grid_side_size*n_coef)

    # Unflatten the neighborhood
    # signal = unflatten_neighborhood(
    #                 signal, neighborhood_vectors, 'grid',
    #                 radius, neighborhood_resolution)
    signal = signal.view(1, grid_side_size, grid_side_size, grid_side_size, n_coef)
    assert signal.shape == (1, grid_side_size, grid_side_size, grid_side_size, n_coef)

    difference = torch.abs(target - signal[0])
    error_ratio = difference.sum() / (difference>=0).sum()
    assert error_ratio < 0.03

    avg_difference = difference.mean()
    print("avg_difference:", avg_difference)
    assert avg_difference < 1e-5


def test_custom_interpolation_crop(prepare_fodf_volume_and_target):
    # Prepare data
    fodf_volume, target, coord, radius = prepare_fodf_volume_and_target

    grid = calc_neighborhood_grid(radius, device=fodf_volume.device, resolution=1.0)

    # Interpolate with custom interpolation technique
    signal = neighborhood_interpolation(fodf_volume, coord.unsqueeze(0), grid)
    signal = signal.squeeze(0)

    print("signal.shape:", signal.shape)
    print("target.shape:", target.shape)

    difference = torch.abs(target - signal)
    error_ratio = difference.mean()

    import matplotlib.pyplot as plt
    def plot_for_coefficient(coefficient, num_rows, index):
        plt.subplot(num_rows, 4, index)
        plt.title("Original Image")
        plt.imshow(fodf_volume[coord[0].to(int), :, :, coefficient].cpu().numpy())
        plt.axis("off")
        plt.subplot(num_rows, 4, index+1)
        plt.title("Target Image")
        plt.imshow(target[radius, :, :, coefficient].cpu().numpy())
        plt.axis("off")
        plt.subplot(num_rows, 4, index+2)
        plt.title("Interpolated Image")
        plt.imshow(signal[radius, :, :, coefficient].cpu().numpy())
        plt.axis("off")
        plt.subplot(num_rows, 4, index+3)
        plt.title("Difference map")
        plt.imshow(difference[radius, :, :, coefficient].cpu().numpy())
        plt.axis("off")
    plot_for_coefficient(0, 3, 1)
    plot_for_coefficient(1, 3, 5)
    plot_for_coefficient(2, 3, 9)
    plt.show()

    assert error_ratio < 1e-8

def test_custom_interpolation_mutiple_coordinates(prepare_fodf_volume_and_targets):
    fodf_volume, targets, coords, radius = prepare_fodf_volume_and_targets

    grid = calc_neighborhood_grid(radius, device=fodf_volume.device, resolution=1.0)

    # Interpolate with custom interpolation technique
    signal = neighborhood_interpolation(fodf_volume, coords, grid)
    
    differences = []
    for i, target in enumerate(targets):
        difference = torch.abs(target - signal[i])
        error_ratio = difference.sum() / (difference>=0).sum()
        differences.append(error_ratio)
    
    for error_ratio in differences:
        assert error_ratio < 1e-5 # TODO: We need to reduce the error ratio here.

@pytest.fixture
def prepare_interpolation_test():
    radius = 50
    nifti_image = nib.load("/Users/jeremilevesque/Documents/uni/FineTrack/data/datasets/ismrm2015_1mm/fodfs/ismrm2015_fodf.nii.gz")
    image_data = nifti_image.get_fdata().astype(np.float32) # Shape: (1, W, D, 45)
    D, H, W, _ = image_data.shape
    coord = [D // 2, H // 2, W // 2]

    cropped_img = image_data[
        coord[0]-radius:coord[0]+radius+1,
        coord[1]-radius:coord[1]+radius+1,
        coord[2]-radius:coord[2]+radius+1]

    return image_data, cropped_img, coord, radius


def test_other_test_interpolation(prepare_interpolation_test):
    image_data, cropped_img, coord, radius = prepare_interpolation_test
    img_tensor = torch.from_numpy(image_data)

    # Create a normalized grid (e.g., 50x50 grid points in the center of the image)
    grid = calc_neighborhood_grid(radius)
    interpolated_img = corrected_neighborhood_interpolation(img_tensor, torch.tensor(coord, dtype=torch.float32), grid)

    difference = np.mean(np.abs(cropped_img - interpolated_img))
    assert difference < 1e-5
    print("difference avg: ", difference)


def test_other_interpolation_speedup(prepare_interpolation_test):
    image_data, cropped_img, coord, radius = prepare_interpolation_test
    img_tensor = torch.from_numpy(image_data).to("mps")
    img_tensor_2 = img_tensor.clone().to('mps')
    coord_tensor = torch.tensor(coord, dtype=torch.float32)

    # Create a normalized grid (e.g., 50x50 grid points in the center of the image)
    grid = calc_neighborhood_grid(radius)
    with SimpleTimer() as timer_custom:
        interpolated_img = corrected_neighborhood_interpolation(img_tensor, torch.tensor(coord, dtype=torch.float32), grid)

    difference = np.mean(np.abs(cropped_img - interpolated_img))
    assert difference < 1e-5

    # DWI-ML style
    n_coef = image_data.shape[-1]
    neighborhood_type = 'grid'
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, radius, neighborhood_resolution)
    
    grid_side_size = radius*2 + 1

    # Interpolate with dwi_ml
    with SimpleTimer() as timer_dwi_ml:
        signal, _ = interpolate_volume_in_neighborhood(img_tensor_2, coord_tensor.unsqueeze(0), neighborhood_vectors)
        signal = signal.view(1, grid_side_size, grid_side_size, grid_side_size, n_coef)
    signal = signal.squeeze(0)
    signal = signal.cpu().numpy()
    difference = np.mean(np.abs(cropped_img - signal))
    assert difference < 1e-5

    print("timer_custom.interval:", timer_custom.interval)
    print("timer_dwi_ml.interval:", timer_dwi_ml.interval)
    assert timer_custom.interval < timer_dwi_ml.interval, f"Custom interpolation is slower than dwi_ml interpolation: {timer_custom.interval} > {timer_dwi_ml.interval}"

    
