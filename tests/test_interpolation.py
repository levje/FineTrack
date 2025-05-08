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
from FineTrack.utils.interpolation import \
    calc_neighborhood_vectors, neighborhood_interpolation

nib_img_path = '/home/jeremi/Documents/FineTrack/data/datasets/archives/1mm/ismrm2015_fodf.nii.gz'
nifti_image = nib.load(nib_img_path)
image_data = nifti_image.get_fdata().astype(np.float32) # Shape: (1, W, D, 45)

@pytest.fixture
def prepare_fodf_volume_and_target():
    fodf_volume = torch.tensor(image_data, dtype=torch.float32)
    # fodf_volume = fodf_volume.permute(3, 0, 1, 2)  # Shape: (45, D, H, W)

    coord = torch.tensor([fodf_volume.shape[0]//2,
                          fodf_volume.shape[1]//2,
                          fodf_volume.shape[2]//2],
                          dtype=torch.long)
    radius = 50

    # Crop the fodf_volume to the neighborhood
    target = fodf_volume[
        coord[0].long()-radius:coord[0].long()+radius+1,
        coord[1].long()-radius:coord[1].long()+radius+1,
        coord[2].long()-radius:coord[2].long()+radius+1]

    return fodf_volume, target, coord.float(), radius

@pytest.fixture
def prepare_fodf_volume_and_targets():
    img = torch.tensor(image_data, dtype=torch.float32)

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
        target = img[
            coord[0].long()-radius:coord[0].long()+radius+1,
            coord[1].long()-radius:coord[1].long()+radius+1,
            coord[2].long()-radius:coord[2].long()+radius+1]
        targets.append(target)

    return img, targets, coords.float(), radius

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
    assert avg_difference < 0.01 # TODO: Is this normal? This error is pretty high.

def test_custom_interpolation_mutiple_coordinates(prepare_fodf_volume_and_targets):
    fodf_volume, targets, coords, radius = prepare_fodf_volume_and_targets

    grid = calc_neighborhood_vectors('grid', radius, resolution=1.0, device=fodf_volume.device)

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
    radius = 9
    D, H, W, _ = image_data.shape
    coord = [D // 2, H // 2, W // 2]

    cropped_img = image_data[
        coord[0]-radius:coord[0]+radius+1,
        coord[1]-radius:coord[1]+radius+1,
        coord[2]-radius:coord[2]+radius+1]
    
    coord = torch.tensor(coord, dtype=torch.float32)

    # Copy the same coordinate 50 times
    coord = coord.repeat(100, 1)


    return image_data, cropped_img, coord, radius

def test_other_interpolation_speedup(prepare_interpolation_test):
    device="cpu"
    image_data, cropped_img, coord, radius = prepare_interpolation_test
    img_tensor = torch.from_numpy(image_data).to(device)
    img_tensor_2 = img_tensor.clone().to(device)
    coord_tensor = coord.to(device)
    coord_tensor_2 = coord_tensor.clone().to(device)

    # Create a normalized grid (e.g., 50x50 grid points in the center of the image)
    grid = calc_neighborhood_vectors('grid', radius, device=device)
    with SimpleTimer() as timer_custom:
        interpolated_img = neighborhood_interpolation(img_tensor, coord_tensor, grid)
    interpolated_img = interpolated_img.cpu().numpy()
    difference = np.mean(np.abs(cropped_img - interpolated_img))
    assert difference < 1e-5

    # DWI-ML style
    n_coef = image_data.shape[-1]
    neighborhood_type = 'grid'
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, radius, neighborhood_resolution).to(device)
    
    grid_side_size = radius*2 + 1

    # Interpolate with dwi_ml
    with SimpleTimer() as timer_dwi_ml:
        signal, _ = interpolate_volume_in_neighborhood(img_tensor_2, coord_tensor_2, neighborhood_vectors)
        signal = signal.view(coord.shape[0], grid_side_size, grid_side_size, grid_side_size, n_coef)
    signal = signal.squeeze(0)
    signal = signal.cpu().numpy()
    difference = np.mean(np.abs(cropped_img - signal))
    assert difference < 0.01 # TODO: Is this normal? This error is pretty high.

    assert timer_custom.interval < timer_dwi_ml.interval, f"Custom interpolation is slower than dwi_ml interpolation: {timer_custom.interval} > {timer_dwi_ml.interval}"

def test_calc_neighborhood_axes():
    """
    The function that prepares the neighborhood vectors
    is not the same as the one implemented in dwi_ml. Just want to make
    sure that it works exactly the same.
    """
    def verify_axes(resolution, radius):
        axes_dwi = prepare_neighborhood_vectors('axes', radius, resolution)
        axes = calc_neighborhood_vectors('axes', radius, resolution)
        assert np.allclose(axes_dwi, axes), f"Axes do not match: {axes_dwi} != {axes}"
    
    verify_axes(1, 1)
    verify_axes(1, 2)
    verify_axes(1, 3)

    verify_axes(0.6, 1)
    verify_axes(0.6, 3)

    verify_axes(0.75, 1)
    verify_axes(0.75, 3)

    verify_axes(1.25, 1)
    verify_axes(1.25, 3)

@pytest.fixture
def prepare_interpolation_axes():
    radius = 1
    D, H, W, _ = image_data.shape
    coord = [D // 2, H // 2, W // 2]

    axes = calc_neighborhood_vectors('axes', 1, 1).numpy().astype(int)
    target = np.zeros((len(axes), image_data.shape[-1]), dtype=np.float32)
    for i, (x, y, z) in enumerate(axes):
        target[i] = image_data[x, y, z, :]
    
    coord = torch.tensor(coord, dtype=torch.float32)

    # Copy the same coordinate 50 times
    nb_coords = 100
    coord = coord.repeat(nb_coords, 1)
    target = np.reshape(target, (target.shape[0]*target.shape[1]))

    return image_data, target, coord, radius

def test_other_interpolation_axes(prepare_interpolation_axes):
    device="cpu"
    image_data, target, coord, radius = prepare_interpolation_axes
    img_tensor = torch.from_numpy(image_data).to(device)
    img_tensor_2 = img_tensor.clone().to(device)
    coord_tensor = coord.to(device)
    coord_tensor_2 = coord_tensor.clone().to(device)

    # Create a normalized grid (e.g., 50x50 grid points in the center of the image)
    grid = calc_neighborhood_vectors('axes', radius, device=device)
    with SimpleTimer() as timer_custom:
        interpolated_img = neighborhood_interpolation(img_tensor, coord_tensor, grid)
    interpolated_img = interpolated_img.cpu().numpy()
    difference = np.mean(np.abs(target - interpolated_img))
    assert difference < 1e-5

    # DWI-ML style
    neighborhood_type = 'axes'
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, radius, neighborhood_resolution).to(device)

    # Interpolate with dwi_ml
    with SimpleTimer() as timer_dwi_ml:
        signal, _ = interpolate_volume_in_neighborhood(img_tensor_2, coord_tensor_2, neighborhood_vectors)
    signal = signal.squeeze(0)
    signal = signal.cpu().numpy()
    difference = np.mean(np.abs(target - signal))
    assert difference < 1e-5

    assert timer_custom.interval < timer_dwi_ml.interval, f"Custom interpolation is slower than dwi_ml interpolation: {timer_custom.interval} > {timer_dwi_ml.interval}"
