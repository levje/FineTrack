import pytest

import torch
import numpy as np
from dwi_ml.data.processing.volume.interpolation import interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import prepare_neighborhood_vectors

@pytest.fixture
def prepare_fodf_volume_and_target():
    rng = np.random.RandomState(42)
    image_data = rng.rand(136, 168, 132, 45)
    fodf_volume = torch.tensor(image_data, dtype=torch.float32)

    # Coordinate in the middle of the volume
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

    signal = signal.view(1, grid_side_size, grid_side_size, grid_side_size, n_coef)
    assert signal.shape == (1, grid_side_size, grid_side_size, grid_side_size, n_coef)

    difference = torch.abs(target - signal[0])

    avg_difference = difference.mean()
    assert avg_difference < 1e-5 # TODO: Is this normal? This error is pretty high (0.1667).
