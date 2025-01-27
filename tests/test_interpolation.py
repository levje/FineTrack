import pytest

import torch
import numpy as np
import nibabel as nib
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import \
    unflatten_neighborhood, prepare_neighborhood_vectors

from FineTrack.utils.interpolation import neighborhood_interpolation

# @pytest.fixture
# def prepare_fodf_volume():
#     fodf_volume = torch.rand(100, 100, 100, 28)
#     return fodf_volume

@pytest.fixture
def prepare_fodf_volume():
    fodf_volume = torch.rand(100, 100, 100, 28)
    volume = torch.arange(100*100*100*28, dtype=torch.float32).reshape(100, 100, 100, 28)
    return volume

@pytest.fixture
def prepare_dwi_ml_neighborhood():
    neighborhood_type = 'grid'
    neighborhood_radius = 9
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, neighborhood_radius, neighborhood_resolution)
    return neighborhood_vectors

def test_custom_interpolation(prepare_fodf_volume, prepare_dwi_ml_neighborhood):
    # Prepare data
    fodf_volume = prepare_fodf_volume
    neighborhood_vectors = prepare_dwi_ml_neighborhood
    
    # Prepare coordinates
    coords = torch.tensor([[10, 10, 10], [20, 20, 20]], dtype=torch.float32)
    # coords = torch.tensor([[10, 10, 10]], dtype=torch.float32)

    # Interpolate with dwi_ml
    target_signal, _ = interpolate_volume_in_neighborhood(
        fodf_volume, coords, neighborhood_vectors, clear_cache=False)
    assert target_signal.shape == (len(coords), 19*19*19*28)

    # Unflatten the neighborhood
    target_signal = unflatten_neighborhood(
                    target_signal, neighborhood_vectors, 'grid',
                    9, 1)
    target_signal = target_signal.permute(0, 4, 3, 2, 1)
    assert target_signal.shape == (len(coords), 28, 19, 19, 19)

    # Now, interpolate with custom interpolation technique
    fodf_volume = fodf_volume.permute(3, 2, 1, 0)
    signal = neighborhood_interpolation(fodf_volume, coords, 9)
    assert signal.shape == (len(coords), 28, 19, 19, 19)

    difference = torch.abs(target_signal - signal)

    point1 = target_signal[0, :, 9, 9, 9]
    point2 = signal[0, :, 9, 9, 9]

    assert torch.all(difference < 1e-6)

    
