import torch

# -*- coding: utf-8 -*-

import numpy as np
import torch


def calc_neighborhood_vectors(
        mode: str, radius: int,
        resolution: float = 1.0,
        device=None) -> torch.Tensor:
    """
    Prepare neighborhood vectors.

    Note: We only support isometric voxels! Adding isometry would also require
    the voxel resolution.

    Params
    ------
    mode: str
        Either the 'axes' option or the 'grid' option. See each method for a
        description.
    radius: int
        Required if neighborhood type is not None.
        - axes: a radius of 1 = 7 neighbhors, a radius of 2 = 13, on two
         concentric spheres.
        - grid option: a radius of 1 = 27 neighbors, a radius of 2 = 125, on
         two concentric cubes.
    resolution: float
        Required if neighborhood type is not None.
        - 'axes': spacing between each concentric sphere.
        - 'grid': resolution of the final grid-like neighborhood.
        Hint: To convert from mm to voxel world, you may use
        dwi_ml.data.processing.space.world_to_vox.convert_world_to_vox(
            radius_mm, affine_mm_to_vox)
    device: torch.device
        Device to use for the neighborhood vectors. If None, will use the
        default device (usually CPU). If you are using a GPU, you can set
        this to torch.device('cuda') to speed up the calculations.

    Returns
    -------
    neighborhood_vectors: tensor of shape (N, 3).
        Results are vectors pointing to a neighborhood point, starting from the
        origin (i.e. current position). The current point (0,0,0) is included.
        Hint: You can now interpolate your DWI data in each direction around
        your point of interest to get your neighbourhood.
        Returns None if neighborhood_radius is None.
    """
    if mode is None:
        return None

    if radius is None:
        raise ValueError("You must provide neighborhood radius to add "
                         "a neighborhood.")
    if resolution is None:
        raise ValueError("You must provide neighborhood resolution to add "
                         "a neighborhood.")
    if mode not in ['axes', 'grid']:
        raise ValueError(
            "Mode must be either 'axes', 'grid' "
            "but we received {}!".format(mode))

    if mode == 'axes':
        neighborhood_vectors = calc_neighborhood_axes(
            radius, resolution, device=device)
    else:
        neighborhood_vectors = calc_neighborhood_grid(
            radius, resolution, device=device)

    return neighborhood_vectors

@torch.no_grad()
def calc_neighborhood_axes(radius: int, resolution: float, device=None) -> torch.Tensor:
    """
    This neighborhood definition lies on a sphere.

    For radius = 1, returns a list of 7 positions (current, up, down, left,
    right, behind, in front) at exactly `resolution` (mm or voxels) from origin
    (i.e. current postion).
    If radius is > 1, returns a multi-radius neighborhood (lying on
    concentring spheres).

    Returns
    -------
    neighborhood_vectors : tensor of shape (N, 3)
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin).
    """
    if radius > 1:
        import warnings
        warnings.warn("Neighborhood radius > 1 wasn't tested for axes. Make sure the behavior is "
                        "as expected. If not, please open an issue on GitHub.")

    tmp_axes = torch.eye(3, dtype=torch.float32)
    unit_axes = torch.concat((tmp_axes, -tmp_axes), dim=0)

    radiuses = torch.arange(1, radius + 1, dtype=torch.float32) * resolution
    neighborhood_vectors = [torch.tensor([0, 0, 0], dtype=torch.float32)]
    for r in radiuses:
        neighborhood_vectors.extend(unit_axes * r)

    neighborhood_vectors = torch.stack(neighborhood_vectors, dim=0).to(device=device)

    return neighborhood_vectors


"""
CUSTOM FUNCTIONS
"""
@torch.no_grad()
def calc_neighborhood_grid(neighborhood_radius: int, resolution: float = 1., device=None):
    # Get the neighborhood grid for the coordinates
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(-neighborhood_radius, neighborhood_radius+1),
        torch.arange(-neighborhood_radius, neighborhood_radius+1),
        torch.arange(-neighborhood_radius, neighborhood_radius+1),
        indexing='ij')
    
    # Convert the neighborhood grid to a tensor
    neighborhood_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).float().to(device=device)

    # Scale the neighborhood_grid to the specified resolution
    neighborhood_grid *= resolution

    return neighborhood_grid

@torch.no_grad()
def old_neighborhood_interpolation(volume: torch.Tensor, coords: torch.Tensor, neighborhood_grid: torch.Tensor, align_corners: bool = False):
    """
    This function interpolates a volume at given coordinates using a neighborhood
    of points around the coordinates.

    The volume is of shape (C, D, H, W) where C is the number of channels, H is the
    height, W is the width and D is the depth of the volume.

    The coordinates are of shape (N, 3) where N is the number of coordinates and
    the 3 columns are the x, y, z coordinates.

    The neighborhood is a grid of points around the coordinates. The radius of the
    neighborhood is given by neighborhood_radius. For example, if neighborhood_radius
    is 1, the neighborhood will have 27 points. If neighborhood_radius is 2, the
    neighborhood will have 125 points. The neighborhood grid for each point is
    of shape (D_neigh, H_neigh, W_neigh) where D_neigh, H_neigh, W_neigh are the
    depth, height and width of the neighborhood grid.

    We need to use the torch.nn.functional.grid_sample function to interpolate the
    volume at the coordinates. We need to prepare the grid for the grid_sample
    function. The grid is of shape (N, D_neigh, H_neigh, W_neigh, 3) where N is
    the number of coordinates and the last dimension is the x, y, z coordinates

    The function returns the interpolated values at the coordinates. The output
    is of shape (N, C, D_neigh, H_neigh, W_neigh).
    """
    N = coords.shape[0] # Coords are in the x, y, z format

    # Add the coordinates to the neighborhood grid
    neighborhood_grid = neighborhood_grid + coords.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
    neighborhood_grid = neighborhood_grid.permute(0, 3, 1, 2, 4)

    # Can we create a view of the volume that is of shape (N, C, D, H, W) to use grid_sample?
    # N being the number of coordinates.
    volume_mod = volume.permute(3, 0, 1, 2).unsqueeze(0)
    volume_mod = volume_mod.expand(N, -1, -1, -1, -1)

    # We need to normalize the grid coordinates to be between -1 and 1 before giving it to grid_sample
    offset = 0.0
    if align_corners:
        offset = 0.5
    neighborhood_grid = 2 * (neighborhood_grid + offset) / torch.tensor([volume_mod.shape[2],
                                                              volume_mod.shape[3],
                                                              volume_mod.shape[4]],
                                                              dtype=torch.float32,
                                                              device=volume_mod.device) - 1



    # Interpolate the volume at the coordinates using grid_sample
    # 'bilinear' interpolation is specified, however, according to the documentation
    # (https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
    # if the input is 5D, the interpolation mode will actually be 'trilinear'.
    #
    # Note: dwi_ml clamps the indices to the volume shape which is similar to
    # padding_mode='border' in grid_sample. Here we use padding_mode='zeros'
    # as I want the agent to know that it's on the edge of the image.
    #
    interpolated_volume = torch.nn.functional.grid_sample(
        volume_mod, neighborhood_grid, mode='bilinear', align_corners=align_corners, padding_mode='border'
    )

    interpolated_volume = interpolated_volume.permute(0, 2, 3, 4, 1)
    
    return interpolated_volume


# Correction from ChatGPT
@torch.no_grad()
def chatgpt_neighborhood_interpolation(volume: torch.Tensor, coords: torch.Tensor, neighborhood_grid: torch.Tensor, align_corners: bool = False):
    """
    Interpolates a 3D volume at given coordinates using a local neighborhood grid.
    """
    N = coords.shape[0]

    # volume: (C, D, H, W) -> (N, C, D, H, W)
    volume_mod = volume.unsqueeze(0).expand(N, -1, -1, -1, -1)

    # Add coordinate center to each neighborhood grid
    # coords: (N, 3) -> (N, 1, 1, 1, 3), broadcast with neighborhood_grid
    neighborhood_grid = neighborhood_grid + coords.view(N, 1, 1, 1, 3) # Each coord point is considered (D, H, W)

    # Normalize grid to [-1, 1]
    spatial_shape = torch.tensor(
        [volume.shape[0], volume.shape[1], volume.shape[2]],
        device=volume.device,
        dtype=torch.float32
    )
    offset = 0.5
    norm_grid = ((neighborhood_grid + offset) * 2 / spatial_shape) - 1 # Each point is considered (D, H, W)

    volume_mod = volume_mod.permute(0, 4, 1, 2, 3)
    interpolated_volume = torch.nn.functional.grid_sample(
        volume_mod,
        norm_grid,
        mode='bilinear',
        align_corners=False,
        padding_mode='zeros'
    )
    interpolated_volume = interpolated_volume.permute(0, 2, 3, 4, 1)
    return interpolated_volume

@torch.no_grad()
def neighborhood_interpolation(volume: torch.Tensor, coords: torch.Tensor, grid: torch.Tensor, align_corners: bool = True):
    """
    Interpolates a 3D volume at given coordinates using a local neighborhood grid.
    ----
    Parameters
    ----------
    volume: torch.Tensor
        The 3D volume to be interpolated. Shape: (C, D, H, W)
    coords: torch.Tensor
        The coordinates at which to interpolate the volume. Shape: (N, 3) or (3,)
    grid: torch.Tensor
        The neighborhood grid to be used for interpolation. Shape: (N, D, H, W, 3) or (D, H, W, 3)
    align_corners: bool
        Whether to align corners when interpolating. Default is True.
    ----
    Returns
    -------
    interpolated_img: numpy.ndarray
        The interpolated image at the given coordinates. Shape: (N, D, H, W, C)
    ----
    """

    # Make sure we have tensors as inputs
    assert isinstance(volume, torch.Tensor), "volume should be a torch.Tensor"
    assert isinstance(coords, torch.Tensor), "coords should be a torch.Tensor"
    assert isinstance(grid, torch.Tensor), "grid should be a torch.Tensor"

    # Make sure the coordinates are of shape (N, 3)
    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)
    elif len(coords.shape) == 0 or len(coords.shape) > 2:
        raise ValueError("coords should be of shape (N, 3) or (3,)")

    # Make sure the grid is of shape (N, D, H, W, 3)
    is_grid = True
    if len(grid.shape) == 4: # (D, H, W, 3)
        grid = grid.expand(coords.shape[0], -1, -1, -1, -1)  # Expand grid to match the number of coordinates
    elif len(grid.shape) == 5: # (N, D, H, W, 3)
        assert grid.shape[0] == coords.shape[0], f"grid should be of shape ({coords.shape[0]}, D, H, W, 3)"
    elif len(grid.shape) == 2: # (W, 3)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(coords.shape[0], -1, -1, -1, -1)  # Expand grid to match the number of coordinates
        is_grid = False # When using "axes"
    elif len(grid.shape) == 3: # (N, W, 3)
        assert grid.shape[0] == coords.shape[0], f"grid should be of shape ({coords.shape[0]}, W, 3)"
        grid = grid.unsqueeze(1).expand(coords.shape[0], -1, -1, -1, -1)
        is_grid = False # When using "axes"
    elif len(grid.shape) < 4 or len(grid.shape) > 5:
        raise ValueError("grid should be of shape (N, D, H, W, 3) or (D, H, W, 3), but we received {}".format(grid.shape))
    
    if coords.device != volume.device:
        coords = coords.to(volume.device) # This could eventually be slow if we call this function a lot
    if grid.device != volume.device:
        grid = grid.to(volume.device) # This could eventually be slow if we call this function a lot

    D, H, W, _ = volume.shape
    spatial_size = torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32, device=grid.device)
    grid = grid + coords[:, None, None, None, [2, 1, 0]]

    # Normalize the grid to be between -1 and 1
    grid = (grid * 2 / spatial_size) - 1

    # Interpolate
    img_mod = volume.unsqueeze(0).expand(grid.shape[0], -1, -1, -1, -1)  # Shape: (1, C, D, H, W)
    img_mod = img_mod.permute(0, 4, 1, 2, 3)
    interpolated = torch.nn.functional.grid_sample(
        img_mod,
        grid,
        mode='bilinear',
        align_corners=align_corners,
        padding_mode='zeros'
    )

    # Prepare for visualization
    interpolated_img = interpolated.permute(0, 2, 3, 4, 1)

    if not is_grid:
        interpolated_img = interpolated_img.reshape(coords.shape[0], -1)

    return interpolated_img