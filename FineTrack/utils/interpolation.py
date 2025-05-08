import torch

@torch.no_grad()
def calc_neighborhood_grid(neighborhood_radius: int, device=None, resolution: float = 1.):
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
def neighborhood_interpolation(volume: torch.Tensor, coords: torch.Tensor, neighborhood_grid: torch.Tensor, align_corners: bool = False):
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
def corrected_neighborhood_interpolation(volume: torch.Tensor, coords: torch.Tensor, grid: torch.Tensor, align_corners: bool = True):
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
    if len(grid.shape) == 4:
        grid = grid.expand(coords.shape[0], -1, -1, -1, -1)  # Expand grid to match the number of coordinates
    elif len(grid.shape) == 5:
        assert grid.shape[0] == coords.shape[0], f"grid should be of shape ({coords.shape[0]}, D, H, W, 3)"
    elif len(grid.shape) < 4 or len(grid.shape) > 5:
        raise ValueError("grid should be of shape (N, D, H, W, 3) or (D, H, W, 3)")
    
    if coords.device != volume.device:
        coords = coords.to(volume.device) # This could eventually be slow if we call this function a lot
    if grid.device != volume.device:
        grid = grid.to(volume.device) # This could eventually be slow if we call this function a lot

    D, H, W, _ = volume.shape
    spatial_size = torch.tensor([W, H, D], dtype=torch.float32, device=grid.device)
    grid = grid + coords[:, [2, 1, 0]]

    # Normalize the grid to be between -1 and 1
    offset = 0.5 if align_corners else 0.0
    grid = (grid + offset) * 2 / spatial_size - 1

    # Interpolate
    img_mod = volume.unsqueeze(0).expand(1, -1, -1, -1, -1)  # Shape: (1, C, D, H, W)
    img_mod = img_mod.permute(0, 4, 1, 2, 3)
    interpolated = torch.nn.functional.grid_sample(
        img_mod,
        grid,
        mode='bilinear',
        align_corners=False,
        padding_mode='zeros'
    )

    # Prepare for visualization
    interpolated_img = interpolated.squeeze(0).permute(1, 2, 3, 0).detach().numpy()
    return interpolated_img