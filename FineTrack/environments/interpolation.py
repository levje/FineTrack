import numpy as np

# from numba import njit


# @njit
def old_nearest_neighbor_interpolation(
    volume: np.array([[[[]]]]),
    coords: np.ndarray,
) -> np.ndarray:
    """
    """
    coords = coords
    volume = volume

    if volume.ndim <= 3 or volume.ndim >= 5:
        raise ValueError("Volume must be 4D!")

    indices_unclipped = np.round(coords).astype(np.int32)

    # Clip indices to make sure we don't go out-of-bounds
    upper = (np.asarray(volume.shape[:3]) - 1)
    indices = np.clip(indices_unclipped, 0, upper).astype(int).T
    output = volume[tuple(indices)]

    return output

def nearest_neighbor_interpolation(
    volume: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Interpolates the given coordinates using nearest neighbor method on a 4D volume.
    """
    if volume.ndim != 4:
        raise ValueError("Volume must be 4D!")

    # Round and clip coordinates
    indices = np.clip(np.round(coords), 0, np.array(volume.shape[:3]) - 1).astype(np.int32)

    # Efficient indexing
    return volume[indices[:, 0], indices[:, 1], indices[:, 2]]
