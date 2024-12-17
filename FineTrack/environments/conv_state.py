import torch
from dataclasses import dataclass

@dataclass
class ConvStateShape(object):
    nb_streamlines: int

    # Sizes for the convolutional state
    nb_sh_coefs: int
    depth: int
    height: int
    width: int

    # Sizes for the previous directions
    prev_dirs: int

    def __post_init__(self):
        assert self.nb_streamlines >= 0, "Number of streamlines must be positive."
        assert self.nb_sh_coefs >= 0, "Number of spherical harmonics coefficients must be positive."
        assert self.depth >= 0, "Depth must be positive."
        assert self.height >= 0, "Height must be positive."
        assert self.width >= 0, "Width must be positive."
        assert self.prev_dirs >= 0, "Number of previous directions must be positive."

        self.conv_state_common_shape = (self.nb_sh_coefs, self.depth, self.height, self.width)

    def to_dict(self):
        return {
            'nb_streamlines': self.nb_streamlines,
            'nb_sh_coefs': self.nb_sh_coefs,
            'depth': self.depth,
            'height': self.height,
            'width': self.width,
            'prev_dirs': self.prev_dirs,
            'conv_state_common_shape': self.conv_state_common_shape
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(
            nb_streamlines=d['nb_streamlines'],
            nb_sh_coefs=d['nb_sh_coefs'],
            depth=d['depth'],
            height=d['height'],
            width=d['width'],
            prev_dirs=d['prev_dirs']
        )
    
    def __repr__(self):
        return f"ConvStateShape({self.nb_streamlines}, {self.nb_sh_coefs}, {self.depth}, {self.height}, {self.width}, {self.prev_dirs})"
    
    def __str__(self):
        return f"ConvStateShape({self.nb_streamlines}, {self.nb_sh_coefs}, {self.depth}, {self.height}, {self.width}, {self.prev_dirs})"

class ConvState(object):
    def __init__(self, state_conv=None, previous_directions=None, device=None):
        if state_conv is not None:
            self._state_conv = state_conv
        else:
            self._state_conv = torch.tensor([], dtype=torch.float32)
            raise ValueError("Convolutional state must be provided.")

        if previous_directions is not None:
            self._previous_directions = previous_directions
        else:
            self._previous_directions = torch.tensor([], dtype=torch.float32)
            raise ValueError("Previous directions must be provided.")

        self._shape = ConvStateShape(
            nb_streamlines=self._state_conv.shape[0],
            nb_sh_coefs=self._state_conv.shape[1],
            depth=self._state_conv.shape[2],
            height=self._state_conv.shape[3],
            width=self._state_conv.shape[4],
            prev_dirs=self._previous_directions.shape[1]
        )
        
    @classmethod
    def zeros(self, shape, prev_dirs_size, device=None, dtype=torch.float32):
        state_conv = torch.zeros(shape, device=device, dtype=dtype)
        previous_directions = torch.zeros((shape[0], prev_dirs_size), device=device, dtype=dtype)
        return ConvState(state_conv, previous_directions)
    
    @classmethod
    def ones(self, shape, prev_dirs_size, device=None, dtype=torch.float32):
        state_conv = torch.ones(shape, dtype=dtype, device=device)
        previous_directions = torch.ones((shape[0], prev_dirs_size), dtype=dtype, device=device)
        return ConvState(state_conv, previous_directions)

    def to(self, device, copy=False, non_blocking=False):
        self._state_conv = self._state_conv.to(device, copy=copy, non_blocking=non_blocking)
        self._previous_directions = self._previous_directions.to(device, copy=copy, non_blocking=non_blocking)
        return self
    
    def pin_memory(self):
        self._state_conv = self._state_conv.pin_memory()
        self._previous_directions = self._previous_directions.pin_memory()
        return self
    
    def index_select(self, dim, index):
        state_slice = self._state_conv.index_select(dim, index)
        dirs_slice = self._previous_directions.index_select(dim, index)
        return ConvState(state_slice, dirs_slice)

    @property
    def shape(self):
        return self._shape
    
    def __len__(self):
        return self._state_conv.shape[0]

    @property
    def conv_state(self):
        return self._state_conv
    
    @conv_state.setter
    def conv_state(self, value):
        self._state_conv = value

    @property
    def prev_dirs(self):
        return self._previous_directions
    
    @prev_dirs.setter
    def prev_dirs(self, value):
        self._previous_directions = value

    def __getitem__(self, indices):
        state_slice = self._state_conv[indices]
        dirs_slice = self._previous_directions[indices]
        return ConvState(state_slice, dirs_slice)
    
    def __setitem__(self, indices, other):
        if isinstance(other, ConvState):
            self._state_conv[indices] = other._state_conv
            self._previous_directions[indices] = other._previous_directions
        elif isinstance(other, tuple) or isinstance(other, list):
            assert len(other) == 2, "Expected a tuple of tensors holding" \
                " the state and previous directions only."
            
            self._state_conv[indices] = other[0]
            self._previous_directions[indices] = other[1]
        elif isinstance(other, torch.Tensor):
            assert other.shape[0] == 2, "Expected a tensor of dim 2 holding" \
                " the state and previous directions only."
            self._state_conv[indices] = other[0]
            self._previous_directions[indices] = other[1]
        else:
            raise ValueError("Expected a ConvState or a tuple of tensors.")