from abc import abstractmethod, ABCMeta
from typing import Tuple
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from FineTrack.filterers.streamlines_sampler \
    import StreamlinesSampler

class Filterer(metaclass=ABCMeta):
    
    def __init__(self, sampler: StreamlinesSampler = None):
        self.sampler = sampler

    def __call__(self, tractogram, out_dir, scored_extension="trk"):
        """Filter a list of tracts."""
        valid, invalid = self._filter(tractogram, out_dir, scored_extension)
        valid, invalid = \
            self.sampler.sample_streamlines(valid, invalid)

        # Make sure the number of valid and invalid is almost equal
        assert np.abs(len(valid) - len(invalid)) < 5, \
            f"Number of valid and invalid streamlines differ by more than 5. " \
            f"Valid: {len(valid)}, invalid: {len(invalid)}"
        
        return valid, invalid
        
    @abstractmethod
    def _filter(self, tractogram, out_dir, scored_extension="trk") -> Tuple[StatefulTractogram, StatefulTractogram]:
        raise NotImplementedError("Filter method not implemented.")

    

