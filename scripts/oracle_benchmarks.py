import argparse
import h5py
import numpy as np
from typing import List
from dipy.tracking.streamline import set_number_of_points

from TrackToLearn.oracles.oracle import OracleSingleton
from TrackToLearn.utils.utils import SimpleTimer
from oracle_histogram_length import load_model
from TrackToLearn.utils.torch_utils import get_device
from nibabel.streamlines.array_sequence import ArraySequence


DEFAULT_REFERENCE = '/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/scoring_data/t1.nii.gz'
DEFAULT_DATASETS = [
    ('/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/streamlines/stable/train_test_classical_tracts_antoine.hdf5', DEFAULT_REFERENCE),
    ('/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/streamlines/stable/train_test_sac_even_bigger_dataset_eq.hdf5', DEFAULT_REFERENCE)
]

def _load_test_data(path, nb_points, reference_path=None):

    if path.endswith('.hdf5'):
        with h5py.File(path, 'r') as f:
            streamlines = f['test/data'][:]
            scores = f['test/scores'][:]

            if streamlines.shape[1] != nb_points:
                array_seq = ArraySequence(streamlines)
                streamlines = set_number_of_points(array_seq, nb_points)
    elif path.endswith('.trk') or path.endswith('.tck'):
        raise NotImplementedError('Need to implement this')
    else:
        raise ValueError('Unknown file format')

    return streamlines, scores

class OracleBenchmarker():
    def __init__(self, model: OracleSingleton, datasets: List[str]):
        self.oracle = model
        self.nb_points = self.oracle.nb_points
        self.datasets = datasets
        self.device = get_device()

    def run(self):
        print(f'Running benchmarks for {self.oracle.__class__.__name__}')
        for i, (streamlines, ref_path) in enumerate(self.datasets):
            self._benchmark_dataset(streamlines, ref_path, i)

    def _benchmark_dataset(self, streamlines_path, ref_path, dataset_no):
        print("=========================================")
        print(f'Benchmarking dataset {dataset_no}')
        print(f'Dataset: {streamlines_path}')
        print(f'Model: {self.oracle.__class__.__name__}')
        print(f'Number of points: {self.nb_points}')
        print("------------------out--------------------")

        streamlines, scores = _load_test_data(streamlines_path, self.nb_points, ref_path)

        with SimpleTimer() as t:
            preds = self.oracle.predict(streamlines)
            predictions_binary = (preds > 0.5)
        
        acc = np.mean((predictions_binary == scores.astype(int)).astype(np.float32))
        print(f'Predicted {len(streamlines)} samples in {t.interval:.2f} seconds')
        print(f'Accuracy: {acc}')

        print("=========================================")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    model_singleton = load_model(args.checkpoint, singleton=True)
    b = OracleBenchmarker(model_singleton, DEFAULT_DATASETS)
    b.run()

if __name__ == '__main__':
    main()
