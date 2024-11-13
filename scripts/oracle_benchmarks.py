import argparse
import h5py
import numpy as np
import torch
from time import time
from typing import List
from dipy.tracking.streamline import set_number_of_points
import logging

from TrackToLearn.oracles.oracle import OracleSingleton, TransformerOracle
from TrackToLearn.utils.utils import SimpleTimer
from oracle_histogram_length import load_sft, predict, load_model
from TrackToLearn.environments.rollout_env import RolloutEnvironment

DEFAULT_REFERENCE = '/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/scoring_data/t1.nii.gz'
DEFAULT_DATASETS = [
    ('/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/streamlines/stable/train_test_classical_tracts_antoine.hdf5', DEFAULT_REFERENCE),
    ('/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/streamlines/stable/train_test_sac_even_bigger_dataset_eq.hdf5', DEFAULT_REFERENCE)
]

def _load_test_data(path, reference_path=None):

    if path.endswith('.hdf5'):
        with h5py.File(path, 'r') as f:
            streamlines = np.array(f['test/data'])
            scores = np.array(f['test/scores'])
    elif path.endswith('.trk') or path.endswith('.tck'):
        sft = load_sft(path, reference_path)
        
        sft.to_vox()
        sft.to_corner()

        streamlines = sft.streamlines
        scores = sft.data_per_point['score']

        raise NotImplementedError('Need to implement this')
    else:
        raise ValueError('Unknown file format')

    return streamlines, scores

class OracleBenchmarker():
    def __init__(self, model: OracleSingleton, datasets: List[str]):
        self.oracle = model
        self.nb_points = self.oracle.nb_points
        self.datasets = datasets

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

        streamlines, scores = _load_test_data(streamlines_path)
        
        with SimpleTimer() as t:
            preds = self.oracle.predict(streamlines)
            predictions_binary = (preds > 0.5).astype(np.int32)
        
        print(f'Predicted {len(streamlines)} samples in {t.interval:.2f} seconds')
        print(f'Accuracy: {np.mean(predictions_binary == scores)}')

        # with SimpleTimer() as t:
        #     predictions = predict(streamlines, scores, self.oracle)

        # print(f'Predicted {len(streamlines)} samples in {t.interval:.2f} seconds')
        # print(f'Accuracy: {np.mean(predictions == scores)}')

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
