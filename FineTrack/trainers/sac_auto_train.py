#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import torch
from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from FineTrack.algorithms.sac_auto import SACAuto, SACAutoHParams
from FineTrack.trainers.sac_train import add_sac_args
from FineTrack.trainers.train import (FineTrackTraining,
                                         add_training_args)
from FineTrack.utils.logging import setup_logging, add_logging_args
from FineTrack.utils.torch_utils import get_device
device = get_device()


class SACAutoFineTrackTraining(FineTrackTraining):
    """
    Train a RL tracking agent using SAC with automatic entropy adjustment.
    """

    def __init__(
        self,
        sac_auto_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        sac_auto_train_dto: dict
        SACAuto training parameters
        comet_experiment: CometExperiment
        Allows for logging and experiment management.
        """

        super().__init__(
            sac_auto_train_dto,
            comet_experiment,
        )

    @property
    def hparams_class(self):
        return SACAutoHParams

    def get_alg(self, max_nb_steps: int, neighborhood_manager):
        alg = SACAuto(
            self.input_size,
            self.action_size,
            self.hp,
            self.rng,
            device)
        return alg


def add_sac_auto_args(parser):
    add_sac_args(parser) # For now, we have the same arguments as SAC.

def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    add_training_args(parser)
    add_sac_auto_args(parser)
    add_logging_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    setup_logging(args)
    print(args)
    experiment = CometExperiment(project_name=args.experiment,
                                workspace=args.workspace, parse_args=False,
                                auto_metric_logging=False,
                                disabled=not args.use_comet)

    # Create and run experiment
    sac_auto_experiment = SACAutoFineTrackTraining(
        vars(args),
        experiment
    )
    sac_auto_experiment.run()


if __name__ == '__main__':
    main()
