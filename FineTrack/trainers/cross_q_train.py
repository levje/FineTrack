#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import torch
from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from FineTrack.algorithms.cross_q import CrossQ, CrossQHParams
from FineTrack.trainers.sac_auto_train import add_sac_auto_args
from FineTrack.trainers.train import (FineTrackTraining,
                                         add_training_args)
from FineTrack.utils.torch_utils import get_device
device = get_device()


class CrossQFineTrackTraining(FineTrackTraining):
    """
    Train a RL tracking agent using CrossQ.
    """

    def __init__(
        self,
        cross_q_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        cross_q_train_dto: dict
        CrossQ training parameters
        comet_experiment: CometExperiment
        Allows for logging and experiment management.
        """

        super().__init__(
            cross_q_train_dto,
            comet_experiment,
        )

    @property
    def hparams_class(self):
        return CrossQHParams

    def get_alg(self, max_nb_steps: int, neighborhood_manager):
        alg = CrossQ(
            self.input_size,
            self.action_size,
            neighborhood_manager,
            self.hp,
            self.rng,
            device)
        return alg


def add_cross_q_auto_args(parser):
    # So far, arguments are identical to SACAuto
    pass


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    
    add_training_args(parser)
    add_sac_auto_args(parser)
    add_cross_q_auto_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    experiment = CometExperiment(project_name=args.experiment,
                                workspace=args.workspace, parse_args=False,
                                auto_metric_logging=False,
                                disabled=not args.use_comet)

    # Create and run experiment
    training_experiment = CrossQFineTrackTraining(
        # Dataset params
        vars(args),
        experiment
    )
    training_experiment.run()


if __name__ == '__main__':
    main()
