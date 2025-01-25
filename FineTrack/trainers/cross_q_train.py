#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import torch
from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from FineTrack.algorithms.cross_q import CrossQ, CrossQHParams
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
        sac_auto_train_dto: dict
        CrossQ training parameters
        comet_experiment: CometExperiment
        Allows for logging and experiment management.
        """

        super().__init__(
            cross_q_train_dto,
            comet_experiment,
        )

        # SACAuto-specific parameters
        self.alpha = cross_q_train_dto['alpha']
        self.batch_size = cross_q_train_dto['batch_size']
        self.replay_size = cross_q_train_dto['replay_size']
        self.fodf_encoder_ckpt = cross_q_train_dto['fodf_encoder_ckpt']

        self.cross_q_hparams = CrossQHParams(
            self.lr,
            self.gamma,
            self.n_actor,
            self.alpha,
            self.batch_size,
            self.replay_size
        )

    def save_hyperparameters(self):
        """ Add SACAuto-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'SACAuto',
             'alpha': self.alpha,
             'batch_size': self.batch_size,
             'replay_size': self.replay_size,
             'big_neighborhood': self.big_neighborhood,
             'fodf_encoder_ckpt': self.fodf_encoder_ckpt})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = CrossQ(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.big_neighborhood,
            self.cross_q_hparams,
            self.rng,
            device)
        return alg


def add_cross_q_auto_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Initial temperature parameter')
    parser.add_argument('--batch_size', default=2**12, type=int,
                        help='How many tuples to sample from the replay '
                        'buffer.')
    parser.add_argument('--replay_size', default=1e6, type=int,
                        help='How many tuples to store in the replay buffer.')
    parser.add_argument('--big_neighborhood', action='store_true',
                        help='Whether to use a bigger neighborhood or just the regular neighborhood without convolutions.')
    parser.add_argument('--fodf_encoder_ckpt', type=str, default=None,
                        help='Path to the encoder checkpoint to use for FODF input.')


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    add_training_args(parser)
    add_cross_q_auto_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    offline = args.comet_offline_dir is not None

    # Create comet-ml experiment
    if offline:
        experiment = CometOfflineExperiment(project_name=args.experiment,
                                    workspace=args.workspace, parse_args=False,
                                    auto_metric_logging=False,
                                    disabled=not args.use_comet,
                                    offline_directory=args.comet_offline_dir)
    else:
        experiment = CometExperiment(project_name=args.experiment,
                                    workspace=args.workspace, parse_args=False,
                                    auto_metric_logging=False,
                                    disabled=not args.use_comet)

    experiment.set_name(args.id)

    # Create and run experiment
    sac_auto_experiment = CrossQFineTrackTraining(
        # Dataset params
        vars(args),
        experiment
    )
    sac_auto_experiment.run()


if __name__ == '__main__':
    main()
