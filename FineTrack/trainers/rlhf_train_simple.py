import os
import argparse
import tempfile
import comet_ml
import json
import shutil
from pathlib import Path

from comet_ml import Experiment as CometExperiment

from FineTrack.utils.logging import get_logger
from FineTrack.trainers.cross_q_train import CrossQFineTrackTraining
from FineTrack.trainers.sac_auto_train import FineTrackTraining, add_sac_auto_args, SACAutoFineTrackTraining
from FineTrack.trainers.tractoraclenet_train import add_oracle_train_args
from FineTrack.trainers.train import add_training_args
from FineTrack.utils.logging import setup_logging, add_logging_args
from FineTrack.algorithms.rl import RLAlgorithm
from FineTrack.algorithms.sac_auto import SACAuto, SACAutoHParams
from FineTrack.algorithms.cross_q import CrossQ, CrossQHParams
from FineTrack.environments.env import BaseEnv
from FineTrack.tracking.tracker import Tracker
from FineTrack.filterers.tractometer_filterer import TractometerFilterer
from FineTrack.oracles.oracle import OracleSingleton
from FineTrack.trainers.oracle.oracle_trainer import OracleTrainer
from FineTrack.trainers.oracle.data_module import StreamlineDataModule
from FineTrack.trainers.oracle.streamline_dataset_manager import StreamlineDatasetManager
from FineTrack.utils.torch_utils import assert_accelerator
from FineTrack.utils.utils import prettier_metrics, prettier_dict
from FineTrack.filterers.streamlines_sampler import StreamlinesSampler
from FineTrack.utils.hooks import RlHookEvent
from tqdm import tqdm
from dataclasses import dataclass
assert_accelerator()

LOGGER = get_logger(__name__)

# TODO:
# Inheriting directly from the CrossQHParams isn't the best way to do it.
# Ideally, the config file would be split between the general experiment
# parameters, the agent parameters, the oracle parameters and the rlhf parameters.
@dataclass
class RlhfHParams(CrossQHParams):
    pretrain_max_ep: int
    agent_checkpoint: str

    oracle_lr: float
    oracle_train_steps: int
    agent_train_steps: int
    num_workers: int
    rlhf_inter_npv: int
    disable_oracle_training: bool
    batch_size: int
    oracle_batch_size: int
    grad_accumulation_steps: int
    nb_new_streamlines_per_iter: int
    max_dataset_size: int
    warmup_agent_steps: int

    dataset_to_augment: str = None

    def __post_init__(self):
        assert self.pretrain_max_ep is not None \
            or (self.agent_checkpoint_dir is not None \
                or self.agent_checkpoint is not None), \
            "Either pretrain_max_ep or (agent_checkpoint | agent_checkpoint_dir) must be provided for RLHF training."
        
        if self.agent_checkpoint:
            assert os.path.isfile(
                self.agent_checkpoint), "Agent checkpoint must be an checkpoint file."

class RlhfTraining(FineTrackTraining):

    def __init__(
        self,
        config: dict,
        trainer_cls: FineTrackTraining,
        comet_experiment: CometExperiment = None
    ):
        # Only load the parameters from the parent instead of calling
        # the full constructor twice. (As we call it for the agent_trainer
        # below).
        self.init_hyperparameters(config)

        # General RLHF parameters.
        self.ref_model_dir = os.path.join(self.hp.experiment_path, "ref_model")
        self.model_saving_dirs.append(self.ref_model_dir)
        if not os.path.exists(self.ref_model_dir):
            os.makedirs(self.ref_model_dir)

        self.oracle_training_dir = os.path.join(self.hp.experiment_path, "oracle")
        if not os.path.exists(self.oracle_training_dir):
            os.makedirs(self.oracle_training_dir)

        if self.hp.disable_oracle_training:
            LOGGER.warning("Oracle training is disabled. The dataset will "
                           "be augmented to evaluate the oracles during the "
                           "agent's training.")

        ################################################
        # Start by initializing the agent trainer.     #
        if comet_experiment is None:
            comet_experiment = CometExperiment(project_name=self.hp.experiment,
                                               workspace=self.hp.workspace, parse_args=False,
                                               auto_metric_logging=False,
                                               disabled=not self.hp.use_comet)

        comet_experiment.set_name(self.hp.experiment_id)

        self.agent_trainer: FineTrackTraining = trainer_cls(config, comet_experiment)
        _ = self.agent_trainer.setup_environment_and_info() # TODO: Remove this?
        
        # Replace the get_alg method by the one from the agent trainer.
        # This way, if we have a CrossQ trainer, we have the CrossQ alg.
        self.get_alg = self.agent_trainer.get_alg

        # Since backuping is implemented in FineTrackTraining, we disable
        # it to avoid backuping the same files twice to control the backuping
        # process from this class.
        self.agent_trainer.backuper.disable()

        ################################################
        # Continue by initializing the oracle trainer. #
        self.dataset_manager = StreamlineDatasetManager(saving_path=self.oracle_training_dir,
                                                        dataset_to_augment_path=self.hp.dataset_to_augment,
                                                        max_dataset_size=self.hp.max_dataset_size)

        # Note: for the two oracle trainers, we disable the automatic checkpointing
        # because we will want to save the checkpoints only when we improve the 
        # total agent. We manually checkpoint those oracles instead.
        self.oracle_reward_trainer = OracleTrainer(
            comet_experiment,
            self.oracle_training_dir,
            self.hp.oracle_train_steps,
            enable_auto_checkpointing=False,
            checkpoint_prefix='reward',
            val_interval=1,
            device=self.device,
            grad_accumulation_steps=self.hp.grad_accumulation_steps,
            metrics_prefix='reward'
        )

        self.oracle_crit_trainer = OracleTrainer(
            comet_experiment,
            self.oracle_training_dir,
            self.hp.oracle_train_steps,
            enable_auto_checkpointing=False,
            checkpoint_prefix='crit',
            val_interval=1,
            device=self.device,
            grad_accumulation_steps=self.hp.grad_accumulation_steps,
            metrics_prefix='crit'
        )

        # Register hooks on best VC reached to save the oracles that
        # contributed to reach that level of VC.
        def _save_oracles_on_best_vc():
            self.oracle_crit_trainer.save_model_checkpoint(is_best=True)
            self.oracle_reward_trainer.save_model_checkpoint(is_best=True)

        self.agent_trainer._hooks_manager.register_hook(
            RlHookEvent.ON_RL_BEST_VC,
            _save_oracles_on_best_vc
        )

    @property
    def hparams_class(self):
        return RlhfHParams

    def setup_logging(self):
        """ Override the setup_logging method to avoid creating a new experiment. """
        self.save_hyperparameters()

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        valid_env: BaseEnv,
        max_ep: int = 10,
        **kwargs
    ):
        """ Train the RL algorithm for N epochs. An epoch here corresponds to
        running tracking on the training set until all streamlines are done.
        This loop should be algorithm-agnostic. Between epochs, report stats
        so they can be monitored during training

        Parameters:
        -----------
            alg: RLAlgorithm
                The RL algorithm, either TD3, PPO or any others
            env: BaseEnv
                The tracking environment
            valid_env: BaseEnv
                The validation tracking environment (forward).
            """
        current_ep = 0

        ################################################
        # Setup agent trainer
        # (needed since we don't call the run method)
        ################################################
        self.agent_trainer.setup_logging()

        if self.hp.pretrain_max_ep is not None \
            and self.hp.pretrain_max_ep > 0:
            self.agent_trainer.rl_train(alg,
                                        env,
                                        valid_env,
                                        max_ep=self.pretrain_max_ep,
                                        starting_ep=0,
                                        save_model_dir=self.ref_model_dir)
            current_ep += self.pretrain_max_ep
        elif self.hp.agent_checkpoint is not None:
            # The agent is already pretrained, just need to fine-tune it.
            LOGGER.info(
                "Skipping pretraining procedure: loading agent from checkpoint...")
            alg.load_checkpoint(self.hp.agent_checkpoint)
            
            # Instead of having to pack and serialize the model again,
            # as this takes time, just copy the file.
            # This aims to do the following:
            # self.save_model(alg, save_model_dir=self.ref_model_dir)
            #
            # We keep a copy of the initial model state just as a reference.
            # This has no real use in the training process.
            ckpt_path = Path(self.ref_model_dir) / "init_model_state.ckpt"
            shutil.copyfile(self.agent_checkpoint, ckpt_path)

            LOGGER.info("Done.")

        self.agent_trainer.comet_monitor.e.add_tag(
            "RLHF-start-ep-{}".format(current_ep))

        ################################################
        # Setup oracle training
        ################################################

        # Load reward oracle
        self.oracle_reward = OracleSingleton(self.hp.oracle_reward_checkpoint,
                                      device=self.device,
                                      batch_size=self.hp.oracle_batch_size,
                                      lr=self.hp.oracle_lr)
        self.oracle_reward_trainer.setup_model_training(self.oracle_reward.model)

        # Load stopping criterion oracle
        self.oracle_crit = OracleSingleton(self.hp.oracle_crit_checkpoint,
                                           device=self.device,
                                           batch_size=self.hp.oracle_batch_size,
                                           lr=self.hp.oracle_lr)
        self.oracle_crit_trainer.setup_model_training(self.oracle_crit.model)

        ################################################
        # Setup environment
        ################################################
        self.tracker_env = self.get_rlhf_env(npv=self.hp.rlhf_inter_npv)
        self.tracker = Tracker(
            alg, self.hp.n_actor, prob=1.0, compress=0.0)

        # Setup filterers which will be used to filter tractograms
        # for the RLHF pipeline.
        sampler = StreamlinesSampler()
        self.filterers = [
            TractometerFilterer(self.hp.scoring_data, self.hp.tractometer_reference,
                                dilate_endpoints=self.hp.tractometer_dilate,
                                sampler=sampler)
        ]

        do_warmup = self.hp.warmup_agent_steps and current_ep < self.hp.warmup_agent_steps - 1

        ################################################
        # RLHF loop to fine-tune the oracle to the RL
        # agent and vice-versa.
        ################################################
        i = 0
        while i < max_ep: 
            self.start_finetuning_epoch(i, do_warmup)

            if not do_warmup:
                ################################################
                # Add new streamlines to the dataset
                ################################################
                self._add_streamlines_to_dataset(i)

                ################################################
                # Train the Oracles
                ################################################
                if not self.disable_oracle_training:
                    self.train_reward()
                    self.train_stopping_criterion()

            ################################################
            # Train the RL agent
            ################################################
            agent_nb_steps = self.hp.agent_train_steps if not do_warmup else self.hp.warmup_agent_steps
            if do_warmup:
                LOGGER.info(
                    "Warming up agent for {} steps.".format(agent_nb_steps))

            self.agent_trainer.rl_train(
                alg,
                env,
                valid_env,
                max_ep=agent_nb_steps,
                starting_ep=current_ep,
                save_model_dir=self.model_dir,
                test_before_training=do_warmup or i == 0)

            self.end_finetuning_epoch(i, do_warmup)

            if do_warmup:
                current_ep += self.hp.warmup_agent_steps
            else:
                # Backup the model after each loop of the RLHF loop.
                # This is very time consuming.
                self.backuper.backup(step=i) 

                current_ep += self.hp.agent_train_steps
                i += 1
            do_warmup = False

    def _add_streamlines_to_dataset(self, iter_num: int):
        """
        Add new streamlines to the dataset from generated tractograms.
        """
        total_added = 0
        with tqdm(total=self.hp.nb_new_streamlines_per_iter,
                        desc="Adding new streamlines to the dataset",
                        mininterval=5.0) as sub_pbar:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Those will hold the streamlines we are collecting
                # to add to the dataset once we have enough.
                sft_valid = None
                sft_invalid = None

                while total_added < self.hp.nb_new_streamlines_per_iter:
                    # Generate a tractogram
                    tractograms_path = os.path.join(tmpdir, "tractograms")
                    if not os.path.exists(tractograms_path):
                        os.makedirs(tractograms_path)
                    LOGGER.info(
                        "Generating tractograms for RLHF training...")
                    tractograms = self.generate_and_save_tractograms(
                        self.tracker, self.tracker_env, tractograms_path)

                    # Filter the tractogram
                    filtered_path = os.path.join(tmpdir, "filtered")
                    if not os.path.exists(filtered_path):
                        os.makedirs(filtered_path)

                    LOGGER.info(
                        "Filtering tractograms for RLHF training...")
                    # Need to filter for each filterer and keep the same order.
                    filtered_tractograms = self.filter_tractograms(
                        tractograms, filtered_path)
                    
                    # Merge the valid and invalid tractograms
                    for valid, invalid in filtered_tractograms:
                        if sft_valid is None:
                            sft_valid = valid
                            sft_invalid = invalid
                        else:
                            sft_valid += valid
                            sft_invalid += invalid

                        nb_new_streamlines = len(valid) + len(invalid)

                    total_added += nb_new_streamlines
                    sub_pbar.update(nb_new_streamlines)

                LOGGER.info(
                    "Adding filtered tractograms to the dataset...")
                self.dataset_manager.add_tractograms_to_dataset(
                    [(sft_valid, sft_invalid)])
            
        # Print dataset stats
        data_stats = self.dataset_manager.fetch_dataset_stats()
        LOGGER.info(
            prettier_dict(data_stats, title="Dataset stats (iter {})".format(
                iter_num)))

    def train_reward(self):
        """
        Train the reward model using the dataset file.
        This reward model should have been trained on full streamlines, which
        means that dense=False and partial=False.
        """
        print(">>> Training reward model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.hp.oracle_batch_size,
                                  num_workers=self.hp.num_workers,
                                  nb_points=self.oracle_reward.nb_points)
        

        dm.setup('test', dense=False, partial=False)
        metrics_before = self.oracle_reward_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_before, title="Test metrics before fine-tuning"))

        dm.setup('fit', dense=False, partial=False)
        self.oracle_reward_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())
        
        # Auto-checkpointing is disabled, we need to save them manually
        self.oracle_reward_trainer.save_model_checkpoint()

        metrics_after = self.oracle_reward_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_after, title="Test metrics after fine-tuning"))
        print(">>> Finished training reward model step <<<")

    def train_stopping_criterion(self):
        """
        Train the stopping criterion oracle model using the dataset file.
        This stopping criterion model should have been trained on cut
        streamlines, which means that dense=True and partial=False.
        """
        print(">>> Training stopping criterion model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.hp.oracle_batch_size,
                                  num_workers=self.num_workers,
                                  nb_points=self.oracle_crit.nb_points)
        
        # Test the performance of the actual model BEFORE fine-tuning.
        # TO REVISE:
        # To get an accuracy plot, we test the stopping criterion on fully
        # tracked streamlines even though it's supposed to predict on partial
        # streamlines.
        dm.setup('test', dense=False, partial=False)
        metrics_before = self.oracle_crit_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_before, title="Test metrics before fine-tuning"))

        dm.setup('fit', dense=True, partial=True)
        self.oracle_crit_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())
        
        # Auto-checkpointing is disabled, we need to save manually
        self.oracle_crit_trainer.save_model_checkpoint()
        
        # Test the performance of the actual model AFTER fine-tuning.
        metrics_after = self.oracle_crit_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_after, title="Test metrics after fine-tuning"))
        print(">>> Finished stopping criterion model training <<<")

    def generate_and_save_tractograms(self, tracker: Tracker, env: BaseEnv, save_dir: str):
        """
        """
        # TODO: Change to only track().
        tractogram, _ = tracker.track_and_validate(self.tracker_env)
        filename = self.save_rasmm_tractogram(
            tractogram,
            env.subject_id,
            env.affine_vox2rasmm,
            env.reference,
            save_dir,
            extension='tck')
        return [filename]

    def filter_tractograms(self, tractograms: str, out_dir: str):
        """
        """
        filterer = self.filterers[0]

        filtered_tractograms = []
        for tractogram in tractograms:
            # TODO: Implement for more than one filterer
            valid_tractogram, invalid_tractogram = filterer(
                tractogram, out_dir, scored_extension="trk")
            filtered_tractograms.append((valid_tractogram, invalid_tractogram))

        return filtered_tractograms

    def save_hyperparameters(self):
        super().save_hyperparameters(filename='rlhf_hyperparameters.json')

    def start_finetuning_epoch(self, epoch: int, warmup: bool = False):
        if warmup:
            print("==================================================")    
            print("=========== Starting WARMUP of {} steps =========".format(self.hp.warmup_agent_steps))
        else:
            print("==================================================")
            print("======= Starting RLHF finetuning epoch {}/{} =======".format(epoch+1, self.hp.max_ep))

    def end_finetuning_epoch(self, epoch: int, warmup: bool = False):
        if warmup:
            print("=========== Finished WARMUP of {} steps =========".format(self.hp.warmup_agent_steps))
            print("==================================================")
        else:
            print("======= Finished RLHF finetuning epoch {}/{} =======".format(epoch+1, self.hp.max_ep))
            print("==================================================")


def add_rlhf_training_args(parser: argparse.ArgumentParser):
    rlhf_group = parser.add_argument_group("RLHF Training Arguments")
    rlhf_group.add_argument('--alg', type=str, required=True,
                            help='The algorithm to use for training the agent.\n'
                            'Possible values are: SACAuto, PPO.')
    rlhf_group.add_argument('--num_workers', type=int, default=10,
                            help='Number of workers to use for data loading.')
    rlhf_group.add_argument("--rlhf_inter_npv", type=int, default=None,
                            help="Number of seeds to use when generating intermediate tractograms\n"
                            "for the RLHF training pipeline. If None, the general npv will be used.")
    rlhf_group.add_argument("--nb_new_streamlines_per_iter", type=int, default=500000,
                            help="Number of new streamlines to add to the dataset at each iteration.")
    rlhf_group.add_argument("--max_dataset_size", type=int, default=5000000,
                            help="Maximum number of streamlines to keep in the dataset.")
    rlhf_group.add_argument("--warmup_agent_steps", type=int,
                            help="Minimum number of steps to warm up the agent before starting the training of the oracle")

    # The following arguments are usually used for PPO, but we are also testing it for other algorithms.
    parser.add_argument('--adaptive_kl', action='store_true',
                        help='This flag enables the adaptive kl penalty.\n'
                        'Otherwise, the penalty coefficient is fixed.')
    parser.add_argument('--kl_penalty_coeff', default=0.02, type=float,
                        help='Initial KL penalty coefficient.')
    parser.add_argument('--kl_target', default=0.005, type=float,
                        help='KL target value.')
    parser.add_argument('--kl_horizon', default=1000, type=int,
                        help='KL penalty horizon.')

    # Agent training RLHF arguments
    agent_group = rlhf_group.add_argument_group("Agent Training Arguments")
    agent_group.add_argument('--agent_train_steps', type=int, required=True,
                             help='Number of steps to fine-tune the agent during RLHF training.')

    agent_init_group = rlhf_group.add_mutually_exclusive_group(required=True)
    agent_init_group.add_argument('--pretrain_max_ep', type=int,
                                  help='Number of epochs for pretraining the RL agent.\n'
                                  'This is done before starting the RLHF pretraining procedure.')
    agent_checkpoint_group = agent_init_group.add_mutually_exclusive_group()
    agent_checkpoint_group.add_argument('--agent_checkpoint', type=str,
                                        help='Path to the agent checkpoint FILE to load.')

    # Oracle training RLHF arguments
    oracle_group = rlhf_group.add_argument_group("Oracle Training Arguments")
    oracle_group.add_argument('--oracle_lr', type=float,
                              help='Learning rate to use for training the oracle.\n'
                              'If not set, the lr stored in the checkpoint will be used.')
    oracle_group.add_argument('--oracle_train_steps', type=int, required=True,
                              help='Number of steps to fine-tune the oracle during RLHF training.')
    oracle_group.add_argument('--oracle_batch_size', type=int, default=2816,
                              help='Batch size to use for training the oracle.')
    oracle_group.add_argument("--dataset_to_augment", type=str, help="Path to the dataset to augment.\n"
                              "If this is not set, the dataset will be created from scratch entirely by the\n"
                              "current learning agent.")
    oracle_group.add_argument("--disable_oracle_training", action="store_true",
                              help="Disable oracle training during RLHF training.\n")
    return parser


def get_trainer_cls_and_args(alg_name: str):
    trainer_map = {
        'SACAuto': SACAutoFineTrackTraining,
        'CrossQ': CrossQFineTrackTraining,
    }

    if alg_name not in trainer_map:
        raise ValueError(f'Invalid algorithm name: {alg_name}')

    return trainer_map[alg_name]


def get_algorithm_cls(alg_name: str):
    algorithm_map = {
        'SACAuto': SACAuto,
        'CrossQ': CrossQ,
    }

    if alg_name not in algorithm_map:
        raise ValueError(f'Invalid algorithm name: {alg_name}')

    return algorithm_map[alg_name]


def parse_args():
    """ Train an agent whilst training oracles in the loop. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_training_args(parser)
    add_sac_auto_args(parser)
    add_rlhf_training_args(parser)
    add_oracle_train_args(parser)
    add_logging_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_args()
    setup_logging(args)

    trainer_cls = get_trainer_cls_and_args(args.alg)

    # Create and run the experiment
    rlhf_experiment = RlhfTraining(
        vars(args),
        trainer_cls
    )
    rlhf_experiment.run()


if __name__ == "__main__":
    main()
