import copy
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

from FineTrack.algorithms.sac_auto import SACAuto
from FineTrack.algorithms.shared.offpolicy_crossq import CrossQActorCritic
from FineTrack.algorithms.shared.replay import OffPolicyReplayBuffer, OffPolicyLazyReplayBuffer
from FineTrack.utils.torch_utils import get_device, gradients_norm
from FineTrack.environments.conv_state import ConvStateShape

LOG_STD_MAX = 2
LOG_STD_MIN = -20

@dataclass
class CrossQHParams:
    lr: float = 3e-4
    gamma: float = 0.99
    n_actors: int = 4096

    alpha: float = 0.2
    batch_size: int = 2**12
    replay_size: int = 1e6

    adaptive_kl: bool = False
    kl_penalty_coeff: float = 0.02
    kl_target: float = 0.005
    kl_horizon: int = 1000

class CrossQ(SACAuto):
    """
    The sample-gathering and training algorithm.
    Based on

        Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ...
        & Levine, S. (2018). Soft actor-critic algorithms and applications.
        arXiv preprint arXiv:1812.05905.

    Implementation is based on Spinning Up's and rlkit

    See https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py  # noqa E501

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_shape: ConvStateShape,
        action_size: int,
        hidden_dims: int,
        hparams: CrossQHParams = CrossQHParams(),
        rng: np.random.RandomState = None,
        device: torch.device = get_device,
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_dims: str
            Dimensions of the hidden layers
        lr: float
            Learning rate for the optimizer(s)
        gamma: float
            Discount factor
        alpha: float
            Initial entropy coefficient (temperature).
        n_actors: int
            Number of actors to use
        batch_size: int
            Batch size to sample the memory
        replay_size: int
            Size of the replay buffer
        rng: np.random.RandomState
            Random number generator
        device: torch.device
            Device to use for the algorithm. Should be either "cuda:0"
        """
        self.hparams = hparams
        
        self.batch_size = hparams.batch_size
        self.gamma = hparams.gamma
        self.alpha = hparams.alpha
        self.n_actors = hparams.n_actors
        self.replay_size = hparams.replay_size

        self.max_action = 1.
        self.t = 1
        self.nb_updates_per_sample = 5

        self.action_size = action_size
        self.device = device

        self.rng = rng

        # Initialize main agent
        self.agent = CrossQActorCritic(
            input_shape, action_size, hidden_dims, device,
        )

        # Auto-temperature adjustment
        # SAC automatically adjusts the temperature to maximize entropy and
        # thus exploration, but reduces it over time to converge to a
        # somewhat deterministic policy.
        starting_temperature = np.log(self.hparams.alpha)  # Found empirically
        self.target_entropy = -np.prod(action_size).item()
        self.log_alpha = torch.full(
            (1,), starting_temperature, requires_grad=True, device=device)
        # Optimizer for alpha
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.hparams.lr)

        # Initialize target agent to provide baseline
        self.target_critic = copy.deepcopy(self.agent.critic)

        # SAC requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(), lr=self.hparams.lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(), lr=self.hparams.lr)

        # SAC-specific parameters
        self.max_action = 1.
        self.on_agent = False

        self.start_timesteps = 80000
        self.total_it = 0
        self.tau = 0.005
        self.agent_freq = 1

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_shape, action_size, max_size=self.hparams.replay_size)

        self.rng = rng

    def load_checkpoint(self, checkpoint_file: str):
        """
        Load a checkpoint into the algorithm.

        Parameters
        ----------
        checkpoint: dict
            Dictionary containing the checkpoint to load.
        """
        checkpoint = torch.load(checkpoint_file, weights_only=False)

        self.agent.load_checkpoint(checkpoint['agent'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if checkpoint.get('replay_buffer', None) is not None:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer'])
        if checkpoint.get('log_alpha', None) is not None:
            self.log_alpha = checkpoint['log_alpha']

    def save_checkpoint(self, checkpoint_file: str, **extra_info):
        """
        Save the current state of the algorithm into a checkpoint.

        Parameters
        ----------
        checkpoint_file: str
            File to save the checkpoint into.
        """
        checkpoint = {
            'agent': self.agent.state_dict(as_dict=True),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer.state_dict(),
            'log_alpha': self.log_alpha,
            **extra_info
        }

        torch.save(checkpoint, checkpoint_file)

    def update(
        self,
        batch,
    ) -> Tuple[float, float]:
        """

        SAC Auto improves upon SAC by automatically adjusting the temperature
        parameter alpha. This is done by optimizing the temperature parameter
        alpha to maximize the entropy of the policy. This is done by
        maximizing the following objective:
            J_alpha = E_pi [log pi(a|s) + alpha H(pi(.|s))]
        where H(pi(.|s)) is the entropy of the policy.


        Parameters
        ----------
        batch: Tuple containing the batch of data to train on.

        Returns
        -------
        losses: dict
            Dictionary containing the losses of the algorithm and various
            other metrics.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            batch
        
        ############################
        # UPDATE ALPHA
        ############################

        # Compute \pi_\theta(s_t) and log \pi_\theta(s_t)
        pi, logp_pi = self.agent.act(
            state, probabilistic=1.0)
        # Compute the temperature loss and the temperature
        alpha_loss = -(self.log_alpha * (
            logp_pi + self.target_entropy).detach()).mean()
        alpha = self.log_alpha.exp()

        # Optimize the temperature
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_(self.log_alpha, 0.5)
        self.alpha_optimizer.step()

        ############################
        # UPDATE ACTOR
        ############################

        # Compute the Q values and the minimum Q value
        self.agent.actor.train() # https://www.reddit.com/r/reinforcementlearning/comments/1bj3rln/trying_to_implement_crossq_in_pytorch_does_not/
        self.agent.critic.eval() 
        q1, q2 = self.agent.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized agent loss
        actor_loss = (alpha * logp_pi - q_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        ############################
        # UPDATE CRITIC
        ############################
        self.agent.actor.eval()
        self.agent.critic.train()

        with torch.no_grad():
            # Target actions come from *current* agent
            next_action, logp_next_action = self.agent.act(
                next_state, probabilistic=1.0)

        # Get Q estimates all at once from the critic
        current_q1, next_q1, current_q2, next_q2 = self.agent.critic(
            state, action, next_state, next_action)
        
        with torch.no_grad():
            next_q = torch.min(next_q1, next_q2)  # Double critic
            backup = reward + self.gamma * not_done * (next_q - alpha * logp_next_action)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(current_q1, backup.detach()).mean()
        loss_q2 = F.mse_loss(current_q2, backup.detach()).mean()

        # Total critic loss
        critic_loss = loss_q1 + loss_q2

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.agent.critic.eval()

        # Compute the norm of the gradients to plot.
        alpha_norm = self.log_alpha.grad.norm(2).cpu().detach().numpy()
        critic_norm = gradients_norm(self.agent.critic)
        actor_norm = gradients_norm(self.agent.actor)

        losses = {
            "alpha_norm": alpha_norm,
            "critic_norm": critic_norm,
            "actor_norm": actor_norm,
        }

        return losses
