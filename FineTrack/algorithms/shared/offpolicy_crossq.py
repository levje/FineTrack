import torch
import torch.nn as nn
from FineTrack.algorithms.shared.offpolicy import SACActorCritic, MaxEntropyActor
from FineTrack.algorithms.shared.utils import (
    format_widths, make_fc_network, make_conv_network)
from FineTrack.environments.conv_state import ConvState, ConvStateShape
from FineTrack.algorithms.shared.batch_renorm import BatchRenorm1d, BatchRenorm3d

class CrossQCritic(nn.Module):
    def __init__(
        self,
        state_dim: ConvStateShape,
        action_dim: int,
        hidden_dims: str,
        critic_size_factor=1,
        batch_renorm=False
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(CrossQCritic, self).__init__()

        conv_state_shape = (state_dim.nb_sh_coefs, state_dim.depth,
                    state_dim.height, state_dim.width)
        conv_output_size = 256

        if batch_renorm:
            self.norm_func_3d = BatchRenorm3d
            self.norm_func_1d = BatchRenorm1d
        else:
            self.norm_func_3d = nn.BatchNorm3d
            self.norm_func_1d = nn.BatchNorm1d
        batch_norm_kwargs = {"momentum": 0.01}

        self.q_neighbor_encoder = make_conv_network(
            input_size=conv_state_shape, output_size=256,
            norm_layer_1d=self.norm_func_1d, norm_layer_3d=self.norm_func_3d,
            norm_layer_kwargs=batch_norm_kwargs
            )
        
        full_fc_state_dim = conv_output_size + state_dim.prev_dirs
        self.hidden_layers = format_widths(
            hidden_dims) * critic_size_factor

        self.activation = nn.ReLU
        
        # From CrossQ implementation, we need to introduce batch renormalization
        # in order to avoid utilizing target networks.
        input_size = full_fc_state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_size, self.hidden_layers[0]),
            nn.ReLU(),
            self.norm_func_1d(self.hidden_layers[0], **batch_norm_kwargs),

            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
            nn.ReLU(),
            self.norm_func_1d(self.hidden_layers[1], **batch_norm_kwargs),

            nn.Linear(self.hidden_layers[1], self.hidden_layers[2]),
            nn.ReLU(),
            self.norm_func_1d(self.hidden_layers[2], **batch_norm_kwargs),

            nn.Linear(self.hidden_layers[2], 1)
        )

    def forward(self, state: ConvState, action, next_state=None, next_action=None) -> torch.Tensor:

        if next_state is None or next_action is None:
            assert next_state is None and next_action is None, \
                "Either both next_state and next_action must be provided or neither."

        if next_state is not None and next_action is not None:    
            # Predict on both state-action pairs at the same time.
            #encoder_states_input = torch.cat([state.conv_state, next_state.conv_state]) # concat batch dimension
            #encoded_neighborhood_1 = self.q_neighbor_encoder(encoder_states_input)

            all_prev_dirs = torch.cat([state.prev_dirs, next_state.prev_dirs]) # concat batch-wise
            
            # add all prev dirs to the input
            #q1_states_input = torch.cat([encoded_neighborhood_1, all_prev_dirs], -1) # concat feature-wise
            q1_actions_input = torch.cat([action, next_action]) # concat batch-wise

            q1_input = torch.cat([all_prev_dirs, q1_actions_input], -1) # Add actions as features
            pred = self.q1(q1_input).squeeze(-1)

            q, next_q = torch.tensor_split(pred, 2, dim=0)
            return q, next_q
        else:
            # Predict on single state-action pair
            encoded_neighborhood_1 = self.q_neighbor_encoder(state.conv_state)
            q1_states_input = torch.cat([encoded_neighborhood_1, state.prev_dirs], -1)
            q1_input = torch.cat([q1_states_input, action], -1)
            pred = self.q1(q1_input).squeeze(-1)
            return pred

class CrossQDoubleCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q1 = CrossQCritic(*args, **kwargs)
        self.q2 = CrossQCritic(*args, **kwargs)
    
    def forward(self, state, action, next_state=None, next_action=None) -> torch.Tensor:
        if next_state is None or next_action is None:
            assert next_state is None and next_action is None, \
                "Either both next_state and next_action must be provided or neither."
            
        args = (state, action, next_state, next_action)

        if next_state is not None and next_action is not None:
            q1, next_q1 = self.q1(*args)
            q2, next_q2 = self.q2(*args)
            return q1, next_q1, q2, next_q2
        else:
            q1 = self.q1(*args)
            q2 = self.q2(*args)
            return q1, q2

class CrossQActorCritic(SACActorCritic):
    def __init__(
            self,
            state_dim: ConvStateShape,
            action_dim,
            hidden_dims,
            device
    ):
        self.device = device
        self.actor = MaxEntropyActor(
            state_dim, action_dim, hidden_dims
        ).to(device)

        self.critic = CrossQDoubleCritic(
            state_dim, action_dim, hidden_dims
        ).to(device)