import numpy as np
import torch

from torch import nn
from collections import defaultdict


def add_item_to_means(means, dic):
    if isinstance(means, defaultdict):
        for k in dic.keys():
            means[k].append(dic[k])
    else:
        means = {k: means[k] + [dic[k]] for k in dic.keys()}
    return means


def add_to_means(means, dic):
    return {k: means[k] + dic[k] for k in dic.keys()}

# TODO: Remove that, it's just to test classic ppo implementation


def old_mean_losses(dic):
    return {k: np.mean(dic[k]) for k in dic.keys()}


def get_mean_item(dic, key):
    if isinstance(dic[key][0], torch.Tensor):
        return np.mean(torch.stack(dic[key]).cpu().numpy())
    return np.mean(dic[key])


def mean_losses(dic):
    new_dict = {}
    for k in dic.keys():
        values = dic[k]
        if isinstance(values, list) and isinstance(values[0], torch.Tensor):
            values = torch.stack(values).cpu().numpy()

        new_dict[k] = np.mean(values, axis=0)
    return new_dict


def add_losses(dic):
    new_dict = {}
    for k in dic.keys():
        values = dic[k]
        if isinstance(values, list) and isinstance(values[0], torch.Tensor):
            values = torch.stack(values).cpu().numpy()

        new_dict[k] = np.sum(values, axis=0)
    return new_dict


def mean_rewards(dic):
    return {k: np.mean(np.asarray(dic[k]), axis=0) for k in dic.keys()}


def harvest_states(i, *args):
    return (a[:, i, ...] for a in args)


def stack_states(full, single):
    if full[0] is not None:
        return (np.vstack((f, s[None, ...]))
                for (f, s) in zip(full, single))
    else:
        return (s[None, :, ...] for s in single)


def format_widths(widths_str):
    return np.asarray([int(i) for i in widths_str.split('-')])


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU,
    last_activation=nn.Identity
):
    layers = [nn.Linear(input_size, widths[0]), activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation()])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size)])
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 2*in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        return out

def make_conv_network(input_size, output_size):

    # input_size = [in_channels, depth, height, width]
    # input_size = [c, d, h, w]
    (c, d, h, w) = input_size
    size_after_conv = 64 * (d) * (h) * (w)
    layers = nn.Sequential(
        # [c, d, h, w]
        nn.Conv3d(c, 16, kernel_size=3, stride=1, padding=1),
        # [16, d, h, w]
        nn.GELU(),
        # nn.MaxPool3d(kernel_size=2, stride=2),
        # [16, d/2, h/2, w/2]

        ResidualBlock(16),

        nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
        # [32, d/2, h/2, w/2]
        nn.GELU(),
        # nn.MaxPool3d(kernel_size=2, stride=2),
        # [32, d/4, h/4, w/4]

        ResidualBlock(32),

        nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
        # [64, d/4, h/4, w/4]
        nn.GELU(),
        # nn.MaxPool3d(kernel_size=2, stride=2),
        # [64, d/8, h/8, w/8]

        ResidualBlock(64),

        nn.Flatten(), # -> 64 * d/8 * h/8 * w/8
        nn.Linear(size_after_conv, 1024),
        nn.GELU(),
        nn.Linear(1024, output_size)
    )

    return layers
