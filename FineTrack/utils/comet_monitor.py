import numpy as np

from os.path import join as pjoin
from comet_ml import Experiment

from FineTrack.utils.logging import get_logger

LOGGER = get_logger(__name__)

def _get_prefix_with_delim(prefix):
    delims = ['_', '-', '/']

    if prefix[-1] not in delims:
        out_prefix = f"{prefix}/"
    elif out_prefix is None:
        out_prefix = ""
    else:
        out_prefix = prefix

    return out_prefix


class CometMonitor():
    """ Wrapper class to track information using Comet.ml
    """

    def __init__(
        self,
        experiment: Experiment,
        experiment_path: str,
        prefix: str,
        render: bool = False,
        use_comet: bool = False
    ):
        """
        Parameters:
        -----------
            experiment: str
                Name of experiment. Will contain many interations
                of experiment based on different parameters
            experiment_path: str
                Experiment path used to fetch images or other stuff
            prefix: str
                Prefix for metrics
            use_comet: bool
                Whether to actually use comet or not. Useful when
                Comet access is limited
        """
        # IMPORTANT
        # This presumes that your API key is in your home folder or at
        # the project root.
        self.experiment_path = experiment_path
        self.e = experiment
        self.prefix = _get_prefix_with_delim(prefix)
        self.render = render
        self.use_comet = use_comet

    def log_parameters(self, hyperparameters: dict):
        if not self.use_comet:
            return

        self.e.log_parameters(hyperparameters)

    def update(
        self,
        reward_monitor,
        len_monitor,
        vc_monitor=None,
        ic_monitor=None,
        nc_monitor=None,
        vb_monitor=None,
        ib_monitor=None,
        ol_monitor=None,
        i_episode=0
    ):
        if not self.use_comet:
            return

        reward_x, reward_y = zip(*reward_monitor.epochs)
        len_x, len_y = zip(*len_monitor.epochs)

        self.e.log_metrics(
            {
                self.prefix + "Reward": reward_y[-1],
                self.prefix + "Length": len_y[-1],
            },
            step=i_episode
        )

        if vc_monitor is not None and len(vc_monitor) > 0:
            vc_x, vc_y = zip(*vc_monitor.epochs)
            nc_x, nc_y = zip(*nc_monitor.epochs)
            ic_x, ic_y = zip(*ic_monitor.epochs)
            vb_x, vb_y = zip(*vb_monitor.epochs)
            ib_x, ib_y = zip(*ib_monitor.epochs)
            ol_x, ol_y = zip(*ol_monitor.epochs)

            self.e.log_metrics(
                {
                    self.prefix + "VC": vc_y[-1],
                    self.prefix + "NC": nc_y[-1],
                    self.prefix + "IC": ic_y[-1],
                    self.prefix + "VB": vb_y[-1],
                    self.prefix + "IB": ib_y[-1],
                    self.prefix + "OL": ol_y[-1],
                },
                step=i_episode
            )

        if self.render:
            self.e.log_image(
                pjoin(self.experiment_path, 'render',
                      '{}.png'.format(i_episode)),
                step=i_episode)

    def log_losses(self, loss_dict, i):
        if not self.use_comet:
            return

        for k, v in loss_dict.items():
            if type(v) is np.ndarray:
                self.e.log_histogram_3d(v, name=self.prefix + k, step=i)
            else:
                self.e.log_metric(self.prefix + k, v, step=i)

    def update_train(
        self,
        monitor,
        i_episode,
    ):
        if not self.use_comet:
            return

        x, y = zip(*monitor.epochs)

        self.e.log_metrics(
            {
                self.prefix + monitor.name: y[-1],

            },
            step=i_episode
        )


class OracleMonitor(object):

    def __init__(
        self,
        experiment: Experiment,
        use_comet: bool = False,
        metrics_prefix: str = None
    ):
        self.experiment = experiment

        self.metrics_prefix = _get_prefix_with_delim(metrics_prefix) if metrics_prefix else None

        self.use_comet = use_comet
        if not self.use_comet:
            LOGGER.warning(
                "Comet is not being used. No metrics will be logged for the "
                "Oracle training.")

    def log_parameters(self, hyperparameters: dict):
        if not self.use_comet:
            return
        
        if self.metrics_prefix:
            prefix = self.metrics_prefix
        else:
            prefix = None
        
        self.experiment.log_parameters(hyperparameters, prefix=prefix)

    def log_metrics(self, metrics_dict, step: int, epoch: int):
        if not self.use_comet:
            return

        for k, v in metrics_dict.items():
            assert isinstance(v, (int, float, np.int64, np.float64,
                              np.float32, np.int32)), "Metrics must be numerical."
            
            if self.metrics_prefix:
                k = f"{self.metrics_prefix}{k}"

            self.experiment.log_metric(k, v, step=step, epoch=epoch)

