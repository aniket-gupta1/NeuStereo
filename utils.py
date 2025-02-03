import os
import subprocess

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
import torch.nn.functional as F
import argparse
import logging
from datetime import datetime
import coloredlogs

import yaml

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config

class DebugFileHandler(logging.FileHandler):
    """File handler that logs only debug messages"""
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        if not record.levelno == logging.DEBUG:
            return
        super().emit(record)

def prepare_logger(opt: argparse.Namespace, log_path: str = None):
    """Creates logging directory, and installs colorlogs

    Args:
        opt: Program arguments, should include --dev and --logdir flag.
             See get_parent_parser()
        log_path: Logging path (optional). This serves to overwrite the settings in
                 argparse namespace

    Returns:
        logger (logging.Logger)
        log_path (str): Logging directory
    """

    if log_path is None:
        if opt.dev:
            log_path = '../logdev'
            shutil.rmtree(log_path, ignore_errors=True)
        else:
            datetime_str = datetime.now().strftime('%y%m%d_%H%M%S')
            if opt.exp_name is not None:
                log_path = os.path.join(opt.logdir, datetime_str + '_' + opt.exp_name)
            else:
                log_path = os.path.join(opt.logdir, datetime_str)
    os.makedirs(log_path, exist_ok=True)

    fmt = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    datefmt = '%m/%d %H:%M:%S'

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # Log to output stream
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(coloredlogs.ColoredFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(stream_handler)

    # Log to file also
    log_formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler = logging.FileHandler(f'{log_path}/log.txt')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Log debug messages into another file to avoid cluttering the standard log file
    log_formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler = DebugFileHandler(f'{log_path}/debug_logs.txt')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger, log_path

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    print('SLURM_PROCID', proc_id)
    print('SLURM_LOCALID', int(os.environ['SLURM_LOCALID']))
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print