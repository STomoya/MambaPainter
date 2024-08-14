"""Copy & paste-able utilities for writing training codes with PyTorch in a single file.

# URL

https://github.com/STomoya/torchutils.py

# copyright

MIT License

Copyright (c) 2024 Tomoya Sawada

"""

from __future__ import annotations

import datetime
import logging
import math
import os
import random
import re
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, ClassVar, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from packaging.version import parse as parse_version
from torch.cuda.amp import GradScaler
from torch.distributed import ReduceOp
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# This version will be modified until distributed checkpoint becomes stable.
if parse_version(torch.__version__) >= parse_version('2.3.0'):
    import torch.distributed.checkpoint.state_dict as dcpsd
else:
    dcpsd = None

__version__ = '0.0.1'


class _State:
    """Automatically initialize distributed env at first call of dist related funcs."""

    _shared_state: ClassVar[dict] = {}

    def __init__(self, **kwargs):
        self.__dict__ = self._shared_state

        if not self.initialized:
            self.device = None
            self.backend = 'none'

            if torch.cuda.is_available() and int(os.environ.get('LOCAL_RANK', -1)) >= 0:
                self.backend = kwargs.pop('backend', 'nccl')
                dist.init_process_group(self.backend)
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
                self.local_rank = int(os.environ.get('LOCAL_RANK'))
                self.device = torch.device('cuda', self.local_rank)
                torch.cuda.set_device(self.device)
            else:
                self.world_size = 1
                self.rank = self.local_rank = 0
                self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @property
    def initialized(self):
        return self.__dict__ != {}


#######################################################################################################################
### Distributed                                                                                                     ###
#######################################################################################################################


def initialize_distributed(**kwargs):
    """Initialize distributed process group with user specified arguments.

    Calling this function is optional.
    See https://pytorch.org/docs/stable/distributed.html#initialization for the arguments.
    """
    assert _State._shared_state == {}, 'This fuction must be called before any other distributed related functions.'
    _State(**kwargs)


def is_primary() -> bool:
    """Is primary process."""
    return _State().rank == 0


def only_on_primary(func: Callable) -> Callable:
    """Decorate function to run only on primary process."""

    @wraps(func)
    def decorator(*args, **kwargs):
        if is_primary():
            return func(*args, **kwargs)

    return decorator


def is_distributed() -> bool:
    """Is process group initialized."""
    return dist.is_initialized()


def destroy_process_group():
    """destroy process group."""
    if is_distributed():
        dist.destroy_process_group()


def is_torchrun() -> bool:
    """Is called via torchrun command."""
    return dist.is_torchelastic_launched()


def barrier() -> None:
    """Sync processes."""
    if is_distributed():
        dist.barrier()


def wait_for_everyone() -> None:
    """Sync processes."""
    barrier()


def get_backend() -> str:
    """Backend."""
    return _State().backend


def get_device() -> torch.device:
    """Device."""
    return _State().device


def get_rank() -> int:
    """Rank."""
    return _State().rank


def get_world_size() -> int:
    """World size."""
    return _State().world_size


def gather(
    obj: Any, dst: int | None = None, into_tensor: bool = True
) -> torch.Tensor | tuple[torch.Tensor] | tuple[Any]:
    """Gather objects between devices.

    Can be a torch.Tensor or a picklable python object.

    By default tensors are gathered into a single tensor. To gather into a list of tensors,
    set `into_tensor=False`. Python objects are not affected by this argument and are always
    gathered into a list.

    By default the objects are gathered to all devices. You can specify the device to gather
    to by passing a valid process index to the `dst` argument (e.g., 0). If `dst` argument
    is specified, `None` will be returned to all other processes.

    If is not a distributed environment, this function will just return the input `obj`.

    Args:
    ----
        obj (Any): object to gather. Can be a Tensor or picklable python object.
        dst (int, optional): destination device. If not given gathers to all devices. Default: None.
        into_tensor (bool, optional): If True and obj is a Tensor gather into a Tensor instead of a list. Default: True.

    Returns:
    -------
        torch.Tensor | tuple[torch.Tensor] | tuple[Any]: gathered object.

    """
    state = _State()
    if not is_distributed():
        return obj
    elif torch.is_tensor(obj):
        output = [torch.empty_like(obj) for _ in range(state.world_size)]
        if dst is None and into_tensor:
            output = torch.cat(output)
            dist.all_gather_into_tensor(output, obj)
            return output
        elif dst is None:
            dist.all_gather(output, obj)
            return output
        else:
            output = output if state.rank == dst else None
            dist.gather(obj, output, dst)
            return torch.cat(output) if output is not None and into_tensor else output
    else:
        output = [None for _ in range(state.world_size)]
        if dst is None:
            dist.all_gather_object(output, obj)
            return output
        else:
            output = output if state.rank == dst else None
            dist.gather_object(obj, output, dst)
            return output


def reduce(tensor: torch.Tensor, dst: int | None = None, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
    """Reduce a tensor according to the given `ReduceOp` enum.

    In contrast to `gather`, this function does not support python objects. If reducing
    a python number, convert object to a Tensor beforehand.

    By default the objects are reduced to all devices. You can specify the device
    by passing a valid process index to the `dst` argument (e.g., 0). If `dst` argument
    is specified, `None` will be returned to all other processes.

    If is not a distributed environment, this function will just return the input `obj`.

    Args:
    ----
        tensor (torch.Tensor): Tensor to reduce.
        dst (int, optional): destination device. If not given reduced to all device. Default: None.
        op (ReduceOp, optional): reduce option. Default: ReduceOp.SUM.

    Returns:
    -------
        torch.Tensor: reduced tensor.

    """
    state = _State()

    if not is_distributed():
        return tensor

    elif dst is None:
        dist.all_reduce(tensor, op)
        return tensor

    else:
        dist.reduce(tensor, dst, op)
        return tensor if state.rank == dst else None


#######################################################################################################################
### Misc                                                                                                            ###
#######################################################################################################################


# from: https://github.com/google/flax/blob/2387439a6f5c88627754905e6feadac4f33d9800/flax/training/checkpoints.py
UNSIGNED_FLOAT_RE = re.compile(r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')


def natural_sort(iter: list[str], reverse: bool = False) -> list[str]:
    """Sort files by numbers.

    Args:
    ----
        iter (list[str]): An iterable to sort.
        reverse (bool, optional): Reverse sorting. Default: False

    Returns:
    -------
        list[str]: The sorted iterable.

    """

    def maybe_num(s):
        return float(s) if UNSIGNED_FLOAT_RE.match(s) else s

    def split_keys(s):
        return [maybe_num(c) for c in UNSIGNED_FLOAT_RE.split(s)]

    return sorted(iter, key=split_keys, reverse=reverse)


def timezone(offset: int, name: str | None = None) -> datetime.tzinfo:
    """Wrap datetime.timezone, supporting int for offset.

    Args:
    ----
        offset (int): offset.
        name (str, optional): name. Defaults to None.

    Returns:
    -------
        datetime.tzinfo: tzinfo

    """
    return datetime.timezone(datetime.timedelta(hours=offset), name=name)


def get_jst_timezone() -> datetime.tzinfo:
    """Create and return a JST (UCT+9) tzinfo object.

    Returns
    -------
        tzinfo: JST tzinfo

    """
    return timezone(9, 'JST')


def get_now_string(format: str = '%Y%m%d%H%M%S', use_jst: bool = True) -> str:
    """Get datetime.datetime.now() as string.

    Args:
    ----
        format (str, optional): format of the datetime. Default: '%Y%m%d%H%M%S'.
        use_jst (bool, optional): use jst timezone. Default: True.

    Returns:
    -------
        str: datetime.

    """
    return datetime.datetime.now(tz=get_jst_timezone() if use_jst else None).strftime(format)


@only_on_primary
def makedirs0(name: str, exist_ok: bool = False):
    """os.makedirs() only on primary process."""
    os.makedirs(name, exist_ok=exist_ok)


#######################################################################################################################
### Logging                                                                                                         ###
#######################################################################################################################


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    filename: str | None = None,
    mode: str = 'a',
    format: str = '%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s',
    auxiliary_handlers: list | None = None,
) -> logging.Logger:
    """Create logger.

    If filename is given the logs will be saved to this file.

    Args:
    ----
        name (str): name of the logger. identical to logging.getLogger(name) if already called once with the same name.
        level (int): Logging level. Default: logging.DEBUG.
        filename (str | None): filename to where the logs are saved. Default: None
        mode (str): write mode of the file. Default: 'a'
        format (str, optional): logging format.
            Default: '%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s'.
        auxiliary_handlers (list, optional): Other user-defined handlers. Default: None

    Returns:
    -------
        logging.Logger: logger object.

    """
    logger = logging.getLogger(name)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(format)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if auxiliary_handlers:
        for handler in auxiliary_handlers:
            logger.addHandler(handler)

    return logger


#######################################################################################################################
### Dataset                                                                                                         ###
#######################################################################################################################


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    worker_init_fn: Callable | None = None,
    generator: torch.Generator = None,
) -> DataLoader:
    """Create dataloader depending on the environment.

    Args:
    ----
        dataset (Dataset): dataset
        batch_size (int): batch size
        shuffle (bool): Shuffle dataset. Default: True.
        drop_last (bool): Drop last batch. Default: False.
        num_workers (int): Number of workers. Default: 0.
        pin_memory (bool): Pin memory. Default: True.
        worker_init_fn (Callable | None): worker_init_fn. Default: None.
        generator (torch.Generator): RNG. Default: None.

    Returns:
    -------
        DataLoader: created dataloader.

    """
    if is_distributed():
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    return dataloader


def get_dataloader_kwargs() -> tuple[Callable, torch.Generator]:
    """Return objects for `worker_init_fn` and `generator` arguments of DataLoader.

    This function is needed when using randomness in datasets and setting `num_workers` to `>1`.
    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader for details about using random numbers in
    DataLoader class.

    Example:
    -------
        ```python
        from torch.utils.data import DataLoader

        dataset = create_dataset(...)
        repr_kwargs = get_dataloader_kwargs()
        dataloader = DataLoader(dataset, batch_size=32, num_workers=8, **repr_kwargs)
        ```

    """

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    rng = torch.Generator()
    rng.manual_seed(0)

    return dict(worker_init_fn=seed_worker, generator=rng)


#######################################################################################################################
### Reproducibility                                                                                                 ###
#######################################################################################################################


def set_seeds(
    seed: int | None = 3407,
    use_deterministic_algorithms: bool = False,
    warn_only: bool = False,
    cudnn_benchmark: bool = False,
) -> None:
    """Set variables for reproducible training.

    Args:
    ----
        seed (int | None, optional): Random number generator seed. Default: 3407.
        use_deterministic_algorithms (bool, optional): use deterministic algorithms?
            True for reproducibility. Default: False.
        warn_only (bool, optional): Warn instead of an exception when using an module
            without a deterministic implementation. Default: False.
        cudnn_benchmark (bool, optional): cudnn benchmark. Default: False.

    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch.use_deterministic_algorithms(use_deterministic_algorithms, warn_only=warn_only)
    if use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = cudnn_benchmark


@contextmanager
def local_seed_builtin(seed: int, enabled: bool = True):
    """Locally set the seed of builtin random module.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.

    """
    if enabled:
        random_state = random.getstate()
        random.seed(seed)

    yield

    if enabled:
        random.setstate(random_state)


@contextmanager
def local_seed_numpy(seed: int, enabled: bool = True):
    """Locally set the seed of numpy.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.

    """
    if enabled:
        random_state = np.random.get_state()
        np.random.seed(seed)

    yield

    if enabled:
        np.random.set_state(random_state)


@contextmanager
def local_seed_torch(seed: int, enabled: bool = True):
    """Locally set the seed of torch.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.

    """
    if enabled:
        random_state = torch.get_rng_state()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        if torch.cuda.is_available():
            random_state_cuda = torch.cuda.get_rng_state_all()

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    yield

    if enabled:
        torch.set_rng_state(random_state)
        torch.use_deterministic_algorithms(deterministic, warn_only)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_state_cuda)


@contextmanager
def local_seed(seed: int, enabled: bool = True, builtin: bool = True, numpy: bool = True, torch: bool = True):
    """Locally set the seed of builtin random, numpy, and torch.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.
        builtin (bool, optional): Independent flag for builtin random. Ignored when enabled=False. Default: True.
        numpy (bool, optional): Independent flag for numpy. Ignored when enabled=False. Default: True.
        torch (bool, optional): Independent flag for torch. Ignored when enabled=False. Default: True.

    """
    if not enabled:
        builtin = numpy = torch = False
    with local_seed_builtin(seed, builtin), local_seed_numpy(seed, numpy), local_seed_torch(seed, torch):
        yield


#######################################################################################################################
### Module                                                                                                          ###
#######################################################################################################################


def wrap_module(module: nn.Module, strategy: str, compile: bool | str = False) -> tuple[nn.Module, nn.Module]:
    """Wrap/compile module.

    This function only supports the simplest ways to wrap models for distributed settings. This function can also be
    used to compile the model using torch.compile.

    'none' can be passed to the `strategy` argument when you want to compile the model outside of a distributed env.
    But its faster to just call `torch.compile`.

    Args:
    ----
        module (nn.Module): The module to wrap.
        strategy (str): Distributed parallel strategy. One of 'ddp', 'fsdp', and 'none'.

            - 'ddp': `DistributedDataParallel(module, device_ids=[device])`

            - 'fsdp': `FullyShardedDataParallel(module, use_orig_params=is_compile_enabled)`

            - 'none': as-is.

        compile (bool | str): If True, compiles the model using mode='default'. If a string, the string is used as the
            mode option. Default: False.

    Returns:
    -------
        tuple[nn.Module, nn.Module]: Wrapped module and compiled module.

    """
    # str if compile is enabled, False otherwise.
    compile = 'default' if isinstance(compile, bool) and compile else compile

    device = get_device()
    module.to(device)

    if is_distributed():
        strategy = strategy.lower()
        if strategy == 'ddp':
            wrapped = DDP(module, device_ids=[device])
        elif strategy == 'fsdp':
            wrapped = FSDP(module, use_orig_params=isinstance(compile, str))
        elif strategy == 'none':
            wrapped = module
    else:
        wrapped = module

    if compile:
        compiled = torch.compile(wrapped, mode=compile)
    else:
        compiled = wrapped

    return wrapped, compiled


def freeze(module: nn.Module) -> None:
    """Freeze module."""
    module.eval()
    module.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    """Unfreeze module."""
    module.requires_grad_(True)
    module.train()


def update_ema(module: nn.Module, module_ema: nn.Module, beta: float, copy_buffers: bool = True) -> None:
    """Update exponential moving average."""
    param_ema = dict(module_ema.named_parameters())
    param = dict(module.named_parameters())

    for key in param_ema:
        param_ema[key].data.mul_(beta).add_(param[key].data, alpha=(1 - beta))

    if copy_buffers:
        buffer_ema = dict(module_ema.named_buffers())
        buffer = dict(module.named_buffers())
        for key in buffer_ema:
            buffer_ema[key].data.copy_(buffer[key].data)


#######################################################################################################################
### Schedulers                                                                                                      ###
#######################################################################################################################


def _warmup(current_step: int, num_warmup_steps: int):
    """Calc factor on warmup."""
    return current_step / max(1.0, num_warmup_steps)


def _get_constant_schedule(num_warmup_steps: int | None = None) -> Callable:
    """Get function for constant schedule.

    Args:
    ----
        num_warmup_steps (int, optional): number of warmup steps.

    Returns:
    -------
        Callable: always returns 1.0

    """
    if num_warmup_steps is None:

        def lr_lambda(current_step: int):
            return 1.0
    else:

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            return 1.0

    return lr_lambda


def _get_multistep_schedule(milestones: list, num_warmup_steps: int | None = None, gamma=0.1) -> Callable:
    """Create function for multistep schedules.

    Args:
    ----
        milestones (list): list of steps on where to decay.
        num_warmup_steps (int, optional): number of warmup steps.
        gamma (float, optional): factor to decay on each milestone. Defaults to 0.1.

    Returns:
    -------
        Callable: function for LambdaLR

    """
    milestones = np.asarray(milestones)
    if num_warmup_steps is None:

        def lr_lambda(current_step: int):
            bigger_count = sum(milestones < current_step)
            return gamma**bigger_count
    else:

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            bigger_count = sum(milestones < current_step)
            return gamma**bigger_count

    return lr_lambda


def _get_linear_schedule(num_training_steps: int, num_warmup_steps: int | None) -> Callable:
    """Create function for linear schedule.

    Args:
    ----
        num_training_steps (int): total number of training steps.
        num_warmup_steps (int, optional): number of warmup steps.

    Returns:
    -------
        Callable: function for LambdaLR

    """
    if num_warmup_steps is None:

        def lr_lambda(current_step: int):
            return max(0.0, (num_training_steps - current_step) / max(1.0, num_training_steps))
    else:

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            return max(0.0, (num_training_steps - current_step) / max(1.0, num_training_steps - num_warmup_steps))

    return lr_lambda


def _get_polynomial_decay_schedule(
    num_training_steps: int, num_warmup_steps: int | None, lr_init: float, power: float = 1.0, lr_end: float = 1e-7
) -> Callable:
    """Create function for polynomial decay schedule.

    Args:
    ----
        num_training_steps (int): total number of training steps.
        num_warmup_steps (int): number of warmup steps.
        lr_init (float): initial learning rate.
        power (float, optional): _description_. Defaults to 1.0.
        lr_end (float, optional): _description_. Defaults to 1e-7.

    Returns:
    -------
        Callable: _description_

    """
    if num_warmup_steps is None:

        def lr_lambda(current_steps: int):
            if current_steps > num_training_steps:
                return lr_end / lr_init
            lr_range = lr_init - lr_end
            remaining = 1 - current_steps / num_training_steps
            decay = lr_range * remaining**power + lr_end
            return decay / lr_init

    else:

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            if current_step > num_training_steps:
                return lr_end / lr_init
            lr_range = lr_init - lr_end
            steps = num_training_steps - num_warmup_steps
            remaining = 1 - (current_step - num_warmup_steps) / steps
            decay = lr_range * remaining**power + lr_end
            return decay / lr_init

    return lr_lambda


def _get_cosine_schedule(
    num_training_steps: int, num_warmup_steps: int | None = None, num_cycles: float = 0.5
) -> Callable:
    """Create function for consine schedule.

    Args:
    ----
        num_training_steps (int): total number of training steps.
        num_warmup_steps (int, optional): number of warmup steps.
        num_cycles (float, optional): The number of waves in the cosine schedule. Default: 0.5.

    Returns:
    -------
        Callable: function for LambdaLR

    """
    if num_warmup_steps is None:

        def lr_lambda(current_step: int):
            progress = current_step / max(1, num_training_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    else:

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return lr_lambda


def create_scheduler(
    optimizer: Optimizer,
    type: str,
    num_training_steps: int,
    *,
    num_iter_per_step: int = 1,
    num_warmup_steps: int | None = None,
    milestones: list[int] | None = None,
    gamma: float = 0.1,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Build scheduler.

    types
    - constant
    - linear
    - poly or polynomial
    - multistep
    - cosine

    Args:
    ----
        optimizer (Optimizer): the optimizer.
        type (str): name of the scheduler.
        num_training_steps (int): total number of training steps. assumes epochs.
            to use iterations use num_iters_per_step or pass in iterations as steps.
        num_iter_per_step (int, optional): number of iterations per step. Default: 1.
        num_warmup_steps (int, optional): number of warmup steps. If None, no warmup phase. Default: None.
        milestones (list[int], optional): milestones for multistep scheduler. Default: None.
        gamma (float, optional): gamma for multistep scheduler. Default: 0.1.
        power (float, optional): power for polynomial decay schedule.
        last_epoch (int, optional): last epoch for resume training. Default: -1.

    Returns:
    -------
        LambdaLR: learning rate scheduler.

    """
    num_training_steps = num_training_steps * num_iter_per_step
    num_warmup_steps = num_warmup_steps * num_iter_per_step if num_warmup_steps is not None else None

    if type == 'constant':
        lr_lambda = _get_constant_schedule(num_warmup_steps)
    elif type == 'linear':
        lr_lambda = _get_linear_schedule(num_training_steps, num_warmup_steps)
    elif type in ('poly', 'polynomial'):
        lr_init = optimizer.defaults['lr']
        lr_lambda = _get_polynomial_decay_schedule(num_training_steps, num_warmup_steps, lr_init, power)
    elif type == 'multistep':
        assert milestones is not None
        milestones = [milestone * num_iter_per_step for milestone in milestones]
        lr_lambda = _get_multistep_schedule(milestones, num_warmup_steps, gamma)
    elif type == 'cosine':
        lr_lambda = _get_cosine_schedule(num_training_steps, num_warmup_steps)
    else:
        raise Exception(f'build_scheduler: No such scheduler type "{type}".')

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


#######################################################################################################################
### Gradient accumulation                                                                                           ###
#######################################################################################################################


def gradient_accumulation_steps(target_batch_size: int, batch_size: int) -> int:
    """Calculate gradient accumulation steps given the target batch size and batch size per iteration."""
    grad_accum_steps = target_batch_size // (batch_size * get_world_size())
    return grad_accum_steps


#######################################################################################################################
### Automatic Mixed Precision                                                                                       ###
#######################################################################################################################


def get_grad_scaler(enabled=True, is_fsdp=False) -> GradScaler | None:
    """Get the proper gradient scaler.

    Args:
    ----
        enabled (bool, optional): Enable gradient scaling? Default: True.
        is_fsdp (bool, optional): is distributed mode FSDP? Default: False.

    Returns:
    -------
        GradScaler | None: gradient scaler class

    """
    scaler = GradScaler(enabled=enabled) if not is_fsdp else ShardedGradScaler(enabled=enabled)
    return scaler


#######################################################################################################################
### Checkpointing                                                                                                   ###
#######################################################################################################################


def _is_stateful(obj: Any) -> bool:
    """Check if input is Stateful object."""
    return (
        hasattr(obj, 'state_dict')
        and hasattr(obj, 'load_state_dict')
        and callable(obj.state_dict)
        and callable(obj.load_state_dict)
    )


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap torch.compiled and DDP wrapped models."""
    if hasattr(model, 'dynamo_ctx'):
        model = model._orig_mod
    if isinstance(model, DDP):
        model = model.module
    return model


@contextmanager
def _fsdp_state_dict_context(model):
    """Context manager for get/set FSDP state_dict."""
    # Support only one pattern for simplicity.
    state_dict_type = StateDictType.FULL_STATE_DICT
    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, state_dict_type, state_dict_config, optim_state_dict_config):
        yield


##############
### Saving ###
##############


def _naive_get_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Naive implementation of get_model_state_dict."""
    model = _unwrap_model(model)

    if isinstance(model, FSDP):
        if not int(torch.__version__[0]) > 1:
            raise Exception('If using FSDP, use more latest versions.')

        with _fsdp_state_dict_context(model):
            return model.state_dict()

    return model.state_dict()


def _naive_get_optim_state_dict(model: nn.Module, optimizer: Optimizer) -> dict[str, torch.Tensor]:
    """Naive implementation of get_optim_state_dict."""
    model = _unwrap_model(model)

    if isinstance(model, FSDP):
        if not int(torch.__version__[0]) > 1:
            raise Exception('If using FSDP, use more latest versions.')

        with _fsdp_state_dict_context(model):
            return optimizer.state_dict()

    return optimizer.state_dict()


def _save_template(dir: str, get_to_save: Callable, name: str):
    """Save function."""
    output_path = os.path.join(dir, name + '.pt')
    to_save = get_to_save()
    if is_primary():
        torch.save(to_save, output_path)


def save_model(dir: str, model: nn.Module, name: str = 'model'):
    """Save model."""

    def get_to_save():
        if dcpsd is not None and is_distributed():
            option = dcpsd.StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state_dict = dcpsd.get_model_state_dict(model, options=option)
        else:
            model_state_dict = _naive_get_model_state_dict(model)
        return model_state_dict

    _save_template(dir, get_to_save, name)


def _save_optimizer(dir: str, model: nn.Module, optimizer: Optimizer, name: str = 'optim'):
    """Save optimizer."""

    def get_to_save():
        if dcpsd is not None and is_distributed():
            option = dcpsd.StateDictOptions(full_state_dict=True, cpu_offload=True)
            optim_state_dict = dcpsd.get_optimizer_state_dict(model, optimizer, options=option)
        else:
            optim_state_dict = _naive_get_optim_state_dict(model, optimizer)
        return optim_state_dict

    _save_template(dir, get_to_save, name)


def _save_object(dir: str, obj: Any, name: str):
    """Save python object."""
    _save_template(dir, lambda: obj, name)


def _save_stateful(dir: str, stateful: Any, name: str):
    """Save stateful object."""
    _save_template(dir, stateful.state_dict, name)


###############
### Loading ###
###############


def _naive_set_model_state_dict(model: nn.Module, state_dict: dict):
    """Naive implementation of set_model_state_dict."""
    model = _unwrap_model(model)

    if isinstance(model, FSDP):
        if not int(torch.__version__[0]) > 1:
            raise Exception('If using FSDP, use more latest versions.')

        with _fsdp_state_dict_context(model):
            return model.load_state_dict(state_dict)

    else:
        model.load_state_dict(state_dict)


def _naive_set_optim_state_dict(model: nn.Module, optimizer: Optimizer, state_dict: dict):
    """Naive implemetation of set_optim_state_dict."""
    model = _unwrap_model(model)

    if isinstance(model, FSDP):
        if not int(torch.__version__[0]) > 1:
            raise Exception('If using FSDP, use more latest versions.')

        with _fsdp_state_dict_context(model):
            return optimizer.load_state_dict(state_dict)

    else:
        return optimizer.load_state_dict(state_dict)


def load_model(dir: str, model: nn.Module, name: str = 'model', map_location='cpu', strict: bool = True):
    """Load state_dict to model."""
    state_path = os.path.join(dir, name + '.pt')
    state_dict = torch.load(state_path, map_location=map_location, weights_only=True)
    if dcpsd is not None and is_distributed():
        option = dcpsd.StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict)
        dcpsd.set_model_state_dict(model, state_dict, options=option)
    else:
        _naive_set_model_state_dict(model, state_dict)


def _load_optimizer(
    dir: str, model: nn.Module, optimizer: Optimizer, name: str = 'optim', map_location='cpu', strict: bool = True
):
    """Load state_dict to optimizer."""
    state_path = os.path.join(dir, name + '.pt')
    state_dict = torch.load(state_path, map_location=map_location)
    if dcpsd is not None and is_distributed():
        option = dcpsd.StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict)
        dcpsd.set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=option)
    else:
        _naive_set_optim_state_dict(model, optimizer, state_dict)


def _load_stateful(dir: str, stateful: Any, name: str, map_location='cpu'):
    """Load state_dict to stateful object."""
    state_path = os.path.join(dir, name + '.pt')
    state_dict = torch.load(state_path, map_location=map_location)
    stateful.load_state_dict(state_dict)


def _load_object(dir: str, name: str, map_location='cpu'):
    """Load python object."""
    state_path = os.path.join(dir, name + '.pt')
    obj = torch.load(state_path, map_location=map_location)
    return obj


def save_checkpoint(
    checkpoint_dir: str,
    *,
    model: nn.Module | list[nn.Module],
    optimizer: Optimizer | list[Optimizer | None],
    scheduler: LRScheduler | list[LRScheduler] | None = None,
    grad_scaler: GradScaler | None = None,
    others: dict[str, Any] | None = None,
):
    """Checkpointing.

    Supports nn.Module, DDP, FSDP wrapped models.

    Args:
    ----
        checkpoint_dir (str): The directory to save the models to.
        model (nn.Module | list[nn.Module]): Model or list of models to save.
        optimizer (Optimizer | list[nn.Module | None]): Optimizer or list of optimizers to save. The order must be
            same as the corresponding model. If a model is not meant to be trained pass a None.
        scheduler (LRScheduler | list[LRScheduler], optional): Scheduler or list of schedulers to save. Default: None.
        grad_scaler (GradScaler, optional): GradScaler object to save if using float16 AMP. Default: None.
        others (dict[str, Any], optional): Other objects. Default: None.

    Example:
    -------
        ```python
        model = create_model(...)
        optim = torch.optim.Adam(model.parameters())
        schdl = torch.optim.lr_scheduler.MultiStepLR(optim, [10])
        gradscaler = GradScaler()
        const = torch.randn(3, 3)

        save_checkpoint(
            './ckpt/1',
            model=model,
            optimizer=optim,
            scheduler=schdl,
            grad_scaler=gradscaler,
            others=dict(const=const),
        )
        ```

    """
    if is_primary():
        os.makedirs(checkpoint_dir, exist_ok=True)

    # save models and optimizers.
    if isinstance(model, Sequence):
        assert isinstance(optimizer, Sequence), 'If using multiple models, optimizer must also be a list.'
        assert len(model) == len(optimizer), 'model and optimizer must be same length.'
        for i, (m, o) in enumerate(zip(model, optimizer, strict=False)):
            save_model(checkpoint_dir, m, f'model.{i}')
            if o is not None:
                _save_optimizer(checkpoint_dir, m, o, name=f'optim.{i}')
    else:
        save_model(checkpoint_dir, model)
        _save_optimizer(checkpoint_dir, model, optimizer, name='optim')

    # save scheduler
    if isinstance(scheduler, Sequence):
        for i, s in enumerate(scheduler):
            _save_stateful(checkpoint_dir, s, name=f'scheduler.{i}')
    elif scheduler is not None:
        _save_stateful(checkpoint_dir, scheduler, name='scheduler')

    # save GradScaler
    if grad_scaler is not None:
        _save_stateful(checkpoint_dir, grad_scaler, name='scaler')

    # save other stateful or python objects.
    if others is not None:
        assert isinstance(others, dict)
        to_save = {}
        for key, value in others.items():
            if _is_stateful(value):
                to_save[key] = value.state_dict()
            else:
                to_save[key] = value
        _save_object(checkpoint_dir, to_save, name='others')

    # save random states.
    random_states = {}
    random_states['builtin'] = random.getstate()
    random_states['numpy'] = np.random.get_state()
    random_states['torch'] = torch.get_rng_state()
    if torch.cuda.is_available():
        random_states['torch.cuda'] = torch.cuda.get_rng_state_all()
    _save_object(checkpoint_dir, random_states, name='random-state')


def load_checkpoint(
    checkpoint_dir: str,
    *,
    model: nn.Module | list[nn.Module],
    optimizer: Optimizer | list[Optimizer | None],
    scheduler: LRScheduler | list[LRScheduler] | None = None,
    grad_scaler: GradScaler | None = None,
    others: dict[str, Any] | None = None,
    allow_empty: bool = False,
) -> dict:
    """Load saved checkpoint.

    Non-stateful objects must be overwritten by user code. See example.
    Supports nn.Module, DDP, FSDP wrapped models.

    Args:
    ----
        checkpoint_dir (str): The directory to saved models.
        model (nn.Module | list[nn.Module]): Model or list of models to save. If using multiple models, the order must
            be completely same as the arguments passed to `save_checkpoint` function.
        optimizer (Optimizer | list[nn.Module | None]): Optimizer or list of optimizers to save. The order must be
            same as the corresponding model. If a model is not meant to be trained pass a None.
        scheduler (LRScheduler | list[LRScheduler], optional): Scheduler or list of schedulers to save. Default: None.
        grad_scaler (GradScaler, optional): GradScaler object to save if using float16 AMP. Default: None.
        others (dict[str, Any], optional): Other objects. Default: None.
        allow_empty (bool): Allow the checkpoint_dir to not exist. If, so, do nothing and return. Default: False

    Returns:
    -------
        dict: constants or python builtin objs that cannot be loaded by this function.

    Example:
    -------
        ```python
        model = create_model(...)
        optim = torch.optim.Adam(model.parameters())
        schdl = torch.optim.lr_scheduler.MultiStepLR(optim, [10])
        gradscaler = GradScaler()

        consts = load_checkpoint(
            './ckpt/1',
            model=model,
            optimizer=optim,
            scheduler=schdl,
            grad_scaler=gradscaler,
            # others=dict(const=const),
        )
        # non-stateful objects must be overwritten manually.
        const = consts.get('const')
        ```

    """
    if not os.path.exists(checkpoint_dir) and allow_empty:
        return {}

    assert os.path.exists(checkpoint_dir)

    # load models and optimizers
    if isinstance(model, Sequence):
        assert isinstance(optimizer, Sequence), 'If using multiple models, optimizer must also be a list.'
        assert len(model) == len(optimizer), 'model and optimizer must be same length.'
        for i, (m, o) in enumerate(zip(model, optimizer)):  # noqa: B905
            # dcpsd.set_model_state_dict raises an error when FSDP modules are not initialized.
            # calling .state_dict() initializes the module.
            m.state_dict()
            load_model(checkpoint_dir, m, name=f'model.{i}')
            if o is not None:
                _load_optimizer(checkpoint_dir, m, o, name=f'optim.{i}')
    else:
        model.state_dict()
        load_model(checkpoint_dir, model)
        _load_optimizer(checkpoint_dir, model, optimizer, name='optim')

    # load schedulers
    if isinstance(scheduler, Sequence):
        for i, s in enumerate(scheduler):
            _load_stateful(checkpoint_dir, s, name=f'scheduler.{i}')
    elif scheduler is not None:
        _load_stateful(checkpoint_dir, scheduler, name='scheduler')

    # load GradScaler
    if grad_scaler is not None:
        _load_stateful(checkpoint_dir, grad_scaler, name='scaler')

    # load other objects.
    consts = {}
    if others is not None:
        assert isinstance(others, dict)
        objects = _load_object(checkpoint_dir, name='others')
        # the user needs to overwrite the constants themselves.
        for key, value in others.items():
            if _is_stateful(value):
                value.load_state_dict(objects[key])
            else:
                consts[key] = objects[key]

    # load random states.
    random_states = _load_object(checkpoint_dir, name='random-state')
    random.setstate(random_states.get('builtin'))
    np.random.set_state(random_states.get('numpy'))
    torch.set_rng_state(random_states.get('torch'))
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(random_states.get('torch.cuda'))

    return consts
