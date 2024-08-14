"""Utilities"""

from __future__ import annotations

import os
import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

import torchutils


def to_object(config: DictConfig | ListConfig):
    """Convert omegaconf objects to python objects recursively.

    This function always resolves the config before converting to python objects.
    omegaconf objects are `issubclass(DictConfig, dict) == issubclass(ListConfig, list) == False`. There
    are some cases were this behavior causes unexpected errors (e.g., isinstance(config, dict) is false).

    Args:
    ----
        config (DictConfig | ListConfig): the config to convert.

    Returns:
    -------
        dict | list: the converted config.

    """
    py_obj = OmegaConf.to_object(config)
    return py_obj


def get_hydra_config(
    config_dir: str, config_name: str, overrides: list[str] = sys.argv[1:], resolve: bool = True
) -> DictConfig:
    """Gather config using hydra.

    Args:
    ----
        config_dir (str): Relative path to directory where config files are stored.
        config_name (str): Filename of the head config file.
        overrides (list[str], optional): Overrides. Usually from command line arguments. Default: sys.argv[1:].
        resolve (bool, optional): resolve the config before returning. Default: True.

    Returns:
    -------
        DictConfig: Loaded config.

    """
    with initialize_config_dir(config_dir=to_absolute_path(config_dir), version_base=None):
        cfg = compose(config_name, overrides=overrides)
    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


@torchutils.only_on_primary
def save_hydra_config(config: DictConfig, filename: str, resolve: bool = True) -> None:
    """Save OmegaConf as yaml file.

    Args:
    ----
        config (DictConfig): Config to save.
        filename (str): filename of the saved config.
        resolve (bool, optional): resolve the config before saving. Default: True.

    """
    if resolve:
        OmegaConf.resolve(config)
    with open(filename, 'w') as fout:
        fout.write(OmegaConf.to_yaml(config))


def init_run(
    config_file: str,
    save_config: bool = True,
    config_dir: str = 'config',
    default_config_file: str = 'config.yaml',
) -> tuple[DictConfig, str]:
    """Load config, create workspace dir, and save config."""
    cmdargs = sys.argv[1:]

    # for resuming:
    # $ python3 train.py　./path/to/config.yaml
    if len(cmdargs) == 1 and cmdargs[0].endswith(config_file):
        config_path = cmdargs[0]
        config = OmegaConf.load(config_path)
        folder = os.path.dirname(config_path)

    # for a new run.
    else:
        config = get_hydra_config(config_dir, default_config_file)
        name = config.run.name
        tag = config.run.get('tag', None)
        if tag is not None:
            if tag == 'date':
                tag = torchutils.get_now_string()
            id = '.'.join([name, tag])
        else:
            id = name
        folder = os.path.join(config.run.folder, id)

        torchutils.makedirs0(folder, exist_ok=True)
        save_hydra_config(config, os.path.join(folder, config_file))

    return config, folder


def make_image_grid(*image_tensors, num_images: int | None = None):
    """Align images.

    Args:
    ----
        *image_tensors: image tensors.
        num_images (int, optional): _description_. Default: None.

    Returns:
    -------
        torch.Tensor: aligned images.

    Examples::
        >>> img1, img2 = [torch.randn(3, 3, 128, 128) for _ in range(2)]
        >>> aligned = make_image_grid(img1, img2)
        >>> # aligned.size() == [6, 3, 128, 128]
        >>> # aligned        == [img1[0], img2[0], img1[1], img2[1], img1[2], img2[2]]

        >>> # Any number of tensors can be passed to this function
        >>> # as long as the sizes are equal except for the batch dimension
        >>> img_tensors = [torch.randn(random.randint(1, 10), 3, 128, 128) for _ in range(24)]
        >>> aligned = make_image_grid(*img_tensors)

    """

    def _split(x):
        return x.chunk(x.size(0), 0)

    image_tensor_lists = map(_split, image_tensors)
    images = []
    for index, image_set in enumerate(zip(*image_tensor_lists, strict=False)):
        images.extend(list(image_set))
        if num_images is not None and index == num_images - 1:
            break
    return torch.cat(images, dim=0)
