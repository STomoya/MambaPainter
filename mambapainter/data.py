import glob
import os
from typing import Callable

import torch
import torchutils
from PIL import Image
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, param_dims: int, num_iters: int) -> None:
        super().__init__()
        self.param_dims = param_dims
        self.num_iters = num_iters

    def __len__(self):
        return self.num_iters

    def __getitem__(self, index):
        params = torch.rand(self.param_dims)
        params[2:4] = params[2:4] * 0.8 + 0.2
        return params


# from https://github.com/pytorch/vision/blob/6512146e447b69cc9fb379eb05e447a17d7f6d1c/torchvision/datasets/folder.py#L242
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


def _is_image_file(path: str) -> bool:
    """Return whether if the path is a PIL.Image.Image openable file.

    Args:
    ----
        path (str): the path to an image.

    Returns:
    -------
        bool: file?

    """
    ext = {os.path.splitext(os.path.basename(path))[-1].lower()}
    return ext.issubset(IMG_EXTENSIONS)


class ImageFolder(Dataset):
    def __init__(self, folder: str, transform: Callable) -> None:
        super().__init__()

        paths = glob.glob(os.path.join(folder, '**', '*'), recursive=True)
        image_paths = list(filter(_is_image_file, paths))
        self.image_paths = torchutils.natural_sort(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_paths[index]
        image = Image.open(image).convert('RGB')
        image = self.transform(image)
        return image
