"""Multi crop inference with overlapping pixels.

1. Split images to patches.
2. Predict stroke parameters.
3. Scale stroke parameters so that most strokes are not cut off at the edge.
4. Render stroke images.
5. Alpha blend strokes.
6. Merge patches, considering the overlapping pixels and scaled stroke parameters.

3-6 is done iteratively.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image

from mambapainter.models.predictor import MambaStrokePredictor
from mambapainter.models.renderer import Renderer
from torchutils import get_logger


def load_image(path: str, size: int, device: str = 'cuda'):
    image = Image.open(path).convert('RGB')
    image = TF.resize(image, (size, size))
    image = TF.to_image(image)
    image = TF.to_dtype(image, scale=True)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image


@dataclass(repr=False)
class ImagePatches:
    """Container for image patches."""

    data: torch.Tensor  # Patches stacked in the batch dimension.

    image_size: tuple[int, int]
    overlap_pixels: int

    patch_size: int
    num_patches: int

    nrow: int
    ncol: int

    pad_size: int

    def __repr__(self):
        return (
            'ImagePatches('
            f'\n  data=tensor(shape={self.data.size()}, device={self.data.device}, dtype={self.data.dtype}),'
            + ''.join([f'\n  {k}={v},' for k, v in self.__dict__.items() if k != 'data'])
            + '\n)'
        )


def create_image_patches(image: torch.Tensor, base_patch_size: int, overlap_pixels: int = 10) -> ImagePatches:
    """Crop images into patches, supporting overlapping pixels.

    Currently, this function only supports square images.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)

    assert image.size(0) == 1, 'currently only supports single images.'
    assert image.size(-1) == image.size(-2), 'currently only supports squared images.'
    assert image.size(-1) % base_patch_size == 0, 'The image size must be divisible by "base_patch_size".'
    assert overlap_pixels % 2 == 0, '"overlap_pixels" must be divisible by 2.'

    kernel_size = base_patch_size + overlap_pixels
    stride = base_patch_size
    padding = (overlap_pixels // 2, overlap_pixels // 2, overlap_pixels // 2, overlap_pixels // 2)

    # pad if needed.
    padded_image = F.pad(image, pad=padding, mode='reflect')

    # image sizes.
    B, C, H, W = image.size()
    _, _, padH, padW = padded_image.size()

    # crop images.
    patches = F.unfold(padded_image, kernel_size=kernel_size, stride=stride)
    num_patches = patches.size(-1)
    patches = patches.permute(0, 2, 1).reshape(B * num_patches, -1)
    patches = patches.reshape(B * num_patches, C, kernel_size, kernel_size)

    # variables.
    nrow = ncol = int(math.sqrt(num_patches))

    image = ImagePatches(
        data=patches,
        image_size=(image.size(-2), image.size(-1)),
        patch_size=base_patch_size,
        overlap_pixels=overlap_pixels,
        num_patches=num_patches,
        nrow=nrow,
        ncol=ncol,
        pad_size=kernel_size,
    )
    return image


def crop_image(canvas: torch.Tensor, padding: int):
    """crop images according to padded pixels."""
    if padding > 0:
        return canvas[..., padding:-padding, padding:-padding]
    else:
        return canvas


def freeze(model: torch.nn.Module):
    model.eval()
    model.requires_grad_(False)


def prepare_models(
    checkpoint_folder: str, device: torch.device, build_triton: bool = False, last_model: bool = True
) -> tuple[Renderer, MambaStrokePredictor]:
    config_file = os.path.join(checkpoint_folder, 'config.yaml')
    config = OmegaConf.load(config_file)

    renderer = Renderer()
    config.model.pop('builder')
    predictor = MambaStrokePredictor(**config.model)

    # loss.torch is almost always the model trained only on MSE.
    # final-model.torch
    state_dict_file = os.path.join(checkpoint_folder, 'last-model.torch' if last_model else 'loss.torch')
    if last_model and not os.path.exists(state_dict_file):
        state_dict_file = os.path.join(checkpoint_folder, 'loss.torch')

    state_dict = torch.load(state_dict_file, map_location='cpu')
    state_dict = state_dict.get('state_dict', state_dict)
    predictor.load_state_dict(state_dict)

    renderer.to(device)
    predictor.to(device)
    freeze(renderer)
    freeze(predictor)

    # build triton kernels before inference using a dummy tensor.
    if build_triton:
        predictor(torch.randn(1, 3, predictor.image_size, predictor.image_size, device='cuda'))

    return renderer, predictor


@torch.no_grad()
def predict_strokes(
    image: ImagePatches,
    predictor: MambaStrokePredictor,
    batch_size: int = 64,
    stroke_scale: int = 0.9,
):
    """forward predictor on image patches."""
    # resize to model input.
    image_patches = TF.resize(image.data, (predictor.image_size, predictor.image_size))

    # batched forward. number of patches can be very large.
    # The first iteration might be very slow if the triton kernel is not build beforehand.
    if image.num_patches > batch_size:
        params_list = []
        for batch in image_patches.split(batch_size, dim=0):
            params_list.append(predictor(batch))
        params = torch.cat(params_list)
    else:
        params = predictor(image_patches)  # [BLP], P: [xywhtrgb]

    # transform stroke parameters so that they probably will not get cut off on the edge.
    params = params.reshape(image.nrow, image.ncol, *params.size()[-2:])
    # scale width, height
    params[..., 2] = params[..., 2] * stroke_scale
    params[..., 3] = params[..., 3] * stroke_scale
    for y in range(image.nrow):
        for x in range(image.ncol):
            # translate x,y
            params[y, x, :, 0] = params[y, x, :, 0] * stroke_scale + (1 - stroke_scale) / 2
            params[y, x, :, 1] = params[y, x, :, 1] * stroke_scale + (1 - stroke_scale) / 2
            # theta and colors are unchanged.

    params = params.reshape(-1, *params.size()[-2:])  # [BLP]
    return params


def merge_patches(rendered_patches: torch.Tensor, image_patches: ImagePatches, patch_size: int):
    """Merge patches considering the overlapping pixels and stroke padding"""
    rendered_patches = rendered_patches.view(image_patches.nrow, image_patches.ncol, 3, patch_size, patch_size)

    pad_size = patch_size - image_patches.patch_size
    h, w = image_patches.image_size
    canvas = torch.zeros(1, 3, h + pad_size, w + pad_size, device=rendered_patches.device, dtype=rendered_patches.dtype)

    patch_size = image_patches.patch_size
    for y in range(image_patches.nrow):
        for x in range(image_patches.ncol):
            y1, y2 = y * patch_size, (y + 1) * patch_size + pad_size
            x1, x2 = x * patch_size, (x + 1) * patch_size + pad_size
            alpha = (rendered_patches[y, x] > 0).float()
            canvas[..., y1:y2, x1:x2] = canvas[..., y1:y2, x1:x2] * (1 - alpha) + rendered_patches[y, x] * alpha

    canvas = crop_image(canvas, pad_size // 2)
    return canvas


def render_image(
    params: torch.Tensor,
    renderer: Renderer,
    image_patches: ImagePatches,
    stroke_scale: float,
    merge_every: int = 10,
    return_progress: bool = True,
):
    # NOTE: renderer on bigger size so that we can avoid resizing which cause aliasing.
    #       resizing after rendering will be more efficient, but with lower reconstruction quality.
    #       maybe make this optional?
    patch_size = int(image_patches.pad_size / stroke_scale)
    patch_size += 1 if patch_size % 2 == 1 else 0

    output = 0
    progress = []
    for param_batch in params.split(merge_every, dim=1):
        rendered_patches = renderer.render_parameters(param_batch, image_size=patch_size)
        temp_output = merge_patches(rendered_patches, image_patches, patch_size)
        alpha = (temp_output > 0).float()
        output = output * (1 - alpha) + temp_output * alpha
        if return_progress:
            progress.append(output.clone())

    if return_progress:
        return output, progress
    return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', help='Path to the folder of saved checkpoints.')
    parser.add_argument('input', help='Input filename.')
    parser.add_argument('--output', '-o', default='.', help='Output results to.')

    parser.add_argument('--params', help='Path to saved parameters.')

    parser.add_argument('--image-size', '-is', default=512, type=int, help='Output image size.')
    parser.add_argument(
        '--patch-size',
        '-ps',
        default=64,
        type=int,
        help='Size of each image patch. The patch size is `--patch-size + --overlap-pixels`',
    )
    parser.add_argument(
        '--overlap-pixels',
        '-op',
        default=10,
        type=int,
        help='Overlapping pixels. The patch size is `--patch-size + --overlap-pixels`',
    )
    parser.add_argument(
        '--stroke-padding',
        '-sp',
        default=20,
        type=int,
        help='Number of pixels to pad to the rendering image size.',
    )
    parser.add_argument('--batch-size', '-bs', default=256, type=int, help='Batch size.')
    parser.add_argument('--merge-every', default=10, type=int, help='Render n strokes to an image per merging.')

    parser.add_argument('--save-timelapse', default=False, action='store_true', help='Save a timelapse as a GIF file.')
    parser.add_argument('--gif-optimize', default=False, action='store_true')
    parser.add_argument('--gif-duration', default=100, type=int)
    parser.add_argument('--gif-loop', default=0, type=int)

    parser.add_argument(
        '--save-parameters',
        default=False,
        action='store_true',
        help='Save predicted parameters. Useful when you want to quickly recreate the timelapse GIF.',
    )

    parser.add_argument(
        '--save-patches', default=False, action='store_true', help='Save image patches used to render the output.'
    )

    parser.add_argument(
        '--save-all', default=False, action='store_true', help='Trigger all saving flags, for peaple who are too lazy.'
    )

    return parser.parse_args()


def main():
    args = get_args()

    logger = get_logger('inference')

    assert (
        torch.cuda.is_available()
    ), 'GPU is required because VMamba, which we use for the image encoder, does not have a CPU implementation. Abort.'
    device = torch.device('cuda')

    stem = os.path.splitext(os.path.basename(args.input))[0]

    # save auguments as json.
    with open(os.path.join(args.output, f'{stem}.args.json'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    logger.info('Loading model. This will take some time to build triton kernels if --params is not given.')
    renderer, predictor = prepare_models(args.model_folder, device, build_triton=args.params is None)
    logger.info('Loaded model.')

    image = load_image(args.input, args.image_size, device)

    image_patchs = create_image_patches(image, args.patch_size, overlap_pixels=args.overlap_pixels)
    logger.info(f'Image patches:\n{image_patchs}')
    if args.save_all or args.save_patches:
        filename = os.path.join(args.output, f'{stem}.patches.png')
        save_image(image_patchs.data, filename, pad_value=255, nrow=image_patchs.nrow)
        logger.info(f'Saved image patches to {filename}.')

    # Measure time.
    torch.cuda.synchronize()
    inference_start = time.time()

    stroke_scale = 1 - (args.stroke_padding / predictor.image_size)

    # predict params (or load saved parameters.)
    params = None
    if args.params is not None:
        try:
            params = torch.load(args.params, map_location=device)
            logger.info(f'Loaded saved parameters from {args.params}.')
        except Exception as e:
            logger.warn(f'Could not load parameters. Error message: "{e!s}". Predicting parameters from image.')
    if params is None:
        params = predict_strokes(image_patchs, predictor, args.batch_size, stroke_scale)
        logger.info('Predicted parameters.')

    # render parameters.
    output, progress = render_image(
        params, renderer, image_patchs, stroke_scale, args.merge_every, return_progress=True
    )
    logger.info('Rendered image.')

    # Log time.
    torch.cuda.synchronize()
    inference_duration = time.time() - inference_start
    logger.info(f'Duration: {inference_duration:.5f} sec.')

    # save result.
    filename = os.path.join(args.output, f'{stem}.oilpaint.png')
    save_image(output, filename, normalize=False)
    logger.info(f'Translated image saved to {filename}.')

    # save timelapse.
    if args.save_all or args.save_timelapse:
        logger.info('Saving timelapse. This might take some time...')

        progress_images = []
        for p in progress:
            # From `torchvision.utils.save_image()`
            p_ndarr = p.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            image = Image.fromarray(p_ndarr)
            progress_images.append(image)

        filename = os.path.join(args.output, f'{stem}.timelapse.gif')
        progress_images[0].save(
            filename,
            save_all=True,
            append_images=progress_images[1:],
            optimize=args.gif_optimize,
            duration=args.gif_duration,
            loop=args.gif_loop,
        )
        logger.info(f'Saved timelapse GIF to {filename}.')

    # save predicted parameters.
    if args.save_all or args.save_parameters:
        filename = os.path.join(args.output, f'{stem}.parameters.pt')
        torch.save(params.cpu(), filename)
        logger.info(f'Saved predicted parameters to {filename}.')

    logger.info('Successfully done. Exiting.')


if __name__ == '__main__':
    main()
