"""Multi crop inference with overlapping pixels."""

import argparse
import glob
import os
import statistics
import time

import lpips
import torch
import torchvision.transforms.v2.functional as TF
from multi_crop_inference_fast import (
    create_image_patches,
    load_image,
    predict_strokes,
    prepare_models,
    render_image,
    save_image,
)
from torchvision.transforms import InterpolationMode


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', help='Path to the folder of saved checkpoints.')
    parser.add_argument('input', help='Dir to ImageNet val.')
    parser.add_argument('--output', '-o', default='.', help='Output results to.')

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
        default=16,
        type=int,
        help='Overlapping pixels. The patch size is `--patch-size + --overlap-pixels`',
    )
    parser.add_argument(
        '--stroke-padding',
        '-sp',
        default=32,
        type=int,
        help='Number of pixels to pad to the rendering image size.',
    )
    parser.add_argument('--batch-size', '-bs', default=256, type=int, help='Batch size.')
    parser.add_argument('--merge-every', default=10, type=int, help='Render n strokes to an image per merging.')

    return parser.parse_args()


def main():
    args = get_args()

    assert (
        torch.cuda.is_available()
    ), 'GPU is required because VMamba, which we use for the image encoder, does not have a CPU implementation. Abort.'
    device = torch.device('cuda')

    renderer, predictor = prepare_models(args.model_folder, device, build_triton=True)

    folder = os.path.abspath(args.input)
    image_folders = sorted(glob.glob(os.path.join(folder, '*')))
    image_paths = []
    # select 1010 images.
    for i, folder in enumerate(image_folders):
        num_images = 2 if i < 10 else 1
        temp_image_paths = glob.glob(os.path.join(folder, '*'))
        temp_image_paths = filter(os.path.isfile, temp_image_paths)
        temp_image_paths = sorted(temp_image_paths)[:num_images]
        image_paths.extend(temp_image_paths)
    print(image_paths[:10])

    durations = []
    pixel_losses = []
    lpips_losses = []
    with open(f'results.{args.image_size}.csv', 'w') as fp:
        fp.write('image_path,duration,pixel_loss,lpips_loss\n')

    loss_pixel = torch.nn.MSELoss()
    loss_lpips = lpips.LPIPS().to(device)

    for image_path in image_paths:
        image = load_image(image_path, args.image_size, device)
        image_patchs = create_image_patches(image, args.patch_size, overlap_pixels=args.overlap_pixels)

        # Measure time.
        torch.cuda.synchronize()
        inference_start = time.time()

        stroke_scale = 1 - (args.stroke_padding / predictor.image_size)

        # predict params
        params = predict_strokes(image_patchs, predictor, args.batch_size, stroke_scale)

        # render parameters.
        output = render_image(params, renderer, image_patchs, stroke_scale, args.merge_every, return_progress=False)

        torch.cuda.synchronize()
        duration = time.time() - inference_start

        pixel_loss = loss_pixel(output, image).item()
        lpips_loss = loss_lpips(
            TF.resize(output, (224, 224), InterpolationMode.BICUBIC),
            TF.resize(image, (224, 224), InterpolationMode.BICUBIC),
            normalize=True, # images are in [0,1]
        ).item()

        durations.append(duration)
        pixel_losses.append(pixel_loss)
        lpips_losses.append(lpips_loss)

        with open(f'results.{args.image_size}.csv', 'a') as fp:
            fp.write(f'{image_path},{duration},{pixel_loss},{lpips_loss}\n')

    # Take 1000 samples.
    # The first sample is excluded because the duration is almost always an outlier.
    durations = durations[1:1001]
    pixel_losses = pixel_losses[1:1001]
    lpips_losses = lpips_losses[1:1001]
    print(len(durations))
    # \pm is LaTeX stuff. Please reread as ±.
    print(statistics.mean(durations), '\\pm', statistics.stdev(durations))
    print(statistics.mean(pixel_losses), '\\pm', statistics.stdev(pixel_losses))
    print(statistics.mean(lpips_losses), '\\pm', statistics.stdev(lpips_losses))

    filename = os.path.join(args.output, 'sample.oilpaint.png')
    save_image(output, filename, normalize=False)


if __name__ == '__main__':
    main()
