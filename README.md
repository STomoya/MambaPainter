
# MambaPainter

Official implementation of "MambaPainter".

[Paper]() | [Code](https://github.com/STomoya/MambaPainter)

<div align="center">
    <img src="assets/results.png">
</div>

## Setup

- Install dependencies. We assume that `torch` and `torchvision` is already installed. Checked that torch versions `2.3.x`, `2.4.0` is moving.

    ```sh
    pip install -r requirements.txt
    ```

- Install the `selective_scan` module in [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba).

    You can optionally erase the `VMamba` repository after installing because we will not be using it.

    ```sh
    git clone https://github.com/MzeroMiko/VMamba.git
    cd VMamba
    git switch -d 0a74a29eefb9223efc1a399000e22a723390defd
    cd kernels/selective_scan
    pip install .
    ```

<details>
<summary>Docker</summary>

We provide the docker compose files that reproduce the environment used to train the models.

- First, setup the `DATASET_DIR` in [`.env`](./.env) to the dataset directory.

- Build image.

    ```sh
    docker compose build
    ```
</details>


## Training

The configuration is done using the [hydra](https://hydra.cc/) package.

1. Train neural renderer.

    Edit the [configuration file](./config/renderer.yaml).

    ```sh
    torchrun train_1_neural_renderer.py
    ```

2. Train MambaPainter.

    Edit the [configuration file](./config/predictor.yaml).

    The values in `<>` must be edited.

    ```sh
    torchrun train_2_stroke_predictor.py \
        data.image_dir=<path/to/dataset> \
        renderer.config_file=<path/to/renderer/config.yaml> \
        renderer.pretrained=<path/to/renderer/weights.pt>
    ```

## Inference

Use the checkpoint folder created above as `<path/to/trained/folder>` in the command below.

You can also download pretrained files from [GDrive]() or [Huggingface](). Create a folder and place the downloaded `config.yaml` and `model.pt` inside. Use the created folder as `<path/to/trained/folder>` in the command below.


```sh
python multi_patch_inference_fast.py \
    <path/to/trained/folder> \
    <path/to/image.jpg> \
    --output . \
    --image-size 512 \
    --patch-size 64 \
    --overlap-pixels 32 \
    --stroke-padding 32
```

The script will automatically create the translated image and a JSON file containing the command line arguments. You can add the `--save-all` option to save an image of patches used in the translation, predicted stroke parameters, and a timelapse GIF.

<details>
<summary>Help</summary>

```sh
$ python multi_patch_inference_fast.py --help
usage: multi_patch_inference_fast.py [-h] [--output OUTPUT] [--params PARAMS] [--image-size IMAGE_SIZE] [--patch-size PATCH_SIZE] [--overlap-pixels OVERLAP_PIXELS]
                                     [--stroke-padding STROKE_PADDING] [--batch-size BATCH_SIZE] [--merge-every MERGE_EVERY] [--save-timelapse] [--gif-optimize]
                                     [--gif-duration GIF_DURATION] [--gif-loop GIF_LOOP] [--save-parameters] [--save-patches] [--save-all]
                                     model_folder input

positional arguments:
  model_folder          Path to the folder of saved checkpoints.
  input                 Input filename.

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output results to.
  --params PARAMS       Path to saved parameters.
  --image-size IMAGE_SIZE, -is IMAGE_SIZE
                        Output image size.
  --patch-size PATCH_SIZE, -ps PATCH_SIZE
                        Size of each image patch. The patch size is `--patch-size + --overlap-pixels`
  --overlap-pixels OVERLAP_PIXELS, -op OVERLAP_PIXELS
                        Overlapping pixels. The patch size is `--patch-size + --overlap-pixels`
  --stroke-padding STROKE_PADDING, -sp STROKE_PADDING
                        Number of pixels to pad to the rendering image size.
  --batch-size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size.
  --merge-every MERGE_EVERY
                        Render n strokes to an image per merging.
  --save-timelapse      Save a timelapse as a GIF file.
  --gif-optimize
  --gif-duration GIF_DURATION
  --gif-loop GIF_LOOP
  --save-parameters     Save predicted parameters. Useful when you want to quickly recreate the timelapse GIF.
  --save-patches        Save image patches used to render the output.
  --save-all            Trigger all saving flags, for peaple who are too lazy.
```
</details>

## Citation

```bibtex
@proceedings{aaa,
    title  =  {},
}
```
