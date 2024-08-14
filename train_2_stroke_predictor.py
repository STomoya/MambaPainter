"""Train predictor."""

import os

import torch
import torchutils
import torchvision.transforms.v2 as T
from mambapainter import loss as gan_loss_fn
from mambapainter.data import ImageFolder
from mambapainter.models.discriminator import Discriminator
from mambapainter.models.predictor import MambaStrokePredictor
from mambapainter.models.renderer import FCN
from mambapainter.utils import init_run, make_image_grid, to_object
from omegaconf import OmegaConf
from torchvision.utils import save_image


def train():
    config, folder = init_run(config_file='config.yaml', default_config_file='predictor.yaml')

    device = torchutils.get_device()
    logger = torchutils.get_logger(
        config.run.name, filename=os.path.join(folder, config.log.log_file) if torchutils.is_primary() else None
    )

    torchutils.set_seeds(**config.reproduce)

    # dataset
    image_size = config.data.image_size
    dataset = ImageFolder(
        config.data.image_dir,
        T.Compose(
            [T.ToImage(), T.ToDtype(dtype=torch.float32, scale=True), T.RandomResizedCrop((image_size, image_size))]
        ),
    )
    dlkwargs = torchutils.get_dataloader_kwargs()
    dataset = torchutils.create_dataloader(
        dataset,
        **config.data.loader,
        num_workers=8,
        **dlkwargs,
    )

    # model
    model = MambaStrokePredictor(**config.model)
    # D
    disc_model = Discriminator(**config.discriminator)

    # load trained renderer
    renderer_config = OmegaConf.load(config.renderer.config)
    renderer_model_config = renderer_config.model
    renderer = FCN(**renderer_model_config)
    state_dict = torch.load(config.renderer.pretrained, map_location=device, weights_only=True)
    renderer.load_state_dict(state_dict.get('state_dict', state_dict))
    torchutils.freeze(renderer)

    # wrap
    _, cmodel = torchutils.wrap_module(model, config.env.strategy, config.env.compile)
    _, cdisc_model = torchutils.wrap_module(disc_model, config.env.strategy, config.env.compile)
    # _, crenderer = torchutils.wrap_module(renderer, config.env.strategy, config.env.compile)
    renderer.to(device)

    # optimizer
    optim_cfg = to_object(config.optimizer)
    optimizer = torch.optim.Adam(model.parameters(), **optim_cfg)
    disc_optim = torch.optim.Adam(disc_model.parameters(), **optim_cfg)

    # For AMP.
    grad_scaler = torchutils.get_grad_scaler(False)

    # criterion
    pixel_loss_fn = torch.nn.MSELoss()

    # initialization
    ckptfolder = os.path.join(folder, 'bins')
    constants = torchutils.load_checkpoint(
        checkpoint_dir=ckptfolder,
        allow_empty=True,
        model=[cmodel, cdisc_model],
        optimizer=[optimizer, disc_optim],
        grad_scaler=grad_scaler,
        others={'batches_done': 0},
    )

    batches_done = constants.get('batches_done', 0)
    while batches_done < config.train.iterations:
        if hasattr(dataset.sampler, 'set_epoch'):
            dataset.sampler.set_epoch(batches_done // len(dataset))

        loss, total = 0, 0
        for batch in dataset:
            gt_pixels = batch.to(device)

            pred_params = model(gt_pixels)  # [B,L,8]
            pred_pixels = renderer.neural_render_parameters(pred_params, batch_size=config.train.rendering_batch_size)

            pixel_loss = pixel_loss_fn(pred_pixels, gt_pixels) * config.train.pixel_lambda
            g_gan_loss = balanced_g_loss = 0
            if batches_done > config.train.gan_from:
                pred_logits = disc_model(pred_pixels)
                g_gan_loss = gan_loss_fn.ns_g_loss(pred_logits)
                balanced_g_loss = (
                    g_gan_loss / g_gan_loss.detach().clone() * pixel_loss.detach().clone() * config.train.gan_lambda
                )
            batch_loss = pixel_loss + balanced_g_loss

            grad_scaler.scale(batch_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

            d_gan_loss, gp_loss, d_loss = 0, 0, 0

            if batches_done > config.train.gan_from:
                # disc_gt_pixels = torch.rand_like(gt_pixels)
                disc_gt_pixels = gt_pixels
                disc_gt_pixels = gan_loss_fn.prepare_input(disc_gt_pixels)

                gt_logits = disc_model(disc_gt_pixels)
                pred_logits = disc_model(pred_pixels.detach())

                d_gan_loss = gan_loss_fn.ns_d_loss(gt_logits, pred_logits)
                gp_loss = gan_loss_fn.r1_regularization(gt_logits, disc_gt_pixels, grad_scaler) * config.train.gp_lambda

                d_loss = d_gan_loss + gp_loss

                grad_scaler.scale(d_loss).backward()
                grad_scaler.step(disc_optim)
                grad_scaler.update()
                disc_optim.zero_grad()

            if (
                batches_done == 1
                or (batches_done < 100 and batches_done % 5 == 0)
                or batches_done % config.log.log_interval == 0
                or batches_done == config.train.iterations
            ):
                percent = batches_done / config.train.iterations * 100
                msg = (
                    f'Progress: {percent: 6.2f}% | MSE: {pixel_loss:.5f} | G: {g_gan_loss:.5f} '
                    f'| Total: {batch_loss:.5f} | D: {d_loss:.5f} | GP: {gp_loss:.5f}'
                )
                logger.info(msg)

            batch_size = gt_pixels.size(0)
            batch_loss = batch_loss.detach()
            total += batch_size
            loss += batch_loss * batch_size

            batches_done += 1

            if batches_done % config.save.ckpt_every == 0:
                torchutils.save_checkpoint(
                    checkpoint_dir=ckptfolder,
                    model=[cmodel, cdisc_model],
                    optimizer=[optimizer, disc_optim],
                    grad_scaler=grad_scaler,
                    others={'batches_done': 0},
                )
                kbatches = f'{batches_done/1000:.2f}k'
                torchutils.save_model(folder, model, f'{kbatches}.pt')
            if batches_done % config.save.snap_every == 0:
                kbatches = f'{batches_done/1000:.2f}k'
                parameteric = renderer.render_parameters(pred_params, image_size)
                images = make_image_grid(gt_pixels, pred_pixels)
                save_image(images, os.path.join(folder, f'snapshot-{kbatches}.png'), pad_value=255)
            if (batches_done - 1) % config.save.running == 0:
                parameteric = renderer.render_parameters(pred_params, image_size)
                images = make_image_grid(gt_pixels, pred_pixels, parameteric)
                save_image(images, os.path.join(folder, 'running.png'), nrow=9, pad_value=255)

            if batches_done >= config.train.iterations:
                break

        loss = torchutils.reduce(loss)
        total = torchutils.reduce(torch.tensor([total], device=device))
        loss = (loss / total).item()
        logger.info(f'Epoch loss: {loss}')

    torchutils.save_model(folder, model, 'last-model')


if __name__ == '__main__':
    train()
