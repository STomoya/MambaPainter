"""NeuralRenderer."""

import os

import lpips
import torch
import torchutils
from mambapainter.data import RandomDataset
from mambapainter.models.renderer import FCN
from mambapainter.utils import init_run, make_image_grid, to_object
from torchvision.utils import save_image


def train_renderer():
    config, folder = init_run(config_file='config.yaml', default_config_file='renderer.yaml')

    device = torchutils.get_device()
    logger = torchutils.get_logger(
        config.run.name, filename=os.path.join(folder, config.log.log_file) if torchutils.is_primary() else None
    )

    torchutils.set_seeds(**config.reproduce)

    # dataset
    image_size = config.data.image_size
    dataset = RandomDataset(config.data.param_dims, config.train.iterations * config.data.loader.batch_size // 10)
    dlkwargs = torchutils.get_dataloader_kwargs()
    dataset = torchutils.create_dataloader(
        dataset,
        **config.data.loader,
        num_workers=1,
        **dlkwargs,
    )

    # modelbrushes
    model = FCN(**config.model)
    wrapped, compiled = torchutils.wrap_module(model, config.env.strategy, config.env.compile)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), **to_object(config.optimizer))

    # For AMP.
    grad_scaler = torchutils.get_grad_scaler(config.env.mixed_precision)

    # criterion
    stroke_loss_fn = torch.nn.MSELoss()
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    alpha_loss_fn = torch.nn.BCEWithLogitsLoss()

    # initilizations
    ckptfolder = os.path.join(folder, 'bins')
    constants = torchutils.load_checkpoint(
        checkpoint_dir=ckptfolder,
        allow_empty=True,
        model=compiled,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        others={'batches_done': 0},
    )

    # main loop
    batches_done = constants.get('batches_done', 0)
    while batches_done < config.train.iterations:
        if hasattr(dataset.sampler, 'set_epoch'):
            dataset.sampler.set_epoch(batches_done // len(dataset))

        loss, total = 0, 0
        for batch in dataset:
            params = batch.to(device)  # [B,D]

            gt_strokes, gt_alphas = model.parameter_to_grey_strokes(params.unsqueeze(1), image_size)  # [B,1,1,H,W] x 2
            gt_strokes = gt_strokes.squeeze(1)  # squash L (=1). [B,1,H,W]
            gt_alphas = gt_alphas.squeeze(1)  # squash L. [B,1,H,W]

            with torch.cuda.amp.autocast(config.env.mixed_precision):
                nr_pixels = compiled(params)  # [B,2,H,W]

                grey, alpha = nr_pixels.chunk(2, dim=1)
                stroke_loss = stroke_loss_fn(grey, gt_strokes) * config.train.stroke_lambda
                lpips_loss = (
                    lpips_loss_fn(grey.repeat(1, 3, 1, 1), gt_strokes.repeat(1, 3, 1, 1), normalize=True).mean()
                    * config.train.lpips_lambda
                )
                alpha_loss = alpha_loss_fn(alpha, gt_alphas) * config.train.alpha_lambda
                batch_loss = stroke_loss + alpha_loss + lpips_loss

            grad_scaler.scale(batch_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

            batches_done += 1

            if (
                batches_done == 1
                or (batches_done < 100 and batches_done % 5 == 0)
                or batches_done % config.log.log_interval == 0
                or batches_done == config.train.iterations
            ):
                percent = batches_done / config.train.iterations * 100
                msg = (
                    f'Progress: {percent:5.2f}% | MSE: {stroke_loss:.5f} | LPIPS; {lpips_loss:.5f} '
                    f'| BCE: {alpha_loss:.5f} | Total: {batch_loss:.5f}'
                )
                logger.info(msg)

            batch_size = params.size(0)
            batch_loss = batch_loss.detach()
            total += batch_size
            loss += batch_loss * batch_size

            if batches_done % config.save.ckpt_every == 0:
                torchutils.save_checkpoint(
                    checkpoint_dir=ckptfolder,
                    model=compiled,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    others={'batches_done': batches_done},
                )
                kbatches = f'{batches_done/1000:.2f}k'
                torchutils.save_model(folder, model, f'{kbatches}')
            if batches_done % config.save.snap_every == 0:
                kbatches = f'{batches_done/1000:.2f}k'
                images = make_image_grid(torch.cat([gt_strokes, gt_alphas], dim=1), nr_pixels)
                save_image(images, os.path.join(folder, f'snapshot-{kbatches}.png'), pad_value=255)
            if (batches_done - 1) % config.save.running == 0:
                save_image(nr_pixels, os.path.join(folder, 'running.png'), pad_value=255)

            if batches_done >= config.train.iterations:
                break

        loss = torchutils.reduce(loss)
        total = torchutils.reduce(torch.tensor([total], device=device))
        loss = (loss / total).item()
        logger.info(f'Epoch loss: {loss}')

    torchutils.save_model(folder, model, 'last-model')
    torchutils.destroy_process_group()


if __name__ == '__main__':
    train_renderer()
