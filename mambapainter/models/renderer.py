import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image


def erosion(x: torch.Tensor, m: int = 1) -> torch.Tensor:
    """Erosion.

    Args:
        x (torch.Tensor): input image.
        m (int, optional): Default: 1.

    Returns:
        torch.Tensor: eroded image.
    """
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.min(channel, dim=2)[0]
    return result


def dilation(x: torch.Tensor, m: int = 1) -> torch.Tensor:
    """Dilation.

    Args:
        x (torch.Tensor): input image.
        m (int, optional): Default: 1.

    Returns:
        torch.Tensor: dilated image.
    """
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=-1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.max(channel, dim=2)[0]
    return result


def _load_image(path: str, mode: str = 'L'):
    image = Image.open(path).convert(mode)
    image = TF.to_image(image)
    image = TF.to_dtype(image, dtype=torch.float32, scale=True)
    return image


def load_meta_brushes(
    image_folder: str,
    vertical_filename: str = 'brush_large_vertical.png',
    horizontal_filename: str = 'brush_large_horizontal.png',
):
    """Load meta brushes."""
    vimage = _load_image(os.path.join(image_folder, vertical_filename))
    himage = _load_image(os.path.join(image_folder, horizontal_filename))
    return torch.stack([vimage, himage], dim=0)  # [2, 1, H, W]


def parameter_to_grey_strokes(
    params: torch.Tensor,
    H: int,
    W: int,
    meta_brushes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate meta brushes given stroke geometry parameters.

    This function assumes that the stroke parameters are formatted as [cx,cy,w,h,theta,...]. Therefore, the last
    dimension must have more than five elements.

    Args:
        params (torch.Tensor): Tensor of shape [B,L,P].
        H (int): Height of the interpolated stroke image.
        W (int): Width.
        meta_brushes (torch.Tensor): The base image to interpolate.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The grey and alpha channel of the interpolated images.
    """
    assert params.size(-1) >= 5, f'Stroke parameters must have more than 5 elements. Got {params.size(-1)}.'

    # pack sequence to batch dim.
    B, L, P = params.size()
    N = B * L
    params = params.reshape(N, P)

    # split params.
    x0, y0, w, h, theta, *_ = torch.unbind(params, dim=1)

    # select stroke direction.
    index = torch.full((N,), -1, device=params.device)
    index[h > w] = 0
    index[h <= w] = 1

    # meta brushes and alphas.
    brush = meta_brushes[index.long()]
    alpha = (brush.detach().clone() > 0).float()

    # warp brush images.
    sin_theta = torch.sin(torch.pi * theta)
    cos_theta = torch.cos(torch.pi * theta)
    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    grid = torch.nn.functional.affine_grid(warp, torch.Size((N, 1, H, W)), align_corners=False)
    brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False, mode='bilinear')
    alpha = torch.nn.functional.grid_sample(alpha, grid, align_corners=False, mode='nearest')

    alpha = erosion(alpha)

    # NOTE: Disabled becuase dilation makes the rendered brush strokes blurry.
    # brush = dilation(brush)

    brush = brush.view(B, L, 1, H, W).contiguous()
    alpha = alpha.view(B, L, 1, H, W).contiguous()

    return brush, alpha


def color_strokes(params: torch.Tensor, strokes: torch.Tensor) -> torch.Tensor:
    """Give color to strokes.

    This function assumes that the stroke parameters are formatted as [...,R,G,B].

    Args:
        params (torch.Tensor): Stroke parameters.
        strokes (torch.Tensor): Grey stroke images of size [B,L,1,H,W]

    Returns:
        torch.Tensor: RGB stroke images
    """
    colored = strokes * params[..., -3:].unsqueeze(-1).unsqueeze(-1)
    return colored


def parameter_to_colored_strokes(
    params: torch.Tensor,
    H: int,
    W: int,
    meta_brushes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate meta brushes given stroke geometry parameters, then colorize.

    This function assumes that the stroke parameters are formatted as [cx,cy,w,h,theta,...,R,G,B].

    Args:
        params (torch.Tensor): Tensor of shape [B,L,P].
        H (int): Height of the interpolated stroke image.
        W (int): Width.
        meta_brushes (torch.Tensor): The base image to interpolate.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The RGB and alpha channel of the interpolated images.
    """

    strokes, alpha = parameter_to_grey_strokes(params=params, H=H, W=W, meta_brushes=meta_brushes)
    colored = color_strokes(params=params, strokes=strokes)

    return colored, alpha


def render_strokes(
    strokes: torch.Tensor, alphas: torch.Tensor, image_size: tuple[int, int], canvas: torch.Tensor | None = None
) -> torch.Tensor:
    """Render strokes to an image.

    If the canvas is not None, the strokes are rendered on this canvas.

    Args:
        strokes (torch.Tensor): Sequence of stroke image. [B,L,3,H,W]
        alphas (torch.Tensor): Sequence of alpha channel. [B,L,1,H,W]
        image_size (tuple[int, int]): Output image size.
        canvas (torch.Tensor | None): Current canvas. Default: None.

    Returns:
        torch.Tensor: Rendered canvas
    """

    device = strokes.device
    batch_size, num_strokes, *_ = strokes.size()

    if canvas is None:
        canvas = torch.zeros(batch_size, 3, *image_size, device=device)

    for i in range(num_strokes):
        foreground = strokes[:, i, :, :, :]
        alpha = alphas[:, i, :, :, :]
        canvas = foreground * alpha + canvas * (1 - alpha)

    return canvas


def render_parameters(
    params: torch.Tensor, image_size: tuple[int, int], meta_brushes: torch.Tensor, canvas: torch.Tensor | None = None
) -> torch.Tensor:
    """Render parameters to image.

    If the canvas is not None, the strokes are rendered on this canvas.

    Args:
        params (torch.Tensor): Sequence of stroke parameters. [B,L,P]
        image_size (tuple[int, int]): Output image size.
        meta_brushes (torch.Tensor): Base stroke images.
        canvas (torch.Tensor | None): Current canvas. Default: None.

    Returns:
        torch.Tensor: Rendered canvas
    """
    strokes, alphas = parameter_to_colored_strokes(params, *image_size, meta_brushes=meta_brushes)
    canvas = render_strokes(strokes, alphas, image_size, canvas=canvas)
    return canvas


class Renderer(nn.Module):
    """Renderer."""

    def __init__(
        self,
        default_image_size: tuple[int, int] | int = 128,
        brush_dir: str = './brushes',
    ) -> None:
        super().__init__()
        assert default_image_size is not None

        self.register_buffer('meta_brushes', load_meta_brushes(brush_dir))
        self.default_height, self.default_width = self._process_size(default_image_size)

    def _process_size(self, size: tuple[int, int] | int | None):
        if isinstance(size, Sequence):
            return size[:2]
        elif isinstance(size, int):
            return size, size
        else:
            return self.default_height, self.default_width

    def parameter_to_grey_strokes(
        self, params, image_size: tuple[int, int] | int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wraps `parameter_to_grey_strokes()`"""
        H, W = self._process_size(image_size)
        stroke, alpha = parameter_to_grey_strokes(params=params, H=H, W=W, meta_brushes=self.meta_brushes)
        return stroke, alpha

    def parameter_to_colored_strokes(
        self, params, image_size: tuple[int, int] | int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wraps `parameter_to_colored_strokes()`"""
        H, W = self._process_size(image_size)
        colored, alpha = parameter_to_colored_strokes(params=params, H=H, W=W, meta_brushes=self.meta_brushes)
        return colored, alpha

    def render_strokes(
        self,
        strokes: torch.Tensor,
        alphas: torch.Tensor,
        image_size: tuple[int, int] | int | None = None,
        canvas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Wraps `render_strokes()`"""
        image_size = self._process_size(image_size)
        canvas = render_strokes(strokes=strokes, alphas=alphas, image_size=image_size, canvas=canvas)
        return canvas

    def render_parameters(
        self,
        parameters: torch.Tensor,
        image_size: tuple[int, int] | int | None = None,
        canvas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Wraps `render_parameters()`"""
        image_size = self._process_size(image_size)
        canvas = render_parameters(
            params=parameters, image_size=image_size, meta_brushes=self.meta_brushes, canvas=canvas
        )
        return canvas

    def batched_render_parameters(
        self,
        parameters: torch.Tensor,
        image_size: tuple[int, int] | int | None = None,
        batch_size: int = 64,
        batch_dim: int = 0,
        return_progress: bool = False,
        num_progresses: int = 100,
        efficient: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Batched version of `render_parameters()`.

        NOTE: Tested only for inference.

        Args:
            parameters (torch.Tensor): Stroke parameters
            image_size (tuple[int, int] | int | None): Output image size. Default: None.
            batch_size (int): Batch size. Default: 64.
            batch_dim (int): Batch dimension. Default: 0.
            return_progress (bool): Return intermediate images. Default: False.
            num_progresses (int): Number of images to return if `return_progress` is `True`. Default: 100.

        Returns:
            torch.Tensor: The rendered image.
            tuple[torch.Tensor, list[torch.Tensor]]: The rendered image and the list of itermediate images.
        """
        image_size = self._process_size(image_size)

        if return_progress:
            save_indices = set(
                torch.linspace(0, parameters.size(batch_dim) - 1, num_progresses + 1, dtype=torch.int64)[:-1].tolist()
            )
        else:
            save_indices = set()

        canvas = torch.zeros(1, 3, *image_size, device=parameters.device, dtype=parameters.dtype)

        progress = []
        cur_iter = 0

        if efficient:
            stroke_size = tuple(size // 2 for size in image_size)
        else:
            stroke_size = image_size

        for params_batch in parameters.split(batch_size, dim=batch_dim):
            strokes, alphas = parameter_to_colored_strokes(params_batch, *stroke_size, self.meta_brushes)

            if efficient:
                B, L, *_ = strokes.size()
                strokes = F.interpolate(strokes.view((B * L, -1, *stroke_size)), size=image_size, mode='nearest').view(
                    B, L, -1, *image_size
                )
                alphas = F.interpolate(alphas.view((B * L, -1, *stroke_size)), size=image_size, mode='nearest').view(
                    B, L, -1, *image_size
                )

            for i in range(strokes.size(1)):
                foreground = strokes[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                canvas = foreground * alpha + canvas * (1 - alpha)

                cur_iter += 1
                if cur_iter in save_indices:
                    progress.append(canvas.detach().clone())

        if return_progress:
            return canvas, progress
        return canvas


class NeuralRendererBase(Renderer):
    """Base class for deep model-based renderer."""

    def __init__(
        self,
        image_size: tuple[int, int] | int,
        brush_dir: str,
        image_channels: int,
        param_dims: int,
    ) -> None:
        super().__init__(default_image_size=image_size, brush_dir=brush_dir)
        self.image_channels = image_channels
        self.param_dims = param_dims
        self.image_size = image_size

    def batched_forward(self, x: torch.Tensor, batch_size: int):
        """Batched version of the forward function."""
        outputs = []
        for batch_x in x.split(batch_size, dim=0):
            outputs.append(self(batch_x))
        return torch.cat(outputs, dim=0)

    def neural_grey_parameter_to_strokes(
        self, params: torch.Tensor, batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate RGB stroke images given stroke parameters.

        NOTE: This function is only for models that outputs grey scale stroke images.

        Args:
            params (torch.Tensor): Stroke parameters
            batch_size (int): Batch size. Default: 32.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: RGB and alpha images.
        """
        assert self.image_channels == 1

        batch_size, num_strokes, _ = params.size()

        params = params.view(batch_size * num_strokes, -1)  # [B*L,8]
        strokes = self.batched_forward(params[:, : self.param_dims], batch_size=batch_size)  # [B*L,2,H,W]
        grey_strokes = strokes[:, [0]]  # [B*L,1,H,W]
        # give color.
        colored = grey_strokes * params[:, -3:].unsqueeze(-1).unsqueeze(-1)  # [B*L,3,H,W]
        alphas = strokes[:, [1]]  # [B*L,1,H,W]

        H, W = colored.shape[-2:]

        colored = colored.view(batch_size, num_strokes, -1, H, W)  # [B,L,3,H,W]
        alphas = alphas.view(batch_size, num_strokes, -1, H, W)  # [B,L,1,H,W]

        return colored, alphas

    def neural_colored_parameter_to_strokes(
        self, params: torch.Tensor, batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate RGB stroke images given stroke parameters.

        NOTE: This function is only for models that outputs RGB stroke images.

        Args:
            params (torch.Tensor): Stroke parameters
            batch_size (int): Batch size. Default: 32.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: RGB and alpha images.
        """
        assert self.image_channels == 3

        batch_size, num_strokes, _ = params.size()

        params = params.view(batch_size * num_strokes, -1)  # [B*L,8]
        strokes = self.batched_forward(params, batch_size=batch_size)
        colored = strokes[:, :-1]
        alphas = strokes[:, [-1]]

        H, W = colored.shape[-2:]

        colored = colored.view(batch_size, num_strokes, -1, H, W)  # [B,L,3,H,W]
        alphas = alphas.view(batch_size, num_strokes, -1, H, W)  # [B,L,1,H,W]
        return colored, alphas

    def neural_render_parameters(
        self, params: torch.Tensor, batch_size: int | None = None, canvas: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Render RGB image given stroke parameters.

        This function automatically switches between rendering function between grey-scale and RGB models.

        Args:
            params (torch.Tensor): Stroke parameters
            batch_size (int): Batch size. Default: 32.
            canvas (torch.Tensor | None): Current canvas. Default: None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: RGB and alpha images.
        """
        batch_size = batch_size or params.size(0)

        if self.image_channels == 1:
            stroke, alpha = self.neural_grey_parameter_to_strokes(params=params, batch_size=batch_size)
        elif self.image_channels == 3:
            stroke, alpha = self.neural_colored_parameter_to_strokes(params=params, batch_size=batch_size)
        else:
            raise Exception('Not supported.')

        image_size = self._process_size(self.image_size)
        canvas = render_strokes(strokes=stroke, alphas=alpha, image_size=image_size, canvas=canvas)
        return canvas


class FCN(NeuralRendererBase):
    """FCN.
    Almost the same model as Compositional Neural Painter, but w/ InstanceNorm.
    """

    def __init__(
        self,
        param_dims: int = 5,
        brush_dir: str = './brushes',
        image_size: int | None = 128,
        image_channels: int | None = 1,
    ):
        super().__init__(image_size=128, brush_dir=brush_dir, image_channels=1, param_dims=param_dims)
        self.fc1 = nn.Linear(param_dims, 512)

        self.fc2 = nn.Linear(512, 1024)
        self.in1d2 = nn.InstanceNorm1d(1024)

        self.fc3 = nn.Linear(1024, 2048)
        self.in1d3 = nn.InstanceNorm1d(2048)

        self.fc4 = nn.Linear(2048, 4096)
        self.in1d4 = nn.InstanceNorm1d(4096)

        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.in2d1 = nn.InstanceNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.in2d3 = nn.InstanceNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)

        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.in2d5 = nn.InstanceNorm2d(8)

        self.conv6 = nn.Conv2d(8, 8, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.in1d2(self.fc2(x)))
        x = F.relu(self.in1d3(self.fc3(x)))
        x = F.relu(self.in1d4(self.fc4(x)))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.in2d1(self.conv1(x)))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.in2d3(self.conv3(x)))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.in2d5(self.conv5(x)))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return x
