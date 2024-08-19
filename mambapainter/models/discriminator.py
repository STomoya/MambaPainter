from __future__ import annotations

import torch.nn as nn


class Discriminator(nn.Sequential):
    """A simple discriminator"""

    def __init__(self, channels: int = 768, image_channels: int = 3, num_res_layers: int = 1):
        layers = [
            nn.Conv2d(image_channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        for i in range(4):
            layers.extend(
                [
                    nn.Conv2d(int(channels // 2**i), int(channels // 2 ** (i + 1)), 3, 2, 1, bias=False),
                    nn.InstanceNorm2d(int(channels // 2 ** (i + 1))),
                    nn.LeakyReLU(negative_slope=0.2),
                ]
            )
            for _ in range(num_res_layers - 1):
                layers.extend(
                    [
                        nn.Conv2d(int(channels // 2 ** (i + 1)), int(channels // 2 ** (i + 1)), 3, 1, 1, bias=False),
                        nn.InstanceNorm2d(int(channels // 2 ** (i + 1))),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
        layers.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(int(channels // 2 ** (i + 1)), 1, bias=False),
            ]
        )

        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample: bool) -> None:
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            2 if downsample else 1,
            kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.dwnorm = nn.InstanceNorm2d(in_channels)
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.pwnorm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if in_channels != out_channels or downsample:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.skip_down = nn.AvgPool2d(2, 2)
        else:
            self.skip = nn.Identity()
            self.skip_down = nn.Identity()

    def forward(self, x):
        skip = self.skip_down(self.skip(x))
        x = self.dwconv(x)
        x = self.dwnorm(x)
        x = self.act(x)
        x = self.pwconv(x)
        x = self.pwnorm(x)
        x = x + skip
        x = self.act(x)
        return x


class ResidualDiscriminator(nn.Sequential):
    """Discriminator with residual blocks."""

    def __init__(
        self,
        channels: int = 768,
        image_channels: int = 3,
        num_res_layers: int = 1,
    ) -> None:
        layers = [
            nn.Conv2d(image_channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        for i in range(4):
            layers.append(ResidualBlock(int(channels // 2**i), int(channels // 2 ** (i + 1)), 3, downsample=True))

        layers.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(int(channels // 2 ** (i + 1)), 1, bias=False),
            ]
        )

        super().__init__(*layers)
