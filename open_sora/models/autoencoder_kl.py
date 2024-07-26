from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .vae import Decoder, DiagonalGaussianDistribution, Encoder


class AutoencoderKL(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding
    latent representations into images.
    """

    def __init__(
        self,
        down_block_types,
        up_block_types,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.latent_channels = latent_channels

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_blocks=len(down_block_types),
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            num_blocks=len(down_block_types),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    def encode(self, x: mx.array):
        """
        Encode a batch of images into latents.

        Args:
            x (`mx.array`): Input batch of images.
        """
        h = self.encoder(x)
        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h

        posterior = DiagonalGaussianDistribution(moments)
        return posterior


    def decode(self, z: mx.array):
        """
        Decode a batch of images.

        Args:
            z (`mx.array`): Input batch of latent vectors.

        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        return self.decoder(z)
