
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from indextts.gpt.conformer.attention import (MultiHeadedAttention,
                                              RelPositionMultiHeadedAttention)
from indextts.gpt.conformer.embedding import (NoPositionalEncoding,
                                              PositionalEncoding,
                                              RelPositionalEncoding)
from indextts.gpt.conformer.subsampling import (Conv2dSubsampling2,
                                                Conv2dSubsampling4,
                                                Conv2dSubsampling6,
                                                Conv2dSubsampling8,
                                                LinearNoSubsampling)
from indextts.utils.common import make_pad_mask
from transformers import GPT2PreTrainedModel, GPT2Model

class FeedForwardModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, activation=nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class ConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size: int = 15, activation: torch.nn.Module = nn.ReLU(), bias: bool = True):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2*channels, kernel_size=1, stride=1, padding=1, bias=bias)

        assert kernel_size % 2 == 1, "kernel_size must be odd number"
        self.lorder = 0

        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=(
            kernel_size-1)//2, groups=channels, bias=bias)

        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.lorder = 0
        self.activation = activation
        self.use_layer_norm = True
        self.norm = nn.LayerNorm(channels)

    def forward(
            self,
            x: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            cache: torch.Tensor = torch.zeros((0, 0, 0)),) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        x = x.transpose(1, 2)
        # 如果有pad的话,把pad部分置0
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)
            # 此时没有cache，前面的pad部分为0，代表无效数据
            if cache.size(2) == 0:  # cache_t == 0
                # 从最后一个维度的左边开始填充self.lorder个0,右边不变
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            # 此时有了cache，前面的pad为cache部分，代表有效数据
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            # 更新cache，最后self.lorder个时间步
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        # for the FNN(FeedForward Neural Network) module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        # for the MHA(MultiHead Attention) module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size,
                                          eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pos_emb: torch.Tensor,
                mask_pad: torch.Tensor = torch.ones(
                    (0, 0, 0), dtype=torch.bool),
                att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
                cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0))) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if (self.feed_forward_macaron is not None):
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * \
                self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, pos_emb, mask, att_cache)

        if self.concat_after:
            x = residual+self.concat_linear(torch.cat((x, x_att), dim=-1))
        else:
            x = residual+self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        if (self.conv_module is not None):
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

class BaseEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d2":
            subsampling_class = Conv2dSubsampling2
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.embed(xs, masks)
        chunk_masks = masks
        mask_pad = masks  # (B, 1, T/subsample_rate)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        macaron_style: bool = False,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """

        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after)

        activation = torch.nn.SiLU()

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            dropout_rate,
        )

        # feed-forward module definition
        positionwise_layer = FeedForwardModule
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size,
                                  cnn_module_kernel,
                                  activation,)

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])