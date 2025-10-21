# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

import indextts.BigVGAN.activations as activations
from indextts.BigVGAN.alias_free_activation.torch.act import \
    Activation1d as TorchActivation1d
from indextts.BigVGAN.ECAPA_TDNN import ECAPA_TDNN
from indextts.BigVGAN.env import AttrDict
from indextts.BigVGAN.utils import get_padding, init_weights

# BigVGAN:Big Vocoder Generative Adversarial Network



# 定义函数 load_hparams_from_json()————从一个 JSON 文件中读取超参数，并以对象方式访问
# 从一个 JSON 文件中读取超参数（hparams），并把它们加载成一个可以像对象一样访问属性的字典（AttrDict）
# 参数 path 是文件路径（一个字符串），表示要读取的 JSON 文件的位置。
# -> AttrDict 表示函数的返回类型是 AttrDict（这是一种带属性访问功能的字典类型，比如 params.learning_rate 而不是 params['learning_rate']）。
def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()     # 打开 path 指定的文件，并将文件内容读入字符串变量 data
    return AttrDict(json.loads(data))
    # 示例：json
    # {
    #     "learning_rate": 0.001,
    #     "batch_size": 32
    # }
    # json.loads(data) 会把 JSON 字符串转换为 Python 字典：{"learning_rate": 0.001, "batch_size": 32}
    # AttrDict(...) 会把这个普通字典包（用键值对访问，例如 hparams['learning_rate']）装成一个可以用点号访问属性的对象，例如：
        # hparams = load_hparams_from_json("config.json")
        # print(hparams.learning_rate)      # 输出 0.001
        # print(hparams.batch_size)  # 输出 32



# 定义一个 PyTorch 模块类 —— AMPBlock1
# 它是一个带有特殊激活函数（如 Snake、SnakeBeta）和空洞卷积（dilated convolution）的一维卷积残差块，常用于音频生成模型（如 HiFi-GAN、BigVGAN 等）中。
# 主要功能：对输入信号（通常是一维音频特征）进行多层卷积 + 激活处理，并通过残差连接增强特征表达能力和稳定训练。
# AMPBlock1 继承自 torch.nn.Module，是所有 PyTorch 神经网络层的基础类
class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,                    # 一个包含模型超参数的 AttrDict（前面解释过的那种对象，可以点号访问）
        channels: int,                  # 输入和输出的通道数（即每层卷积的特征维度）
        kernel_size: int = 3,           # 卷积核大小（默认3）
        dilation: tuple = (1, 3, 5),    # 空洞卷积的膨胀系数，用于扩大感受野（默认 (1, 3, 5)）
        activation: str = None,         # 激活函数类型，支持 "snake" 或 "snakebeta"
    ):
        super().__init__()              # 调用父类（超类）的构造方法。在子类的构造方法中使用，以确保父类被正确初始化。
                                        # 当在子类中定义__init__方法时，通常需要调用父类的__init__方法来初始化从父类继承来的属性。
                                        # 使用super()可以避免直接使用父类的名字，这在多重继承中特别有用。

        self.h = h

        # 1. 第一组卷积（convs1）：这部分创建了一个 模块列表（ModuleList），里面放了多个一维卷积层。
        # 使用不同的空洞率扩展感受野（捕捉不同尺度特征）
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                # weight_norm()是 PyTorch 的权重归一化函数（来自 torch.nn.utils.weight_norm）。
                # 它对卷积层权重进行归一化，有助于训练更稳定。返回一个新的卷积层对象。
                    Conv1d(             # 每层卷积的参数：
                        channels,       # in_channels
                        channels,       # out_channels
                        kernel_size,    # 卷积核大小
                        stride=1,       # 步长
                        dilation=d,     # 膨胀因子（例如 1, 3, 5）
                        padding=get_padding(kernel_size, d),        # 自动计算填充以保持输入输出长度一致
                        # get_padding(kernel_size, dilation) 用于计算保持输入输出长度一致所需的 padding。返回一个整数填充大小。
                    )
                )
                for d in dilation       # for 循环会遍历传入的 dilation 元组，比如 (1, 3, 5)
                                        # 每次循环：创建一个 空洞卷积层（dilation=d），用 weight_norm() 包装，加入到列表中
                                        # 于是：dilation = (1, 3, 5) 会创建出三层卷积：
                                            # convs1[0]	1	普通卷积                第一层提取局部特征（dilation = 1）
                                            # convs1[1]	3	空洞卷积（扩大感受野）     第二层提取中等范围特征（dilation = 3）
                                            # convs1[2]	5	空洞卷积（感受野更大）     第三层提取长程依赖（dilation = 5）
                                            # 每一层都带残差连接，使得信息在整个模块中顺畅流动。
                                        # 最终 self.convs1 就是一个 ModuleList，保存这几层卷积。
            ]
        )
        self.convs1.apply(init_weights)
        # init_weights()：初始化权重的函数（例如正态分布或 Xavier 初始化），通过 self.convs1.apply(init_weights) 对每层卷积调用初始化函数。

        # 2. 第二组卷积（convs2），第二组卷积与第一组对应，每个空洞卷积后接一个普通卷积。
        # 其作用是：形成一对 (空洞卷积 → 普通卷积) 的子模块，构成一个完整的残差子块。————对卷积输出进行融合与调整（稳定输出特征）
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))       # 循环次数与上面 dilation 的数量相同，但不关心具体的 dilation 值。
                                                    # convs2 作为每个残差单元的第二层卷积，并不需要不同的膨胀率（它固定为 dilation=1）。但是每个 convs1 对应一个 convs2，所以数量要一致。
                                                    # len(dilation) == len(self.convs1) == len(self.convs2)
            ]
        )
        self.convs2.apply(init_weights)

        # 循环部分	                        循环对象	                    功能	                            dilation设置
        # for d in dilation	                dilation = (1,3,5)	        创建多层不同膨胀率卷积	            分别为 1, 3, 5
        # for _ in range(len(dilation))	    不关心数值，只循环次数	        创建与上面对应数量的卷积层	        全部固定为 1

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers



        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        # 3. 动态加载 CUDA 版本，根据配置文件决定是否使用 GPU 优化版的激活函数。
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import \
                Activation1d as CudaActivation1d

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d



        # Activation functions
        # 4. 根据配置选择激活函数类型
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )
        # Snake / SnakeBeta
        # 这是“别名自由（alias-free）”激活函数的特殊版本，常用于音频生成任务。与 ReLU 不同，它是周期性激活函数，能更好地捕捉波形特征。
        # 一般形式：Snake(x) = x + (1 / α) * sin²(αx)，其中 α 是可学习参数。



    # 5. 前向传播
    # 5.1. 将激活函数分成两组（每两层卷积对应两个激活）
    # 5.2. 对输入 x：
    # 先经过激活 a1 → 卷积 c1 → 激活 a2 → 卷积 c2，再加上残差 x = xt + x，形成残差连接（Residual Connection）
    # 5.3. 输出增强后的特征。
    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
    # 残差连接，防止梯度消失、改善训练稳定性
        return x



        # 输入x
        # │
        # ├──► [激活 a1] ─► [卷积 c1, dilation = 1] ─► [激活 a2] ─► [卷积 c2, dilation = 1] ─┐
        # │                                                                               │
        # ├──► [激活 a3] ─► [卷积 c1, dilation = 3] ─► [激活 a4] ─► [卷积 c2, dilation = 1] ─┤
        # │                                                                               │
        # └──► [激活 a5] ─► [卷积 c1, dilation = 5] ─► [激活 a6] ─► [卷积 c2, dilation = 1] ─┤
        #                                                                                 ▼
        #                                                                               输出x


    # 6. 移除权重归一化
    # remove_weight_norm() 是 PyTorch 提供的配套函数。
    # 训练结束后，可以调用这个函数，去除所有卷积层的 weight_norm 包装，加快推理速度（因为权重归一化在推理阶段不再需要）。
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)



class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import \
                Activation1d as CudaActivation1d

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


'''
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
'''


class BigVGAN(
    torch.nn.Module,
):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (AttrDict): Hyperparameters.
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.

    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import \
                Activation1d as CudaActivation1d

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.feat_upsample = h.feat_upsample
        self.cond_in_each_up_layer = h.cond_d_vector_in_each_upsampling_layer

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.gpt_dim, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        self.speaker_encoder = ECAPA_TDNN(h.num_mels, lin_neurons=h.speaker_embedding_dim)
        self.cond_layer = nn.Conv1d(h.speaker_embedding_dim, h.upsample_initial_channel, 1)
        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = h.upsample_initial_channel // (2 ** (i + 1))
                self.conds.append(nn.Conv1d(h.speaker_embedding_dim, ch, 1))

    def forward(self, x, mel_refer, lens=None):
        # Speaker reference
        speaker_embedding = self.speaker_encoder(mel_refer, lens)
        n_batch = x.size(0)
        contrastive_loss = None
        if n_batch * 2 == speaker_embedding.size(0):
            spe_emb_chunk1, spe_emb_chunk2 = speaker_embedding[:n_batch, :, :], speaker_embedding[n_batch:, :, :]
            contrastive_loss = self.cal_clip_loss(spe_emb_chunk1.squeeze(1), spe_emb_chunk2.squeeze(1),
                                                  self.logit_scale.exp())

            speaker_embedding = speaker_embedding[:n_batch, :, :]
        speaker_embedding = speaker_embedding.transpose(1, 2)

        # upsample feat
        if self.feat_upsample:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),
                scale_factor=[4],
                mode="linear",
            ).squeeze(1)
        else:
            x = x.transpose(1, 2)

        # BigVGAN
        # Pre-conv
        x = self.conv_pre(x)
        x = x + self.cond_layer(speaker_embedding)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)

            if self.cond_in_each_up_layer:
                x = x + self.conds[i](speaker_embedding)

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x, contrastive_loss

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",  # Additional argument
        strict: bool = False,  # Additional argument
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="bigvgan_generator.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model
