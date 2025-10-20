# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
import re
from collections import OrderedDict

import torch
import yaml


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    checkpoint = torch.load(model_pth, map_location='cpu')
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(checkpoint, strict=True)
    info_path = re.sub('.pth$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_checkpoint(model: torch.nn.Module, model_pth: str, configs: dict = None):
    """
    保存模型权重和配置到文件。
    - 模型参数保存在 model_pth (例如 'checkpoint/gpt_epoch1.pth')
    - 配置信息（可选）保存在同名的 .yaml 文件中

    Args:
        model (torch.nn.Module): 要保存的模型
        model_pth (str): 保存的权重路径（.pth）
        configs (dict, optional): 需要保存的配置字典
    """
    # 1. 确保路径存在
    os.makedirs(os.path.dirname(model_pth), exist_ok=True)

    # 2. 保存模型权重
    torch.save({'model': model.state_dict()}, model_pth)
    print(f">> [save_checkpoint] Model weights saved to: {model_pth}")

    # 3. 保存配置到同名 .yaml
    if configs is not None:
        info_path = re.sub(r'\.pth$', '.yaml', model_pth)
        with open(info_path, 'w', encoding='utf-8') as fout:
            yaml.dump(configs, fout, allow_unicode=True)
        print(f">> [save_checkpoint] Config saved to: {info_path}")

