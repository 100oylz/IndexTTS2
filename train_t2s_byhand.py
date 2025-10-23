import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict, field, FrozenInstanceError
from typing import Optional, List, Tuple, Dict
from torch.optim import Adam, AdamW, SGD, Adagrad
import numpy as np

# 假设以下导入有效
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import  TextNormalizer,TextTokenizer

# ---------------------- 子配置类（保持不变） ----------------------
@dataclass
class ConditionModules:
    output_size: int = 256
    linear_units: int = 2048
    attention_heads: int = 8
    num_blocks: int = 6
    input_layer: str = "conv2d"
    perceiver_mult: int = 4


@dataclass
class EmoConditionModule:
    output_size: int = 256
    linear_units: int = 2048
    attention_heads: int = 8
    num_blocks: int = 6
    input_layer: str = "conv2d"
    perceiver_mult: int = 4


# ---------------------- 模型配置类（保持不变） ----------------------
@dataclass
class UnifiedVoiceConfig:
    layers: int = 8
    model_dim: int = 512
    heads: int = 8
    max_text_tokens: int = 1024
    max_mel_tokens: int = 1024
    max_conditioning_inputs: int = 1
    mel_length_compression: int = 1024
    start_text_token: int = 0
    stop_text_token: int = 1
    number_mel_codes: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    train_solo_embeddings: bool = False
    use_mel_codes_as_input: bool = True
    checkpointing: bool = True
    types: int = 1
    condition_num_latent: int = 32
    condition_type: str = "conformer_perceiver"
    condition_module: dict = field(
        default_factory=lambda: asdict(ConditionModules())
    )
    emo_condition_module: dict = field(
        default_factory=lambda: asdict(EmoConditionModule())
    )


# ---------------------- 训练超参数类（冻结，不可修改） ----------------------
@dataclass(frozen=True)  # frozen=True 使类实例不可修改
class TrainHParams:
    """神经网络训练超参数配置类（适配T2S场景）- 冻结不可修改"""
    # ---------------------- 基础训练配置 ----------------------
    epochs: int = 100
    batch_size: int = 32
    val_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    seed:int =42
    # device 用 default_factory 且冻结，确保自动检测后不可修改
    device: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu",
        init=False  # 禁止实例化时手动传入（强制自动检测）
    )

    # ---------------------- 优化器配置 ----------------------
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    # ---------------------- 学习率调度配置（直接移除，不保留无效参数） ----------------------
    # （已忽略学习率递减，故删除所有调度器相关字段）

    # ---------------------- 训练稳定性配置 ----------------------
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    amp: bool = field(default=False, init=False)  # 强制禁用混合精度，不可修改

    # ---------------------- 数据相关配置 ----------------------
    vocab_file_path:str="./checkpoints/bpe.model"
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    json_index_path: str = "./data/dataset_index.json"
    max_text_length: int = 1024
    max_mel_length: int = 1024
    audio_sample_rate: int = 16000
    speed_aug_range: List[float] = field(
        default_factory=lambda: [0.8, 1.0, 1.2],
        init=False  # 禁止实例化时修改增强范围
    )

    # ---------------------- 正则化配置 ----------------------
    dropout_rate: float = 0.1

    # ---------------------- 日志与保存配置 ----------------------
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_freq: int = 1
    save_best_only: bool = True
    log_freq: int = 100
    val_freq: int = 5

    # ---------------------- 早停配置 ----------------------
    early_stop: bool = True
    early_stop_patience: int = 20
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"


class T2STrainer:
    def __init__(self):
        # 1. 加载训练超参数（冻结，不可修改）和模型配置
        self.train_hparams = TrainHParams()  # 只能使用默认值，无法修改
        self.model_config = UnifiedVoiceConfig()
        text_normalizer=TextNormalizer()
        self.text_tokenizer=TextTokenizer(self.train_hparams.vocab_file_path,text_normalizer)
        # 2. 固定随机种子（保证训练可复现）
        self._set_seed(self.train_hparams.seed)

        # 3. 初始化训练设备（设备已由TrainHParams自动检测，不可修改）
        self.device = torch.device(self.train_hparams.device)
        print(f"训练设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # 4. 初始化模型并移至设备
        self.model = UnifiedVoice(**asdict(self.model_config)).to(self.device)
        print(f"模型初始化完成，参数总数: {self._count_model_params():.2f}M")

        # 5. 构建优化器（无学习率调度器）
        self.optimizer = self._build_optimizer()
        print(f"优化器: {self.train_hparams.optimizer_type}")
        print("学习率调度器: 未启用（已忽略）")
        print(f"混合精度训练: 未启用（已忽略）")

        # 6. 创建日志和checkpoint保存目录（确保路径存在）
        self._create_save_dirs()

        # 7. 初始化早停相关变量
        self.early_stop_counter = 0  # 早停计数器
        self.best_metric = float("inf") if self.train_hparams.monitor_mode == "min" else -float("inf")  # 最佳指标

        # 8. 初始化训练状态变量
        self.current_epoch = 0
        self.global_step = 0

    def _set_seed(self, seed: int):
        """固定随机种子，保证训练可复现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 多GPU场景
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭自动优化，保证确定性
        print(f"随机种子已固定: {seed}")

    def _count_model_params(self) -> float:
        """统计模型参数量（单位：百万）"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params / 1e6

    def _build_optimizer(self):
        """根据超参数构建优化器（无调度器）"""
        params = self.model.parameters()
        hparams = self.train_hparams

        if hparams.optimizer_type.lower() == "adam":
            return Adam(
                params,
                lr=hparams.learning_rate,
                betas=hparams.betas,
                eps=hparams.eps,
                weight_decay=hparams.weight_decay
            )
        elif hparams.optimizer_type.lower() == "adamw":
            return AdamW(
                params,
                lr=hparams.learning_rate,
                betas=hparams.betas,
                eps=hparams.eps,
                weight_decay=hparams.weight_decay
            )
        elif hparams.optimizer_type.lower() == "sgd":
            return SGD(
                params,
                lr=hparams.learning_rate,
                momentum=0.9,
                weight_decay=hparams.weight_decay,
                nesterov=True
            )
        elif hparams.optimizer_type.lower() == "adagrad":
            return Adagrad(
                params,
                lr=hparams.learning_rate,
                eps=hparams.eps,
                weight_decay=hparams.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {hparams.optimizer_type}")

    def _create_save_dirs(self):
        """创建日志和checkpoint保存目录"""
        dirs = [self.train_hparams.log_dir, self.train_hparams.checkpoint_dir]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"创建目录: {dir_path}")
            else:
                print(f"目录已存在: {dir_path}")

    def load_pretrained(self, ckpt_path: str):
        """加载预训练权重（微调时使用）"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"成功加载预训练权重: {ckpt_path}")
        else:
            # 兼容直接保存的模型权重
            self.model.load_state_dict(checkpoint, strict=False)
            print(f"成功加载预训练权重（兼容模式）: {ckpt_path}")

        # 可选：加载优化器和训练状态（继续训练）
        if "optimizer_state_dict" in checkpoint and "epoch" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"] if "global_step" in checkpoint else 0
            self.best_metric = checkpoint["best_metric"] if "best_metric" in checkpoint else self.best_metric
            print(f"恢复训练状态:  epoch={self.current_epoch}, global_step={self.global_step}")

    def save_checkpoint(self, metric: float, is_best: bool = False):
        """保存模型checkpoint"""
        hparams = self.train_hparams
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "train_hparams": asdict(hparams),  # 冻结的配置仍可转换为字典保存
            "model_config": asdict(self.model_config)
        }

        # 保存普通checkpoint（按轮次）
        ckpt_name = f"epoch_{self.current_epoch}_step_{self.global_step}.pth"
        ckpt_path = os.path.join(hparams.checkpoint_dir, ckpt_name)
        torch.save(checkpoint, ckpt_path)
        print(f"保存checkpoint: {ckpt_path}")

        # 保存最佳模型
        if is_best:
            best_ckpt_path = os.path.join(hparams.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_ckpt_path)
            print(f"更新最佳模型: {best_ckpt_path}")

    def update_early_stop(self, current_metric: float) -> bool:
        """更新早停状态，返回是否需要停止训练"""
        hparams = self.train_hparams
        if not hparams.early_stop:
            return False

        # 判断当前指标是否优于最佳指标
        if hparams.monitor_mode == "min":
            is_improved = current_metric < self.best_metric
        else:
            is_improved = current_metric > self.best_metric

        if is_improved:
            self.best_metric = current_metric
            self.early_stop_counter = 0  # 重置计数器
            return False
        else:
            self.early_stop_counter += 1
            print(f"早停计数器: {self.early_stop_counter}/{hparams.early_stop_patience}")
            if self.early_stop_counter >= hparams.early_stop_patience:
                print(f"早停触发！连续{hparams.early_stop_patience}轮无指标提升")
                return True
        return False

    def train(self,text,stage):
        tokenized_text=self.text_tokenizer.batch_encode(text)
        print(tokenized_text)

        if(stage==1):
            pass
        elif(stage==2):
            pass
        elif(stage==3):
            pass
        else:
            raise ValueError("Stage Must be 1 or 2 or 3")



if __name__ == '__main__':
    # 初始化训练器
    trainer = T2STrainer()

    # 验证TrainHParams不可修改（以下代码会抛出异常，注释后可正常运行）
    # try:
    #     trainer.train_hparams.batch_size = 16  # 尝试修改超参数
    # except FrozenInstanceError as e:
    #     print(f"\n验证：超参数不可修改，抛出异常: {e}")

    # 可选：加载预训练权重（微调场景）
    # trainer.load_pretrained("./pretrained/unified_voice_best.pth")
    test_text=["Hello,World!"]
    trainer.train(test_text,stage=1)
    print("\n训练器初始化完成，可开始训练流程")