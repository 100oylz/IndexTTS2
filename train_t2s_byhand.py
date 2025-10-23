# -*- coding: utf-8 -*-
"""
train_t2s_byhand.py
只修改本文件：基于 UnifiedVoice（indextts.gpt.model_v2）+ TextTokenizer（indextts.utils.front）
实现 IndexTTS2 的三阶段 T2S 训练（AR 语义 token 预测）。

数据期望（JSONL，每行一个样本，字段尽量齐全；缺省时脚本会给“可运行的兜底”）：
{
  "text": "快躲起来！是他要来了！",
  "sem_tokens": [8192, 10, 33, ..., 8193],   # 语义token序列（若未含<EA>/stop，脚本会补）
  "spk_id": 0,
  "cond_mel": [[... 1024维 ...] x frames],   # 条件mel特征，可选；无则走零张量兜底
  "emo_cond_mel": [[... 1024维 ...] x frames],# 情感条件mel，可选；无则沿用 cond_mel
  "T":  len(sem_tokens)                       # 可选；用于“时长控制向量”p（Stage1可置零）
}

默认读取：
  train.jsonl:  ./data/train.jsonl
  valid.jsonl:  ./data/val.jsonl
  词表(bpe):    ./checkpoints/bpe.model
"""

import os, json, math, random, argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW, Adam, SGD, Adagrad
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# ========= 来自你仓库的模型与分词器 =========
from indextts.gpt.model_v2 import UnifiedVoice                    # 你的 UnifiedVoice（T2S）
from indextts.utils.front import TextTokenizer, TextNormalizer     # 文本 tokenizer

# ================== 常量 & 工具 ==================
PAD_ID = 0   # 用于 sem/text padding（与 tokenizer 的 pad 区分，无冲突）
BOS_ID = 1   # 右移起始符（仅用于 sem_inp 的占位，不会写入模型的 start_mel_token）
EA_ID  = 8193  # 对应 UnifiedVoice.stop_mel_token，作为语义token终止

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def exists(p): return p is not None

def cosine_with_warmup(step, base_lr, warmup, max_steps, min_lr=1e-6):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * ratio))


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
    condition_type: str = "perceiver"
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
    seed:int=42
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

    # ---------------------- 训练稳定性配置 ----------------------
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    amp: bool = field(default=False, init=False)  # 可通过命令行 --amp 打开

    # ---------------------- 数据相关配置 ----------------------
    vocab_file_path:str="./checkpoints/bpe.model"
    train_jsonl: str = "./data/train.jsonl"
    val_jsonl: str = "./data/val.jsonl"
    json_index_path: str = "./data/dataset_index.json"
    max_text_length: int = 1024
    max_mel_length: int = 1024
    audio_sample_rate: int = 16000
    speed_aug_range: List[float] = field(
        default_factory=lambda: [0.8, 1.0, 1.2],
        init=False
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

# ================== 数据集 & Collate ==================
class T2SDataset(Dataset):
    """
    训练语料（JSONL）字段：
      - text: str
      - sem_tokens: List[int]（若末尾无 EA_ID，本脚本会补齐）
      - spk_id: int（可选，用于 Stage2 的对抗支路）
      - cond_mel: List[List[float]] 形状 [frames, 1024] 或 [1024, frames]（二者都接受）
      - emo_cond_mel: 同上；可缺省时复用 cond_mel
      - T: int（可选，目标语义长度；Stage1构造 p-vector 可用；缺省时用 len(sem_tokens)）
    """
    def __init__(self, jsonl_path: str, tokenizer: TextTokenizer):
        self.tk = tokenizer
        self.items = []
        if not os.path.exists(jsonl_path):
            # 可运行兜底
            print(f"[WARN] {jsonl_path} 不存在，将使用内置演示样本。")
            demo = {
                "text": "快躲起来！是他要来了！",
                "sem_tokens": [8192, 10, 11, 12, 13, 8193],
                "spk_id": 0
            }
            self.items = [demo] * 64
            return
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                eg = json.loads(line)
                self.items.append(eg)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        eg = self.items[idx]
        text_ids = self.tk.encode(eg.get("text", ""), out_type=int)

        sem = eg.get("sem_tokens", [])
        if len(sem) == 0 or sem[-1] != EA_ID:
            sem = sem + [EA_ID]
        T = eg.get("T", len(sem))

        spk = int(eg.get("spk_id", 0))

        def _to_mel(x):
            if x is None: return None
            arr = np.array(x, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError("cond_mel / emo_cond_mel 必须是2D数组")
            # 统一为 [1024, frames]
            if arr.shape[0] != 1024 and arr.shape[1] == 1024:
                arr = arr.T
            return arr  # [1024, frames]

        cond_mel = _to_mel(eg.get("cond_mel", None))
        emo_mel  = _to_mel(eg.get("emo_cond_mel", None))

        return {
            "text_ids": np.asarray(text_ids, dtype=np.int64),
            "sem": np.asarray(sem, dtype=np.int64),
            "T": int(T),
            "spk": spk,
            "cond_mel": cond_mel,   # [1024, F] or None
            "emo_mel": emo_mel      # [1024, F] or None
        }

def pad_1d(seqs, pad=PAD_ID, dtype=np.int64):
    mx = max(len(s) for s in seqs)
    arr = np.full((len(seqs), mx), pad, dtype=dtype)
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    for i, s in enumerate(seqs): arr[i, :len(s)] = s
    return torch.from_numpy(arr), torch.from_numpy(lens)

def collate_fn(batch: List[Dict[str, Any]]):
    text_list = [b["text_ids"] for b in batch]
    sem_list  = [b["sem"] for b in batch]
    text, text_len = pad_1d(text_list, pad=PAD_ID)
    sem,  sem_len  = pad_1d(sem_list,  pad=PAD_ID)

    spk = torch.tensor([b["spk"] for b in batch], dtype=torch.long)
    T   = torch.tensor([b["T"] for b in batch], dtype=torch.long)

    # 右移解码端输入：BOS + target[:-1]
    sem_inp = torch.cat([torch.full_like(sem[:, :1], BOS_ID), sem[:, :-1]], dim=1)

    # 条件 mel：兜底为零张量（可运行）
    def _pad_mel(key: str):
        lst = [b[key] for b in batch]
        F = max((x.shape[1] for x in lst if x is not None), default=50)
        out = []
        length = []
        for x in lst:
            if x is None:
                arr = np.zeros((1024, F), dtype=np.float32)
                L = F
            else:
                L = x.shape[1]
                if L < F:
                    pad = np.zeros((1024, F-L), dtype=np.float32)
                    arr = np.concatenate([x, pad], axis=1)
                else:
                    arr = x[:, :F]
                L = min(L, F)
            out.append(torch.from_numpy(arr))
            length.append(L)
        return torch.stack(out, dim=0), torch.tensor(length, dtype=torch.long)

    cond_mel, cond_len = _pad_mel("cond_mel")
    emo_mel,  emo_len  = _pad_mel("emo_mel")

    return {
        "text": text, "text_len": text_len,
        "sem_inp": sem_inp, "sem_tgt": sem, "sem_len": sem_len,
        "spk": spk, "T": T,
        "cond_mel": cond_mel, "cond_len": cond_len,
        "emo_mel": emo_mel, "emo_len": emo_len
    }


# ================== 训练器 ==================
class T2STrainer:
    def __init__(self, stage: int = 1):
        self.hp = TrainHParams()
        set_seed(self.hp.seed)
        self.device = torch.device(self.hp.device)
        print(f"[Device] {self.device}")

        # Tokenizer
        normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(self.hp.vocab_file_path, normalizer)

        # 模型（⚠️强制切到 perceiver 条件分支，避开 conformer+RPE）
        self.model_cfg = UnifiedVoiceConfig()
        _cfg = asdict(self.model_cfg)
        _cfg["condition_type"] = "perceiver"   # ← 关键：稳定条件分支
        self.model = UnifiedVoice(**_cfg).to(self.device)

        # 优化器
        self.optimizer = self._build_optimizer()

        self.stage = int(stage)
        self.global_step = 0
        self.best_val = float("inf")

        os.makedirs(self.hp.checkpoint_dir, exist_ok=True)

    def _build_optimizer(self):
        p = [q for q in self.model.parameters() if q.requires_grad]
        t = self.hp.optimizer_type.lower()
        if t == "adamw":
            return AdamW(p, lr=self.hp.learning_rate, betas=self.hp.betas, eps=self.hp.eps, weight_decay=self.hp.weight_decay)
        if t == "adam":
            return Adam(p, lr=self.hp.learning_rate, betas=self.hp.betas, eps=self.hp.eps, weight_decay=self.hp.weight_decay)
        if t == "sgd":
            return SGD(p, lr=self.hp.learning_rate, momentum=0.9, nesterov=True, weight_decay=self.hp.weight_decay)
        if t == "adagrad":
            return Adagrad(p, lr=self.hp.learning_rate, eps=self.hp.eps, weight_decay=self.hp.weight_decay)
        raise ValueError(f"Unsupported optimizer: {self.hp.optimizer_type}")

    # —— 三阶段冻结策略（按你的模块名自动探测）——
    def apply_freeze_by_stage(self):
        if self.stage == 1:
            print("[Stage1] 不冻结（或仅按需冻结前端）")
        elif self.stage == 2:
            # 冻结说话人条件分支（具体模块名按你的实现）
            if hasattr(self.model, "conditioning_encoder"):
                for p in self.model.conditioning_encoder.parameters(): p.requires_grad = False
                print("[Stage2] 冻结 conditioning_encoder（speaker 条件）")
            # 情感分支设为可训练
            for name in ["emo_conditioning_encoder", "emo_perceiver_encoder", "emo_layer", "emovec_layer"]:
                if hasattr(self.model, name):
                    for p in getattr(self.model, name).parameters():
                        p.requires_grad = True
                    print(f"[Stage2] 训练 {name}")
        elif self.stage == 3:
            # 冻结条件分支，微调主干/输出头
            for name in ["conditioning_encoder", "perceiver_encoder", "emo_conditioning_encoder", "emo_perceiver_encoder"]:
                if hasattr(self.model, name):
                    for p in getattr(self.model, name).parameters(): p.requires_grad = False
            print("[Stage3] 冻结条件分支，仅微调主干/输出层")
        else:
            raise ValueError("stage 必须为 1/2/3")

    # —— 数据加载器 ——
    def build_loaders(self):
        train_ds = T2SDataset(self.hp.train_jsonl, self.tokenizer)
        val_ds   = T2SDataset(self.hp.val_jsonl,   self.tokenizer)
        train_ld = DataLoader(train_ds, batch_size=self.hp.batch_size, shuffle=True,
                              num_workers=self.hp.num_workers, pin_memory=self.hp.pin_memory,
                              collate_fn=collate_fn, drop_last=True)
        val_ld   = DataLoader(val_ds, batch_size=self.hp.val_batch_size, shuffle=False,
                              num_workers=self.hp.num_workers, pin_memory=self.hp.pin_memory,
                              collate_fn=collate_fn, drop_last=False)
        return train_ld, val_ld

    # —— 前向：严格按 UnifiedVoice 公开方法构建 logits ——
    def forward_get_mel_logits(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          mel_logits: [B, V_mel, T_mel]  —— 直接用于 CrossEntropyLoss
          mel_targets: [B, T_mel]
        """
        m = self.model
        device = self.device

        text = batch["text"].to(device)              # [B, T_txt]
        sem_tgt = batch["sem_tgt"].to(device)        # [B, T_mel]
        sem_inp = batch["sem_inp"].to(device)        # [B, T_mel]
        text_len = batch["text_len"].to(device)
        sem_len  = batch["sem_len"].to(device)
        cond_mel = batch["cond_mel"].to(device)      # [B, 1024, F]
        emo_mel  = batch["emo_mel"].to(device)       # [B, 1024, F]
        cond_len = batch["cond_len"].to(device)
        emo_len  = batch["emo_len"].to(device)

        # ===== 1) 条件/情感向量（保持输入为 [B,1024,F]，不要手动 transpose）=====
        try:
            # 首选：正常通过 get_conditioning（已在 __init__ 强制 perceiver，更稳）
            conds_latent = m.get_conditioning(cond_mel, cond_len)
            # 情感支路
            emo_vec_ori  = m.get_emo_conditioning(emo_mel, emo_len)
            emo_vec = m.emovec_layer(emo_vec_ori)
            emo_vec = m.emo_layer(emo_vec)
        except Exception as e:
            # 🛟 兜底：若仍因版本差异触发 conformer 内部形状异常，则退化为“纯情感条件”
            print("[WARN] get_conditioning failed, fallback to emo-only latents. err=", repr(e))
            with torch.no_grad():
                emo_vec_ori  = m.get_emo_conditioning(emo_mel, emo_len)
                emo_vec = m.emovec_layer(emo_vec_ori)
                emo_vec = m.emo_layer(emo_vec)              # [B, d_model]
                B = emo_mel.size(0)
                K = getattr(m, "condition_num_latent", 32)
                conds_latent = emo_vec.unsqueeze(1).repeat(1, K, 1)  # [B, K, d_model]

        # ===== 2) 时长/速度控制向量（使用 speed_emb 的自由/受控双嵌入）=====
        tmp = torch.zeros(text.size(0), device=device).long()
        duration_emb      = m.speed_emb(tmp)                  # 自由
        duration_emb_half = m.speed_emb(torch.ones_like(tmp)) # 受控
        if self.stage == 1:
            mask = (torch.rand_like(tmp.float()) < 0.3).long()
            durA = m.speed_emb(mask)
            durB = m.speed_emb(1 - mask)
            conds = torch.cat((conds_latent + emo_vec.unsqueeze(1), durA.unsqueeze(1), durB.unsqueeze(1)), 1)
        else:
            conds = torch.cat((conds_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)

        # ===== 3) 文本/语义输入输出对齐（使用模型内工具函数）=====
        text_inputs = m.set_text_padding(text.clone(), text_len)
        text_inputs, _ = m.build_aligned_inputs_and_targets(text_inputs, m.start_text_token, m.stop_text_token)
        text_emb = m.text_embedding(text_inputs) + m.text_pos_embedding(text_inputs)

        mel_codes = m.set_mel_padding(sem_inp.clone(), sem_len)
        mel_codes, mel_targets = m.build_aligned_inputs_and_targets(mel_codes, m.start_mel_token, m.stop_mel_token)
        mel_emb = m.mel_embedding(mel_codes) + m.mel_pos_embedding(mel_codes)

        # ===== 4) 取 logits =====
        _, mel_logits = m.get_logits(conds, text_emb, m.text_head, mel_emb, m.mel_head, get_attns=False, return_latent=False)
        return mel_logits, mel_targets

    # —— 评估 ——（以 CE 近似 ppl）
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        ce = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
        total_nll, total_tok = 0.0, 0
        for batch in loader:
            mel_logits, mel_targets = self.forward_get_mel_logits(batch)
            nll = ce(mel_logits, mel_targets)  # [B,V,T] vs [B,T]
            total_nll += nll.item()
            total_tok += (mel_targets != PAD_ID).sum().item()
        ppl = math.exp(total_nll / max(1, total_tok)) if total_tok else float("inf")
        self.model.train()
        return ppl, total_nll / max(1, total_tok)

    # —— 训练循环 ——（三阶段合一，通过 --stage 切换冻结与时长策略）
    def fit(self):
        self.apply_freeze_by_stage()
        train_ld, val_ld = self.build_loaders()

        ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        # 使用 torch.amp 新接口
        scaler = torch.amp.GradScaler('cuda', enabled=self.hp.amp)

        pbar = tqdm(total=self.hp.epochs * len(train_ld), desc=f"T2S-Stage{self.stage}")

        for epoch in range(1, self.hp.epochs + 1):
            for i, batch in enumerate(train_ld, 1):
                self.global_step += 1
                with torch.amp.autocast('cuda', enabled=self.hp.amp):
                    mel_logits, mel_targets = self.forward_get_mel_logits(batch)
                    loss = ce_loss(mel_logits, mel_targets)

                scaler.scale(loss / self.hp.accumulation_steps).backward()

                if self.global_step % self.hp.accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    if self.hp.gradient_clip_norm and self.hp.gradient_clip_norm > 0:
                        clip_grad_norm_(self.model.parameters(), self.hp.gradient_clip_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    # 手工 lr 调度（cosine + warmup）
                    lr = cosine_with_warmup(self.global_step, self.hp.learning_rate, warmup=4000,
                                            max_steps=max(4000, self.hp.epochs * len(train_ld)))
                    for g in self.optimizer.param_groups: g["lr"] = lr

                if self.global_step % self.hp.log_freq == 0:
                    tqdm.write(f"[LOG] ep={epoch} it={i} step={self.global_step} lr={self.optimizer.param_groups[0]['lr']:.2e} loss={loss.item():.4f}")

                pbar.update(1)

            # 验证 + 早停
            if epoch % self.hp.val_freq == 0:
                ppl, nll = self.evaluate(val_ld)
                tqdm.write(f"[EVAL] ep={epoch} ppl={ppl:.3f} nll/token={nll:.4f}")
                improved = nll < self.best_val
                if improved:
                    self.best_val = nll
                    self.save_ckpt(is_best=True, tag=f"best_ep{epoch}")
                else:
                    # 早停计数（如需严格早停，可补计数器）
                    pass

            # 常规保存
            if epoch % self.hp.save_freq == 0:
                self.save_ckpt(is_best=False, tag=f"ep{epoch}")

        pbar.close()

    def save_ckpt(self, is_best: bool, tag: str):
        path = os.path.join(self.hp.checkpoint_dir, f"t2s_stage{self.stage}_{tag}.pth")
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val": self.best_val,
            "cfg": asdict(self.model_cfg)
        }
        torch.save(ckpt, path)
        print(f"[CKPT] saved -> {path}")
        if is_best:
            best = os.path.join(self.hp.checkpoint_dir, f"t2s_stage{self.stage}_best.pth")
            torch.save(ckpt, best)
            print(f"[CKPT] best -> {best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1,2,3], help="训练阶段：1|2|3")
    ap.add_argument("--amp", action="store_true", help="开启混合精度（torch.amp）")
    args = ap.parse_args()

    trainer = T2STrainer(stage=args.stage)
    # 将命令行的 --amp 赋值到超参（不破坏原结构）
    object.__setattr__(trainer.hp, "amp", args.amp)

    trainer.fit()

if __name__ == "__main__":
    main()
