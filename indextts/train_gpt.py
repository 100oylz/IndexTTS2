# train_gpt.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import save_checkpoint, load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
import random
import numpy as np

class GPTDataset(Dataset):
    """
    自定义 Dataset，用于 GPT 训练：
    每条数据包括：text_tokens, semantic_codes
    假设 semantic_codes 是一个整数序列（token ids 或 code indices）。
    """
    def __init__(self, list_meta, tokenizer: TextTokenizer, semantic_codes_dir: str, max_len=256):
        """
        list_meta: list of tuples (text, semantic_code_path)
        tokenizer: 文本 tokenizer 实例
        semantic_codes_dir: semantic codes 所在目录
        """
        self.list_meta = list_meta
        self.tokenizer = tokenizer
        self.semantic_codes_dir = semantic_codes_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.list_meta)

    def __getitem__(self, idx):
        text, code_filename = self.list_meta[idx]
        # 文本转 tokens
        token_ids = self.tokenizer.convert_text_to_ids(text)
        token_ids = token_ids[:self.max_len]
        # 加上 eos 或 padding 可选
        # 加载 semantic_codes
        code_path = os.path.join(self.semantic_codes_dir, code_filename)
        # 假设 semantic_codes 保存为 numpy 或 torch tensor
        codes = torch.load(code_path) if code_path.endswith('.pt') else torch.from_numpy(np.load(code_path))
        # 你可能需要截断或 pad codes 到固定长度
        return {
            "text_ids": torch.LongTensor(token_ids),
            "code_ids": codes.long()
        }

def collate_fn(batch):
    # batch 为 list of dict
    text_ids = [item["text_ids"] for item in batch]
    code_ids = [item["code_ids"] for item in batch]
    # pad text_ids 和 code_ids
    text_ids_pad = nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0)
    code_ids_pad = nn.utils.rnn.pad_sequence(code_ids, batch_first=True, padding_value=0)
    # 长度
    text_lens = torch.LongTensor([len(x) for x in text_ids])
    code_lens = torch.LongTensor([len(x) for x in code_ids])
    return {
        "text_ids": text_ids_pad,
        "text_lens": text_lens,
        "code_ids": code_ids_pad,
        "code_lens": code_lens
    }

def main():
    # 加载配置
    cfg = OmegaConf.load("checkpoints/config.yaml")  # 或者 e.g. "configs/gpt_train.yaml"
    gpt_cfg = cfg.gpt  # 假设配置中有 gpt 部分

    # 设置 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 实例化 tokenizer & normalizer （根据你原代码）
    normalizer = TextNormalizer()
    normalizer.load()
    tokenizer = TextTokenizer(os.path.join(cfg.dataset["bpe_model_path"]), normalizer)

    # 构建数据列表（这里是示例，你需要自己实现 meta 生成）
    # 假设：metadata.txt 每行：<code_filename>\t<text>
    meta_path = cfg.dataset["gpt_meta"]  # e.g. "data/gpt_meta.txt"
    semantic_codes_dir = cfg.dataset["semantic_codes_dir"]
    list_meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            code_fn, text = line.strip().split("\t", 1)
            list_meta.append((text, code_fn))

    dataset = GPTDataset(list_meta, tokenizer, semantic_codes_dir, max_len=gpt_cfg.max_text_tokens)
    dataloader = DataLoader(dataset,
                            batch_size=gpt_cfg.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=gpt_cfg.num_workers,
                            drop_last=True)

    # 初始化模型
    model = UnifiedVoice(**gpt_cfg.model_params)
    model = model.to(device)
    if gpt_cfg.pretrained_checkpoint is not None:
        load_checkpoint(model, gpt_cfg.pretrained_checkpoint, load_only_params=True)
        print(f">> Loaded pretrained GPT checkpoint from {gpt_cfg.pretrained_checkpoint}")

    # 优化器 & 学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=gpt_cfg.learning_rate, weight_decay=gpt_cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=gpt_cfg.lr_step_size, gamma=gpt_cfg.lr_gamma)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 根据你的输出设定

    global_step = 0
    model.train()

    for epoch in range(gpt_cfg.max_epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for batch in dataloader:
            text_ids = batch["text_ids"].to(device)
            text_lens = batch["text_lens"].to(device)
            code_ids = batch["code_ids"].to(device)
            code_lens = batch["code_lens"].to(device)

            optimizer.zero_grad()
            # 假设 model 返回 logits：[B, T, V] 对 semantic code vocab 预测
            logits = model(text_ids, text_lens, code_ids, code_lens)
            # shift targets if needed
            target = code_ids[:, 1:].contiguous()
            pred = logits[:, :-1, :].contiguous()

            loss = criterion(pred.view(-1, pred.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % gpt_cfg.log_steps == 0:
                print(f"[Epoch {epoch} Step {global_step}] loss = {loss.item():.4f}")

            if global_step % gpt_cfg.save_steps == 0:
                ckpt_path = os.path.join(gpt_cfg.output_dir, f"gpt_ckpt_step{global_step}.pt")
                save_checkpoint(model, optimizer, global_step, ckpt_path)
                print(f">> Saved checkpoint to {ckpt_path}")

        t1 = time.time()
        print(f"Epoch {epoch} finished in {t1-t0:.2f} s, avg loss = {epoch_loss / len(dataloader):.4f}")

        # End epoch checkpoint
        ckpt_path = os.path.join(gpt_cfg.output_dir, f"gpt_ckpt_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, global_step, ckpt_path)
        print(f">> Saved epoch checkpoint to {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()