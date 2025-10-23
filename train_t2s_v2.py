# train_t2s_with_indextts_gptv2.py
"""
Training script that tries to use index-tts's indextts.gpt.model_v2.UnifiedVoice as
the autoregressive semantic generator. If import/interface mismatch, falls back to
a local GPTSemanticGenerator implementation so the training loop remains runnable.

Usage:
  - Put your train.txt in the same folder (format: each line "utt_id <text>" or "<text>")
  - If you cloned index-tts locally, ensure PYTHONPATH includes the repo root, or install it.
  - Run: python train_t2s_with_indextts_gptv2.py
"""

import os
import random
import time
import math
import json
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Hyperparameters (tweak as needed)
# -----------------------
class HParams:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 256            # D
    max_T = 64                 # maximum supported semantic sequence length (for duration one-hot)
    alpha = 0.5                # adversarial loss coefficient
    batch_size = 8
    lr = 1e-4
    epochs_per_stage = 3       # small for debugging
    n_speakers = 100
    vocab_min_freq = 1
    save_dir = "./checkpoints"
    seed = 42

hp = HParams()
random.seed(hp.seed)
torch.manual_seed(hp.seed)
os.makedirs(hp.save_dir, exist_ok=True)

# -----------------------
# GradReverse (for adversarial training)
# -----------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, l=1.0):
    return GradReverse.apply(x, l)

# -----------------------
# Vocab builder (character-level default)
# -----------------------
def build_vocab_from_text(txt_path, add_special_tokens=True):
    chars = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            text = parts[1] if len(parts) == 2 else parts[0]
            for ch in text:
                chars[ch] = chars.get(ch, 0) + 1
    tokens = [ch for ch, _ in sorted(chars.items()) if chars[ch] >= hp.vocab_min_freq]
    vocab = {}
    idx = 0
    if add_special_tokens:
        for t in ["<pad>", "<unk>", "<bos>", "<eos>"]:
            vocab[t] = idx
            idx += 1
    for ch in tokens:
        if ch in vocab:
            continue
        vocab[ch] = idx
        idx += 1
    print(f"[Vocab] built vocab size = {len(vocab)}")
    return vocab

# -----------------------
# Dataset
# -----------------------
class T2SDataset(Dataset):
    def __init__(self, txt_path, vocab, max_T=hp.max_T, embed_dim=hp.embed_dim):
        assert os.path.exists(txt_path), f"{txt_path} not found"
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_T = max_T
        self.embed_dim = embed_dim
        self.samples = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                text = parts[1] if len(parts) == 2 else parts[0]
                token_ids = [vocab.get(ch, vocab.get("<unk>")) for ch in text]
                max_body = max(1, self.max_T - 2)
                token_ids = token_ids[:max_body]
                self.samples.append(token_ids)
        print(f"[Dataset] loaded {len(self.samples)} samples from {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        body = self.samples[idx]
        bos = self.vocab["<bos>"]
        eos = self.vocab["<eos>"]
        seq = [bos] + body + [eos]
        seq = [t if 0 <= t < self.vocab_size else self.vocab["<unk>"] for t in seq]
        sem_target = torch.tensor(seq, dtype=torch.long)
        T = max(1, len(body))   # used for duration control
        # placeholders for encoders/prompts; in real use, replace with real encoder outputs
        Etext = torch.randn(self.embed_dim)
        spk_prompt = torch.randn(self.embed_dim)
        emo_prompt = torch.randn(self.embed_dim)
        return Etext, sem_target, spk_prompt, emo_prompt, T

def t2s_collate_fn(batch):
    Etexts, sem_targets, spk_prompts, emo_prompts, Ts = zip(*batch)
    Etexts = torch.stack(Etexts)
    spk_prompts = torch.stack(spk_prompts)
    emo_prompts = torch.stack(emo_prompts)
    Ts = torch.tensor(Ts, dtype=torch.long)
    max_len = max([s.shape[0] for s in sem_targets])
    # For GPT-style teacher forcing we keep full sequences padded with pad_id 0
    pad_id = 0
    padded = torch.full((len(batch), max_len), fill_value=pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, s in enumerate(sem_targets):
        L = s.shape[0]
        padded[i, :L] = s
        attention_mask[i, :L] = 1
    # dec_input: teacher forcing inputs = padded[:, :-1]; dec_target = padded[:, 1:]
    dec_input = padded[:, :-1].contiguous()
    dec_target = padded[:, 1:].contiguous()
    attn_input_mask = attention_mask[:, :-1].contiguous()
    return Etexts, dec_input, dec_target, attn_input_mask, spk_prompts, emo_prompts, Ts



# -----------------------
# Simple local fallback GPTSemanticGenerator (used if UnifiedVoice unavailable)
# -----------------------
# 弃用
# 弃用
# 弃用
# 弃用
class GPTSemanticGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead=4, num_layers=4, max_len=hp.max_T):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.to_logits = nn.Linear(embed_dim, vocab_size)
        self.cond_proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, cond, input_ids, attention_mask):
        # cond: [B, 3D] -> project -> memory [1,B,D]
        B, L = input_ids.shape
        mem = self.cond_proj(cond).unsqueeze(0)  # [1,B,D]
        emb = self.token_embedding(input_ids) + self.pos_embedding[:, :L, :]
        emb = emb.transpose(0,1)  # [L,B,D]
        # tgt_key_padding_mask expects True for positions that should be masked (i.e., pad)
        tgt_key_padding_mask = (attention_mask == 0)
        out = self.decoder(emb, mem, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.to_logits(out.transpose(0,1))  # [B,L,V]
        return logits
# 弃用
# 弃用
# 弃用
# 弃用
# 弃用



# -----------------------
# Model wrapper that tries to use index-tts's UnifiedVoice
# T2SModelWrapper — 主模型封装类
# -----------------------
class T2SModelWrapper(nn.Module):
    def __init__(self, hp, vocab_size):
        super().__init__()
        D = hp.embed_dim
        self.hp = hp
        self.vocab_size = vocab_size

        # small encoders used as placeholders; in real pipeline replace with model_v2's encoders if desired
        self.text_encoder = nn.Linear(D, D)
        # TODO：包括在UnifiedVoice get_conditioning函数中
        self.spk_encoder = nn.Linear(D, D)
        # speaker classifier for adversarial loss (simple MLP)
        # TODO：GRL部分需要调整
        self.spk_classifier = nn.Sequential(nn.Linear(D, D//2), nn.ReLU(), nn.Linear(D//2, hp.n_speakers), nn.Softmax(dim=-1))
        # TODO：等待T2E模块
        self.emo_encoder = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))

        # duration control
        self.num_embed = nn.Linear(hp.max_T, D)
        self.Wnum = nn.Linear(D, D)

        # attempt to import UnifiedVoice from index-tts
        self.use_unifiedvoice = False
        # TODO：使用这个模型与本地仓库gpt2的unifiedvoice适配
        self.unified = None
        try:
            # try to import model_v2.UnifiedVoice
            from indextts.gpt.model_v2 import UnifiedVoice  # type: ignore
            # instantiate UnifiedVoice here if its __init__ has expected minimal args.
            # NOTE: actual UnifiedVoice signature may differ; adapt as needed.
            try:
                # try default instantiate (this may need config or checkpoint in real repo)
                self.unified = UnifiedVoice()
                self.use_unifiedvoice = True
                print("[T2SModelWrapper] Using indextts.gpt.model_v2.UnifiedVoice as generator.")
            except Exception as e_inner:
                print("[T2SModelWrapper] Imported UnifiedVoice but failed to instantiate without args.")
                print("Error:", e_inner)
                print("Falling back to local GPTSemanticGenerator.")
        except Exception as e:
            # import failed -> fallback
            print("[T2SModelWrapper] Could not import indextts.gpt.model_v2.UnifiedVoice. Falling back.")
            # print traceback for user
            # traceback.print_exc()

        # if UnifiedVoice not available or not instantiated, use local generator
        if not self.use_unifiedvoice:
            self.gpt = GPTSemanticGenerator(vocab_size, D, max_len=hp.max_T)
        else:
            self.gpt = None  # unified will be used

        # small projector for cond -> cond_dim if using local gpt
        self.cond_dim = D * 3
        if not self.use_unifiedvoice:
            self.cond_proj = nn.Linear(self.cond_dim, D)  # used by local GPT to make memory vector


    #TODO：根据stage参数控制stage 1，2，3
    def forward(self, Etext, dec_input, attn_input_mask, spk_prompt, emo_prompt, Ts, stage):
        """
        Etext: [B, D]  (placeholder or encoded text)
        dec_input: [B, L]  (teacher forcing input tokens)
        attn_input_mask: [B, L]  (1 for real token, 0 for pad)
        spk_prompt, emo_prompt: [B, D] placeholders
        Ts: [B] lengths
        stage: int
        """
        B = Etext.shape[0]
        device = Etext.device
        c = self.spk_encoder(spk_prompt)       # [B, D]
        e = self.emo_encoder(emo_prompt)       # [B, D]

        # duration control
        Ts_clamped = Ts.clamp(min=0, max=self.hp.max_T - 1)
        one_hot = F.one_hot(Ts_clamped, num_classes=self.hp.max_T).float().to(device)
        h_T = self.num_embed(one_hot)      # [B, D]
        p = self.Wnum(h_T)                 # [B, D]
        if stage == 1:
            mask = (torch.rand(B, device=device) < 0.3).float().unsqueeze(-1)
            p = p * (1.0 - mask)

        cond_first = (c + e) if stage >= 2 else c
        cond = torch.cat([cond_first, p, Etext], dim=-1)  # [B, 3D]

        # If using UnifiedVoice from index-tts, call its generation/forward API:
        if self.use_unifiedvoice and self.unified is not None:
            # NOTE: the real UnifiedVoice API likely requires config & more inputs (audio prompts, etc.)
            # Here we attempt a generic call pattern; you must adapt to the repo's actual signature.
            try:
                # Attempt a common forward signature - adapt if your UnifiedVoice expects
                # different argument names. We call it in teacher forcing (training) mode if supported.
                # Typical names: input_ids, attention_mask, cond (speaker/emotion/duration), return_dict=True
                logits = self.unified.forward(input_ids=dec_input, attention_mask=attn_input_mask,
                                              cond_vec=cond, return_dict=False)
                # If unified returns dict or tuple, adapt:
                if isinstance(logits, tuple) or isinstance(logits, list):
                    # assume logits at position 0
                    logits = logits[0]
                return logits, self.spk_classifier(e), e
            except Exception as e_unf:
                print("[T2SModelWrapper] UnifiedVoice forward failed with exception; falling back to local GPTSemanticGenerator.")
                print("Exception:", e_unf)
                traceback.print_exc()
                self.use_unifiedvoice = False
                self.gpt = GPTSemanticGenerator(self.vocab_size, hp.embed_dim, max_len=hp.max_T)

        # Local GPT path (always runnable)
        if not self.use_unifiedvoice:
            # cond -> memory vector for local GPT
            mem = self.cond_proj(cond)  # [B, D]
            logits = self.gpt(cond, dec_input, attn_input_mask)  # [B, L, V]
            q_e_probs = self.spk_classifier(e)
            return logits, q_e_probs, e

# -----------------------
# Loss function
# -----------------------
def compute_loss(logits, dec_target, q_e_probs, Ts, alpha):
    """
    logits: [B, L, V]
    dec_target: [B, L] (token ids; pad=0)
    q_e_probs: [B, n_speakers]
    Ts: [B]
    """
    B, L, V = logits.shape
    logits_flat = logits.view(-1, V)
    targets_flat = dec_target.view(-1)
    # use ignore_index = 0? we used pad_id=0; better to set pad token id as 0 and ignore it in loss
    # But cross_entropy's ignore_index uses target value; here pad id = 0 -> ignore_index=0
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, ignore_index=0, reduction='none')
    loss_per_token = loss_per_token.view(B, L)
    # denom uses Ts (body length). Add 1 to avoid zero division
    token_sum = (loss_per_token * (dec_target != 0).float()).sum(dim=1)
    denom = (Ts.float() + 1.0).to(token_sum.device)
    main_loss = token_sum / denom
    q_e_max, _ = q_e_probs.max(dim=-1)
    adv_term = - alpha * torch.log(q_e_max + 1e-9)
    total = (main_loss + adv_term).mean()
    return total, main_loss.mean().item(), adv_term.mean().item()

# -----------------------
# Training loop
# -----------------------
def train(txt_path="train.txt"):
    # TODO:使用utils/front/tokenizer进行token化
    # 词表vocab使用checkpoints/bpe.model，使用方法剑见utils/front main
    vocab = build_vocab_from_text(txt_path)
    vocab_size = len(vocab)
    # TODO：Dataset 中文拼音随机替换，随即替换函数见utils/replacepinyin
    ds = T2SDataset(txt_path, vocab, max_T=hp.max_T, embed_dim=hp.embed_dim)
    dl = DataLoader(ds, batch_size=hp.batch_size, shuffle=True, collate_fn=t2s_collate_fn, drop_last=False)

    model = T2SModelWrapper(hp, vocab_size).to(hp.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] total params: {total_params:,}")

    global_step = 0
    for stage in [1, 2, 3]:
        print(f"\n===== START STAGE {stage} =====")
        # TODO： STAGE freeze rules 检查一下对不对
        if stage == 2:
            # freeze speaker encoder parameters except classifier (we only have linear spk_encoder here)
            for name, p in model.named_parameters():
                if "spk_encoder" in name:
                    p.requires_grad = False
            print("[Stage2] spk encoder frozen (approx).")
        elif stage == 3:
            # freeze feature extractors: text_encoder, spk_encoder, emo_encoder
            for name, p in model.named_parameters():
                if any(k in name for k in ["text_encoder", "spk_encoder", "emo_encoder"]):
                    p.requires_grad = False
            # ensure Wnum and generator trainable
            for name, p in model.named_parameters():
                if any(k in name for k in ["Wnum", "gpt", "unified", "cond_proj"]):
                    p.requires_grad = True
            print("[Stage3] encoders frozen; Wnum+generator trainable (approx).")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.lr)
        model.train()
        for epoch in range(hp.epochs_per_stage):
            epoch_loss = 0.0
            t0 = time.time()
            for batch in dl:
                Etexts, dec_input, dec_target, attn_input_mask, spk_prompts, emo_prompts, Ts = batch
                Etexts = Etexts.to(hp.device).float()
                dec_input = dec_input.to(hp.device).long()
                dec_target = dec_target.to(hp.device).long()
                attn_input_mask = attn_input_mask.to(hp.device).long()
                spk_prompts = spk_prompts.to(hp.device).float()
                emo_prompts = emo_prompts.to(hp.device).float()
                Ts = Ts.to(hp.device).long()

                # optional text encoder transform
                Etexts_enc = model.text_encoder(Etexts)

                logits, q_e_probs, e = model(Etexts_enc, dec_input, attn_input_mask, spk_prompts, emo_prompts, Ts, stage)

                # adversarial GRL handling: if stage >= 2, apply grad_reverse on e before classifier
                if stage >= 2:
                    e_rev = grad_reverse(e, l=1.0)
                    q_e_probs = model.spk_classifier(e_rev)
                else:
                    q_e_probs = model.spk_classifier(e.detach())

                loss, main_l, adv_l = compute_loss(logits, dec_target, q_e_probs, Ts, hp.alpha)

                optimizer.zero_grad()
                loss.backward()
                # gradient clipping may help
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 50 == 0:
                    print(f"[Stage {stage}] step {global_step} loss={loss.item():.4f} main={main_l:.4f} adv={adv_l:.4f}")

            t1 = time.time()
            avg = epoch_loss / max(1, len(dl))
            print(f"[Stage {stage}] Epoch {epoch} done. avg_loss={avg:.4f} time={t1-t0:.1f}s")

        # save checkpoint
        ckpt = {
            "stage": stage,
            "state_dict": model.state_dict(),
            "hp": vars(hp),
            "vocab": vocab
        }
        ckpt_path = os.path.join(hp.save_dir, f"t2s_stage{stage}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"[Stage {stage}] checkpoint saved -> {ckpt_path}")

    print("Training complete.")

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    # If you want accurate tracebacks for CUDA device asserts:
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train("train.txt")