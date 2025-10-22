"""
Complete T2S training script (text-based dataset).
- Reads train.txt (each line: utt_id + tokenized text OR raw chars)
- Builds vocab (adds <pad>, <unk>, <bos>, <eos>)
- T2SDataset returns: Etext (placeholder), sem_target (ids with <bos>/<eos>), spk_prompt, emo_prompt, T
- Transformer decoder with teacher forcing
- Duration control: p = Wnum(h(T)) where h(T) = num_embed(one_hot(T))
- Emotion adversarial: GRL + speaker classifier -> q(e)
- Three-stage training:
    Stage 1: p randomly zeroed with prob 0.3
    Stage 2: freeze speaker encoder, apply GRL on e
    Stage 3: freeze encoders, fine-tune Wnum + decoder
"""

import os
import random
import math
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Hyperparameters
# -----------------------
class HParams:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 256            # D
    max_T = 64                 # maximum supported semantic sequence length (for duration one-hot)
    alpha = 0.5                # adversarial loss coefficient
    batch_size = 8
    lr = 1e-4
    epochs_per_stage = 3       # for real training increase
    n_speakers = 100
    vocab_min_freq = 1
    save_dir = "./checkpoints"
    seed = 42

hp = HParams()
random.seed(hp.seed)
torch.manual_seed(hp.seed)
os.makedirs(hp.save_dir, exist_ok=True)

# -----------------------
# Utility: Gradient Reversal Layer
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
# Vocab builder (character-level by default)
# -----------------------
def build_vocab_from_text(txt_path, add_special_tokens=True):
    chars = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support lines with "utt_id text..." or raw text
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                _, text = parts
            else:
                text = parts[0]
            for ch in text:
                chars[ch] = chars.get(ch, 0) + 1
    # create sorted vocab of tokens that meet min freq
    tokens = []
    for ch, freq in sorted(chars.items()):
        if freq >= hp.vocab_min_freq:
            tokens.append(ch)
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
    """
    Reads lines from train.txt (format: '<utt_id> <text>' or '<text>').
    Produces:
        - Etext: placeholder embedding tensor [embed_dim]
        - sem_target: token id sequence including <bos> ... <eos>
        - spk_prompt, emo_prompt: placeholder embeddings [embed_dim]
        - T: int length of semantic token sequence (excluding bos/eos?)
    """
    def __init__(self, txt_path, vocab, max_T=hp.max_T, embed_dim=hp.embed_dim):
        assert os.path.exists(txt_path), f"{txt_path} not found"
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_T = max_T
        self.embed_dim = embed_dim
        self.samples = []  # list of token id lists (without bos/eos)
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    _, text = parts
                else:
                    text = parts[0]
                # tokenization: character-level on the provided text (if already tokenized with spaces, those spaces are kept as tokens)
                # We'll keep exact characters; if you have space-separated tokens you can adapt here.
                token_ids = [vocab.get(ch, vocab.get("<unk>")) for ch in text]
                # limit length to max_T - 2 (to keep room for bos/eos)
                max_body = max(1, self.max_T - 2)
                token_ids = token_ids[:max_body]
                self.samples.append(token_ids)
        print(f"[Dataset] loaded {len(self.samples)} samples from {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        body = self.samples[idx]
        # add bos/eos
        bos = self.vocab["<bos>"]
        eos = self.vocab["<eos>"]
        seq = [bos] + body + [eos]
        # clip safety
        seq = [t if 0 <= t < self.vocab_size else self.vocab["<unk>"] for t in seq]
        sem_target = torch.tensor(seq, dtype=torch.long)
        T = max(1, len(body))  # body length used for duration scaling (you can adjust)
        # placeholders for Etext, spk_prompt, emo_prompt
        Etext = torch.randn(self.embed_dim)
        spk_prompt = torch.randn(self.embed_dim)
        emo_prompt = torch.randn(self.embed_dim)
        return Etext, sem_target, spk_prompt, emo_prompt, T

def t2s_collate_fn(batch):
    """
    Batch: list of (Etext, sem_target, spk_prompt, emo_prompt, T)
    Return:
      Etext: [B, D]
      decoder_input: [B, L]  (for teacher forcing: sem_target[:-1])
      decoder_target: [B, L] (for loss: sem_target[1:], padded with -100)
      spk_prompts: [B, D]
      emo_prompts: [B, D]
      Ts: [B]
    """
    Etexts, sem_targets, spk_prompts, emo_prompts, Ts = zip(*batch)
    Etexts = torch.stack(Etexts)
    spk_prompts = torch.stack(spk_prompts)
    emo_prompts = torch.stack(emo_prompts)
    Ts = torch.tensor(Ts, dtype=torch.long)

    # pad sem_targets to max len in batch
    max_len = max([s.shape[0] for s in sem_targets])
    dec_inputs, dec_targets = [], []
    for s in sem_targets:
        input_seq = s[:-1]
        target_seq = s[1:]
        # pad to (max_len - 1)
        pad_len = max_len - 1
        pad_id = 0  # <pad> id for input
        pad_tgt = -100  # ignore_index for target
        inp = torch.full((pad_len,), pad_id, dtype=torch.long)
        tgt = torch.full((pad_len,), pad_tgt, dtype=torch.long)
        inp[:input_seq.shape[0]] = input_seq
        tgt[:target_seq.shape[0]] = target_seq
        dec_inputs.append(inp)
        dec_targets.append(tgt)


    dec_input = torch.stack(dec_inputs)   # [B, L]
    dec_target = torch.stack(dec_targets) # [B, L]

    return Etexts, dec_input, dec_target, spk_prompts, emo_prompts, Ts

# -----------------------
# Model components
# -----------------------
class SimpleTextEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)

class SimpleSpeakerEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, n_speakers):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        # classifier outputs speaker distribution
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, n_speakers),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.proj(x)
    def classify_from_e(self, e):
        probs = self.classifier(e)
        maxp, _ = probs.max(dim=-1)
        return maxp

class SimpleEmotionEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------
# Transformer Decoder (teacher forcing)
# -----------------------
class TransformerDecoder(nn.Module):
    def __init__(self, cond_dim, vocab_size, d_model=256, nhead=4, num_layers=4, max_len=hp.max_T):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.cond_proj = nn.Linear(cond_dim, d_model)

    def forward(self, cond, dec_input):
        """
        cond: [B, cond_dim]
        dec_input: [B, L] (token ids)  -- teacher forcing input (without final token)
        returns logits: [B, L, V]
        """
        B, L = dec_input.shape
        cond_vec = self.cond_proj(cond).unsqueeze(0)  # [1, B, D]
        tgt_emb = self.embedding(dec_input) + self.pos_encoding[:, :L, :]  # [B, L, D]
        tgt_emb = tgt_emb.transpose(0, 1)  # [L, B, D]
        # TransformerDecoder expects tgt (L,B,D) and memory (S,B,E). We'll use cond_vec as memory of length 1.
        out = self.decoder(tgt_emb, cond_vec)  # [L, B, D]
        logits = self.fc_out(out.transpose(0, 1))  # [B, L, V]
        return logits

# -----------------------
# Full T2S model
# -----------------------
class T2SModel(nn.Module):
    def __init__(self, hp, vocab_size):
        super().__init__()
        D = hp.embed_dim
        self.hp = hp
        self.text_encoder = SimpleTextEncoder(D, D)
        self.spk_encoder = SimpleSpeakerEncoder(D, D, hp.n_speakers)
        self.emo_encoder = SimpleEmotionEncoder(D, D)
        # duration control
        self.num_embed = nn.Linear(hp.max_T, D)
        self.Wnum = nn.Linear(D, D)
        # condition: (c [+ e], p, Etext) => D + D + D = 3D
        cond_dim = D * 3
        self.decoder = TransformerDecoder(cond_dim, vocab_size, d_model=D, max_len=hp.max_T)

    def forward(self, Etext, dec_input, stage, spk_prompt, emo_prompt, Ts):
        """
        Etext: [B, D]
        dec_input: [B, L] (teacher forcing inputs)
        spk_prompt: [B, D]
        emo_prompt: [B, D]
        Ts: [B] int lengths for duration control
        stage: 1/2/3
        """
        B = Etext.shape[0]
        device = Etext.device
        c = self.spk_encoder(spk_prompt)      # [B, D]
        e = self.emo_encoder(emo_prompt)      # [B, D]

        # duration control
        Ts_clamped = Ts.clamp(min=0, max=self.hp.max_T - 1)
        one_hot = F.one_hot(Ts_clamped, num_classes=self.hp.max_T).float().to(device)  # [B, max_T]
        h_T = self.num_embed(one_hot)  # [B, D]
        p = self.Wnum(h_T)  # [B, D]

        # Stage 1: randomly zero p with prob 0.3 per sample
        if stage == 1:
            mask = (torch.rand(B, device=device) < 0.3).float().unsqueeze(-1)  # [B,1]
            p = p * (1.0 - mask)

        # Build cond: use c+e for stage>=2 else c
        if stage >= 2:
            cond_first = c + e
        else:
            cond_first = c
        cond = torch.cat([cond_first, p, Etext], dim=-1)  # [B, 3D]

        logits = self.decoder(cond, dec_input)  # [B, L, V]

        # Speaker classifier distribution for e
        q_e_probs = self.spk_encoder.classifier(e)  # [B, n_speakers]

        return logits, q_e_probs, e

# -----------------------
# Loss function
# -----------------------
def compute_loss(logits, dec_target, q_e_probs, Ts, alpha):
    """
    logits: [B, L, V]
    dec_target: [B, L] with -100 padded positions (ignore_index)
    q_e_probs: [B, n_speakers] (softmax)
    Ts: [B] lengths (for denom)
    """
    B, L, V = logits.shape
    # Flatten for CrossEntropy
    logits_flat = logits.view(-1, V)          # [B*L, V]
    targets_flat = dec_target.view(-1)        # [B*L]
    # CrossEntropyLoss with ignore_index=-100
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100, reduction='none')  # [(B*L)]
    loss_per_token = loss_per_token.view(B, L)  # [B, L]
    token_sum = loss_per_token.sum(dim=1)       # [B]
    denom = (Ts.float() + 1.0).to(token_sum.device)  # [B]
    main_loss = token_sum / denom                # [B]

    # adversarial: use max class probability as q(e)
    q_e_max, _ = q_e_probs.max(dim=-1)  # [B]
    adv_term = - alpha * torch.log(q_e_max + 1e-9)  # [B]

    total_per_sample = main_loss + adv_term
    total_loss = total_per_sample.mean()
    return total_loss, main_loss.mean().item(), adv_term.mean().item()

# -----------------------
# Training orchestration
# -----------------------
def train(txt_path="train.txt"):
    # build vocab & dataset
    vocab = build_vocab_from_text(txt_path)
    vocab_size = len(vocab)
    pad_id = vocab["<pad>"]
    ds = T2SDataset(txt_path, vocab, max_T=hp.max_T, embed_dim=hp.embed_dim)
    dl = DataLoader(ds, batch_size=hp.batch_size, shuffle=True, collate_fn=t2s_collate_fn)

    model = T2SModel(hp, vocab_size).to(hp.device)
    # print param counts
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] total params: {total_params:,}")

    # optimizer will be recreated per stage to track trainable params only
    global_step = 0
    for stage in [1, 2, 3]:
        print(f"\n===== START STAGE {stage} =====")
        # Stage freezing rules
        if stage == 2:
            # freeze speaker encoder (as requested)
            for p in model.spk_encoder.parameters():
                p.requires_grad = False
            print("[Stage2] speaker encoder frozen.")
        elif stage == 3:
            # freeze feature extractors (text/spk/emo)
            for p in model.text_encoder.parameters():
                p.requires_grad = False
            for p in model.spk_encoder.parameters():
                p.requires_grad = False
            for p in model.emo_encoder.parameters():
                p.requires_grad = False
            # Ensure Wnum and decoder remain trainable
            for p in model.Wnum.parameters():
                p.requires_grad = True
            for p in model.decoder.parameters():
                p.requires_grad = True
            print("[Stage3] text/spk/emo encoders frozen. Wnum+decoder trainable.")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.lr)

        model.train()
        for epoch in range(hp.epochs_per_stage):
            epoch_loss = 0.0
            t0 = time.time()
            for batch in dl:
                Etexts, dec_input, dec_target, spk_prompts, emo_prompts, Ts = batch
                # to device
                Etexts = Etexts.to(hp.device).float()
                dec_input = dec_input.to(hp.device).long()
                dec_target = dec_target.to(hp.device).long()
                spk_prompts = spk_prompts.to(hp.device).float()
                emo_prompts = emo_prompts.to(hp.device).float()
                Ts = Ts.to(hp.device).long()

                # Etext may be placeholder; optionally pass through text_encoder
                Etexts_enc = model.text_encoder(Etexts)  # [B, D]

                # FORWARD
                logits, q_e_probs, e = model(Etexts_enc, dec_input, stage, spk_prompts, emo_prompts, Ts)

                # For adversarial GRL: if stage >=2, compute classifier on reversed-grad e
                if stage >= 2:
                    e_rev = grad_reverse(e, l=1.0)
                    q_e_probs = model.spk_encoder.classifier(e_rev)
                else:
                    # stage1: do not apply adversarial gradient, keep classifier detached (no penalty effect on encoder)
                    q_e_probs = model.spk_encoder.classifier(e.detach())

                loss, main_l, adv_l = compute_loss(logits, dec_target, q_e_probs, Ts, hp.alpha)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 50 == 0:
                    print(f"[Stage {stage} Epoch {epoch}] step {global_step} loss={loss.item():.4f} main={main_l:.4f} adv={adv_l:.4f}")

            t1 = time.time()
            avg = epoch_loss / len(dl)
            print(f"Stage {stage} Epoch {epoch} done. avg_loss={avg:.4f} time={t1-t0:.1f}s")

        # optional stage-3 fine-tune step on Wnum (perform small synthetic updates)
        if stage == 3:
            print("[Stage3] Optional Wnum fine-tuning over synthetic T grid")
            model.train()
            for t_val in range(1, min(16, hp.max_T)):
                bsz = min(hp.batch_size, 8)
                Etexts = torch.randn(bsz, hp.embed_dim, device=hp.device)
                spk_prompts = torch.randn(bsz, hp.embed_dim, device=hp.device)
                emo_prompts = torch.randn(bsz, hp.embed_dim, device=hp.device)
                Ts_ft = torch.tensor([t_val] * bsz, dtype=torch.long, device=hp.device)
                Etexts_enc = model.text_encoder(Etexts)
                dec_input = torch.zeros((bsz, 1), dtype=torch.long, device=hp.device)  # dummy
                logits, q_e_probs, e = model(Etexts_enc, dec_input, stage, spk_prompts, emo_prompts, Ts_ft)
                dummy_loss = logits.abs().mean()
                optimizer.zero_grad()
                dummy_loss.backward()
                optimizer.step()

        # save checkpoint
        ckpt = {
            "stage": stage,
            "state_dict": model.state_dict(),
            "hp": vars(hp),
            "vocab_size": vocab_size,
            "vocab": {k: int(v) for k, v in vocab.items()}
        }
        ckpt_path = os.path.join(hp.save_dir, f"t2s_stage{stage}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"[Stage {stage}] checkpoint saved -> {ckpt_path}")

    print("Training complete.")

# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    # optional: set CUDA_LAUNCH_BLOCKING=1 if you need precise error traces:
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train("train.txt")