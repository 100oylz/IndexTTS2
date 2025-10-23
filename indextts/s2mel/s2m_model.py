import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from indextts.s2mel.modules.commons import MyModel, init_weights
from indextts.s2mel.modules.length_regulator import InterpolateRegulator
from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.s2mel.wav2vecbert_extract import Extract_wav2vectbert


class S2MModel(MyModel):
    """
    IndexTTS2 S2M 模块 (修正版)
    Flow-matching based Mel generator
    """

    def __init__(self, cfg):
        super().__init__(cfg, use_gpt_latent=True)
        self.cfg = cfg
        s2m_cfg = cfg.s2mel

        self.mel_dim = s2m_cfg.DiT.in_channels       # 80
        self.semantic_dim = s2m_cfg.length_regulator.in_channels  # 1024
        self.gpt_dim = cfg.gpt.model_dim             # 1280
        self.spk_dim = s2m_cfg.style_encoder.dim     # 192

        # GPT latent 融合层（投影到语义空间）
        self.gpt_fusion = nn.Sequential(
            nn.Linear(self.gpt_dim, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
            nn.ReLU(),
        )

        # Flow 网络 (简单 Transformer/MLP)
        self.flow_net = nn.Sequential(
            nn.Linear(self.mel_dim + self.semantic_dim + self.spk_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.mel_dim)
        )

        # 编码器
        self.semantic_codec = build_semantic_codec(cfg.semantic_codec)
        self.wav2vecbert = Extract_wav2vectbert(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.apply(init_weights)

    def forward(
        self, semantic_tokens, gpt_latent, speaker_emb,
        mel_target=None, train=True
    ):
        """
        semantic_tokens: [B, T, 1024]
        gpt_latent: [B, 1280]
        speaker_emb: [B, 192]
        mel_target: [B, T, 80]
        """

        B, T, _ = semantic_tokens.shape

        # 论文公式: Q_sem + 0.5*MLP(H_GPT)
        if train and torch.rand(1).item() < 0.5:
            gpt_proj = self.gpt_fusion(gpt_latent)          # [B, 1024]
            fused_sem = semantic_tokens + gpt_proj.unsqueeze(1)
        else:
            fused_sem = semantic_tokens

        spk = speaker_emb.unsqueeze(1).expand(-1, T, -1)
        cond = torch.cat([fused_sem, spk], dim=-1)          # [B, T, 1216]

        if train:
            # Flow Matching training (L1 loss)
            t = torch.rand(B, 1, 1, device=semantic_tokens.device)
            noise = torch.randn_like(mel_target)
            x_t = (1 - t) * mel_target + t * noise
            v_target = noise - mel_target
            v_pred = self.flow_net(torch.cat([x_t, cond], dim=-1))
            loss = F.l1_loss(v_pred, v_target)
            return loss
        else:
            # 推理: ODE 解算 (RK4)
            y = torch.randn(B, T, self.mel_dim, device=semantic_tokens.device)
            t_span = torch.linspace(0, 1, 50).to(y.device)

            def ode_func(ti, yi):
                return self.flow_net(torch.cat([yi, cond], dim=-1))

            y_pred = odeint(ode_func, y, t_span, method='rk4')[-1]
            return y_pred
