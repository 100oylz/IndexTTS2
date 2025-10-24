import os
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from indextts.s2mel.s2m_model import S2MModel


# ===== 1️⃣ 模拟数据集 =====
class DummyS2MDataset(Dataset):
    def __init__(self, num_samples=16, seq_len=150):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "semantic_code": torch.randn(self.seq_len, 1024),
            "gpt_latent": torch.randn(1280),
            "speaker_emb": torch.randn(192),
            "mel": torch.randn(self.seq_len, 80),
        }

# ===== 2️⃣ 配置加载辅助函数 =====
def load_config_with_s2mel_on_top(config_path):
    cfg = OmegaConf.load(config_path)
    if hasattr(cfg, "s2mel"):
        for k, v in cfg.s2mel.items():
            # 仅当顶层没有该 key 时才提升
            if not hasattr(cfg, k):
                cfg[k] = v
    return cfg

# ===== 3️⃣ 训练器 =====
class S2MTrainer:
    def __init__(self, config_path="checkpoints/config.yaml", ckpt_dir="checkpoints"):
        self.config = load_config_with_s2mel_on_top(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = S2MModel(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def train(self, dataset, epochs=3):
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()
                loss = self.model(
                    batch["semantic_code"].to(self.device),
                    batch["gpt_latent"].to(self.device),
                    batch["speaker_emb"].to(self.device),
                    mel_target=batch["mel"].to(self.device),
                    train=True
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Avg L1 Loss = {total_loss / len(loader):.6f}")

        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, os.path.join(self.ckpt_dir, "s2m_dummy_test.pth"))

    def test_inference(self):
        self.model.eval()
        with torch.no_grad():
            semantic = torch.randn(1, 200, 1024).to(self.device)
            gpt = torch.randn(1, 1280).to(self.device)
            speaker = torch.randn(1, 192).to(self.device)
            mel_pred = self.model(semantic, gpt, speaker, train=False)
            print("Inference mel_pred shape:", mel_pred.shape)
            print("mel_pred sample values:", mel_pred[0, :5, :5])

# ===== 4️⃣ 运行测试 =====
if __name__ == "__main__":
    dummy_data = DummyS2MDataset(num_samples=20, seq_len=180)
    trainer = S2MTrainer("checkpoints/config.yaml", "checkpoints")
    trainer.train(dummy_data, epochs=3)
    trainer.test_inference()
