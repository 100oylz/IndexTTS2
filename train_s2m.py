import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from omegaconf import OmegaConf
from indextts.s2mel.s2m_model import S2MModel
from indextts.gpt.model_v2 import UnifiedVoice


class DummyS2MDataset(Dataset):
    """
    简化示例数据集（可替换为真实语音数据）
    """
    def __init__(self, num_samples=32, seq_len=200):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        semantic_code = torch.randn(self.seq_len, 1024)
        gpt_latent = torch.randn(1280)
        speaker_emb = torch.randn(192)
        mel = torch.randn(self.seq_len, 80)
        return {
            "semantic_code": semantic_code,
            "gpt_latent": gpt_latent,
            "speaker_emb": speaker_emb,
            "mel": mel
        }


class S2MTrainer:
    def __init__(self, config_path, checkpoint_dir):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = S2MModel(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, dataset, epochs=5):
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
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }, os.path.join(self.checkpoint_dir, f"s2m_epoch{epoch+1}.pth"))


if __name__ == "__main__":
    trainer = S2MTrainer("checkpoints/config.yaml", "checkpoints")
    dataset = DummyS2MDataset()
    trainer.train(dataset, epochs=3)
