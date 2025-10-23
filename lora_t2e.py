import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import datetime
from pathlib import Path

# 导入 front.py 中的 TextTokenizer 和 TextNormalizer
from indextts.utils.front import TextTokenizer, TextNormalizer
from data_t2e_extract import load_and_process_jsonl


# 超参数配置类
class TrainingConfig:
    """训练配置参数"""

    def __init__(self):
        # 模型参数
        self.model_name = "Qwen/Qwen3-1.7B"
        self.vocab_file = "checkpoints/bpe.model"

        # LoRA参数
        self.lora_rank = 2
        self.lora_alpha = 8
        self.lora_dropout = 0.1
        self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        # 训练参数
        self.batch_size = 2
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.save_dir = "checkpoints/emotion_finetune"

        # 数据参数
        self.emotions = ["Anger", "Happiness", "Fear", "Disgust", "Sadness", "Surprise", "Neutral"]


class EmotionFineTuner:
    def __init__(self, config=None):
        """初始化情感微调器"""
        self.config = config if config else TrainingConfig()

        # 检查是否有可用的CUDA设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化Tokenizer
        text_normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(
            vocab_file=self.config.vocab_file,
            normalizer=text_normalizer
        )

        # 设置pad_token
        self.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id

        # 加载基础模型并移动到指定设备
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)

        # 获取模型隐藏层维度
        self.hidden_size = self.base_model.config.hidden_size
        print(f"模型隐藏层维度: {self.hidden_size}")

        # 配置LoRA参数
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )

        # 应用LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()

        # 定义情感预测头
        self.emotion_head = nn.Linear(
            self.hidden_size, len(self.config.emotions)).to(self.device)

        # 创建保存目录
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 训练历史记录
        self.training_history = {
            'epochs': [],
            'losses': [],
            'config': self.config.__dict__
        }

    def prepare_data(self, dataset_path):
        """准备训练数据"""
        data = load_and_process_jsonl(dataset_path)

        texts = [item["text"] for item in data]
        emotion_probs = [item['emotion_vector'] for item in data]

        # 使用TextTokenizer进行编码
        batch_encoded = self.tokenizer.batch_encode(
            texts,
            out_type=int
        )

        # 手动处理padding
        max_length = max(len(seq) for seq in batch_encoded)

        padded_input_ids = []
        attention_masks = []

        for seq in batch_encoded:
            if len(seq) > max_length:
                padded_seq = seq[:max_length]
                attention_mask = [1] * max_length
            else:
                padded_seq = seq + [self.pad_token_id] * \
                             (max_length - len(seq))
                attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))

            padded_input_ids.append(padded_seq)
            attention_masks.append(attention_mask)

        # 转换为tensor并移动到设备
        input_ids_tensor = torch.tensor(
            padded_input_ids, dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor(
            attention_masks, dtype=torch.long).to(self.device)
        emotion_tensor = torch.tensor(
            emotion_probs, dtype=torch.float32).to(self.device)

        dataset = Dataset.from_dict({
            "input_ids": input_ids_tensor.cpu().numpy(),
            "attention_mask": attention_mask_tensor.cpu().numpy(),
            "labels": emotion_tensor.cpu().numpy()
        })

        return dataset

    def compute_loss(self, outputs, emotion_probs):
        """计算L1损失函数"""
        last_hidden_state = outputs.hidden_states[-1]
        cls_embedding = last_hidden_state[:, -1, :]
        emotion_logits = self.emotion_head(cls_embedding)
        emotion_pred_probs = torch.softmax(emotion_logits, dim=1)
        loss = nn.L1Loss(reduction="mean")(emotion_pred_probs, emotion_probs)
        return loss

    def save_checkpoint(self, epoch, loss, optimizer, is_best=False):
        """保存检查点"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型相关文件
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'emotion_head_state_dict': self.emotion_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }

        # 常规检查点
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        torch.save(checkpoint, checkpoint_path)

        # 如果是最好模型，额外保存
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存到: {best_path}")

        # 保存训练历史
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)

        print(f"检查点已保存到: {checkpoint_path}")

    def train(self, dataset, batch_size=None, num_epochs=None, learning_rate=None):
        """训练模型"""
        # 使用配置参数或传入参数
        batch_size = batch_size or self.config.batch_size
        num_epochs = num_epochs or self.config.num_epochs
        learning_rate = learning_rate or self.config.learning_rate

        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long).to(
                    self.device),
                "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.float32).to(self.device)
            }

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate)

        self.model.train()
        best_loss = float('inf')

        print("开始训练...")
        print(f"超参数: batch_size={batch_size}, num_epochs={num_epochs}, learning_rate={learning_rate}")

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = len(dataloader)

            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True
                )

                loss = self.compute_loss(outputs, batch["labels"])

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % 10 == 0:  # 每10个batch打印一次
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i}/{num_batches}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)

            # 记录训练历史
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['losses'].append(avg_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}, 平均Loss: {avg_loss:.4f}")

            # 保存检查点
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            self.save_checkpoint(epoch + 1, avg_loss, optimizer, is_best)

    def predict_emotion(self, text):
        """预测文本的情感分布"""
        self.model.eval()

        with torch.no_grad():
            # 编码文本
            input_ids = self.tokenizer.encode(text, out_type=int)

            # 转换为tensor并添加batch维度，移动到设备
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long).to(self.device)
            }

            # 预测时同样需要获取隐藏状态
            outputs = self.model(**inputs, output_hidden_states=True)

            # 获取预测结果
            last_hidden_state = outputs.hidden_states[-1]
            cls_embedding = last_hidden_state[:, -1, :]
            emotion_logits = self.emotion_head(cls_embedding)
            emotion_probs = torch.softmax(emotion_logits, dim=-1)

            return emotion_probs.cpu().numpy()[0]

    def compute_emotion_vector(self, text, emotion_embeddings):
        """计算情感向量"""
        emotion_probs = self.predict_emotion(text)

        emotion_vector = np.zeros_like(
            next(iter(emotion_embeddings.values()))[0])

        for i, emotion in enumerate(self.config.emotions):
            if emotion in emotion_embeddings:
                avg_embedding = np.mean(emotion_embeddings[emotion], axis=0)
                emotion_vector += emotion_probs[i] * avg_embedding

        return emotion_vector

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.emotion_head.load_state_dict(checkpoint['emotion_head_state_dict'])

        # 加载训练历史
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        print(f"已加载检查点: {checkpoint_path}")
        print(f"训练轮次: {checkpoint['epoch']}, 损失: {checkpoint['loss']:.4f}")


# 使用示例
def main():
    # 创建配置实例（可以自定义参数）
    config = TrainingConfig()
    config.batch_size = 4
    config.num_epochs = 5
    config.learning_rate = 5e-5
    config.save_dir = "checkpoints/my_emotion_model"

    # 初始化微调器
    tuner = EmotionFineTuner(config)

    # 准备数据并训练
    dataset = tuner.prepare_data("data/processed_emotion_data.jsonl")
    tuner.train(dataset)

    # 测试预测
    test_text = "今天天气很好，心情愉快"
    emotion_probs = tuner.predict_emotion(test_text)

    print("情感概率分布:")
    for emotion, prob in zip(tuner.config.emotions, emotion_probs):
        print(f"{emotion}: {prob:.4f}")


def create_lora_configs():
    """创建不同的LoRA配置"""
    configs = {
        "small": LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        ),
        "medium": LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj"]
        ),
        "large": LoraConfig(
            r=16, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj",
                            "o_proj", "gate_proj", "up_proj"]
        )
    }
    return configs


if __name__ == "__main__":
    main()