import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
import warnings

# 导入 front.py 中的 TextTokenizer 和 TextNormalizer
from indextts.utils.front import TextTokenizer, TextNormalizer
from data_t2e_extract import load_and_process_jsonl


class EmotionFineTuner:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", lora_rank=2, lora_alpha=8, vocab_file="checkpoints/bpe.model"):
        """初始化情感微调器"""
        self.model_name = model_name
        self.vocab_file = vocab_file

        # 检查是否有可用的CUDA设备
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化Tokenizer
        text_normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(
            vocab_file=self.vocab_file,
            normalizer=text_normalizer
        )

        # 设置pad_token
        self.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id

        # 加载基础模型并移动到指定设备
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)

        # 获取模型隐藏层维度（关键：用于匹配情感头输入维度）
        self.hidden_size = self.base_model.config.hidden_size
        print(f"模型隐藏层维度: {self.hidden_size}")

        # 配置LoRA参数
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        # 应用LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()

        self.emotions = ["Anger", "Happiness", "Fear",
                         "Disgust", "Sadness", "Surprise", "Neutral"]

        # 定义情感预测头（输入维度匹配模型隐藏层维度）
        self.emotion_head = nn.Linear(
            self.hidden_size, len(self.emotions)).to(self.device)

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
        """计算L1损失函数（修复隐藏状态提取方式）"""
        # 正确获取最后一层的隐藏状态（而非logits）
        # 对于因果语言模型，hidden_states是元组，最后一个元素是最后一层的输出
        # 形状: [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # 取最后一个token的隐藏状态作为序列表示
        # 形状: [batch_size, hidden_size]
        cls_embedding = last_hidden_state[:, -1, :]

        # 通过线性层将隐藏状态映射到7个情绪类别（现在维度匹配）
        emotion_logits = self.emotion_head(
            cls_embedding)  # 形状: [batch_size, 7]

        # 将logits转换为概率分布
        emotion_pred_probs = torch.softmax(emotion_logits, dim=1)

        # 计算L1损失
        loss = nn.L1Loss(reduction="mean")(emotion_pred_probs, emotion_probs)

        return loss

    def train(self, dataset, batch_size=2, num_epochs=10, learning_rate=1e-4):
        """训练模型"""
        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long).to(self.device),
                "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.float32).to(self.device)
            }

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate)

        self.model.train()
        print("start training")
        for epoch in range(num_epochs):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                # 前向传播时获取隐藏状态（需要设置output_hidden_states=True）
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True  # 关键：启用隐藏状态输出
                )

                # 计算损失
                loss = self.compute_loss(outputs, batch["labels"])
                print(f"iter {i} loss：{loss.item()}")

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

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

            # 获取预测结果（使用正确的隐藏状态）
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

        for i, emotion in enumerate(self.emotions):
            if emotion in emotion_embeddings:
                avg_embedding = np.mean(emotion_embeddings[emotion], axis=0)
                emotion_vector += emotion_probs[i] * avg_embedding

        return emotion_vector


# 使用示例
def main():
    tuner = EmotionFineTuner()
    dataset = tuner.prepare_data("data/processed_emotion_data.jsonl")
    tuner.train(dataset)

    test_text = "今天天气很好，心情愉快"
    emotion_probs = tuner.predict_emotion(test_text)

    print("情感概率分布:")
    for emotion, prob in zip(tuner.emotions, emotion_probs):
        print(f"{emotion}: {prob:.4f}")


def create_lora_configs():
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
