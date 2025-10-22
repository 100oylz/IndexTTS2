import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
import warnings
from transformers import AutoModelForCausalLM
from transformers.generation import GenerationConfig

from torch.cuda.amp import autocast

import json

warnings.filterwarnings('ignore')

# 导入 front.py 中的 TextTokenizer 和 TextNormalizer
from indextts.utils.front import TextTokenizer, TextNormalizer
from data_t2e_extract import load_and_process_jsonl


class EmotionFineTuner:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", lora_rank=8, lora_alpha=32, vocab_file="checkpoints/bpe.model"):
        """
        初始化情感微调器

        Args:
            model_name: 预训练模型名称
            lora_rank: LoRA秩
            lora_alpha: LoRA alpha参数
            vocab_file: TextTokenizer使用的词表文件路径
        """
        self.model_name = model_name
        self.vocab_file = vocab_file

        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation and GPU drivers.")

        # 设置设备为CUDA
        self.device = torch.device("cuda")
        print(f"Using device: {self.device}")

        # 使用 front.py 中的 TextTokenizer 替换 AutoTokenizer
        text_normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(
            vocab_file=self.vocab_file,
            normalizer=text_normalizer
        )

        # 由于 TextTokenizer 的 pad_token 是只读属性，我们创建一个实例变量来存储 pad_token
        # 使用 eos_token 作为 pad_token
        self.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id

        # 加载基础模型，不自动分配设备
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,  # 不自动分配设备
            trust_remote_code=True,
            output_hidden_states=True  # 确保返回隐藏状态
        )

        # 手动将模型移动到CUDA
        self.base_model = self.base_model.to(self.device)

        # 配置LoRA参数
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # 针对Qwen模型的注意力模块
        )

        # 应用LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()

        # 确保模型在CUDA上
        self.model = self.model.to(self.device)

        # 定义7种基本情绪
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

        # 获取模型的隐藏层大小
        hidden_size = self.base_model.config.hidden_size
        print(f"Model hidden size: {hidden_size}")

        # 初始化线性层并移动到CUDA
        self.emotion_classifier = nn.Linear(hidden_size, len(self.emotions)).to(self.device)

    def prepare_data(self, dataset_path):
        """
        准备训练数据

        Args:
            dataset_path: 数据集路径，包含文本和对应的情绪分布
        """
        data = load_and_process_jsonl(dataset_path)

        texts = [item["text"] for item in data]
        emotion_probs = [item['emotion_vector'] for item in data]

        # 使用 TextTokenizer 进行编码
        batch_encoded = self.tokenizer.batch_encode(
            texts,
            out_type=int
        )

        # 手动处理padding，因为TextTokenizer没有内置的padding功能
        max_length = max(len(seq) for seq in batch_encoded)

        padded_input_ids = []
        attention_masks = []

        for seq in batch_encoded:
            # 截断或填充序列
            if len(seq) > max_length:
                padded_seq = seq[:max_length]
                attention_mask = [1] * max_length
            else:
                # 使用 eos_token_id 作为填充
                padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
                attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))

            padded_input_ids.append(padded_seq)
            attention_masks.append(attention_mask)

        # 转换为tensor
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        emotion_tensor = torch.tensor(emotion_probs, dtype=torch.float32)

        dataset = Dataset.from_dict({
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": emotion_tensor
        })

        return dataset

    def compute_loss(self, outputs, emotion_probs):
        """
        计算损失函数 - 对应你提供的数学表达式

        Args:
            outputs: 模型输出
            emotion_probs: 真实的情感概率分布
        """
        # 获取模型的隐藏状态而不是logits
        hidden_states = outputs.hidden_states[-1]  # 取最后一层的隐藏状态

        # 检查隐藏状态的形状
        batch_size, seq_len, hidden_size = hidden_states.shape
        print(f"Hidden states shape: {hidden_states.shape}")

        # 取最后一个token的隐藏状态作为序列表示
        last_hidden_state = hidden_states[:, -1, :]  # shape: (batch_size, hidden_size)
        print(f"Last hidden state shape: {last_hidden_state.shape}")

        # 检查emotion_probs的形状
        print(f"Emotion probs shape: {emotion_probs.shape}")

        # 通过线性层将隐藏状态映射到7个情绪类别
        emotion_logits = self.emotion_classifier(last_hidden_state)
        print(f"Emotion logits shape: {emotion_logits.shape}")

        # 使用交叉熵损失
        loss = nn.CrossEntropyLoss()(emotion_logits, emotion_probs)

        return loss

    def train(self, dataset, batch_size=4, num_epochs=10, learning_rate=1e-4):
        """
        训练模型

        Args:
            dataset: 训练数据集
            batch_size: 批次大小
            num_epochs: 训练轮数
            learning_rate: 学习率
        """

        def collate_fn(batch):
            # 确保所有张量都在CUDA上
            return {
                "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long).to(
                    self.device),
                "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.float32).to(self.device)
            }

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        print("start training")
        for epoch in range(num_epochs):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                # 确保所有输入都在CUDA上
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播，确保返回隐藏状态
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True  # 确保返回隐藏状态
                )

                # 计算损失
                loss = self.compute_loss(outputs, labels)
                print(f"iter {i} loss：{loss.item()}")

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def predict_emotion(self, text):
        """
        预测文本的情感分布

        Args:
            text: 输入文本

        Returns:
            情感概率分布
        """
        self.model.eval()

        with torch.no_grad():
            # 使用 TextTokenizer 编码文本
            input_ids = self.tokenizer.encode(text, out_type=int)

            # 转换为tensor并添加batch维度，确保在CUDA上
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            attention_mask_tensor = torch.tensor([[1] * len(input_ids)], dtype=torch.long).to(self.device)

            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                output_hidden_states=True  # 确保返回隐藏状态
            )

            # 获取隐藏状态而不是logits
            hidden_states = outputs.hidden_states[-1]
            last_hidden_state = hidden_states[:, -1, :]

            # 获取预测结果并转换为概率分布
            emotion_logits = self.emotion_classifier(last_hidden_state)
            emotion_probs = torch.softmax(emotion_logits, dim=-1)

            return emotion_probs.cpu().numpy()[0]

    def compute_emotion_vector(self, text, emotion_embeddings):
        """
        计算情感向量 - 对应你提供的数学表达式

        Args:
            text: 输入文本
            emotion_embeddings: 情感嵌入字典 {emotion: [embedding_vectors]}

        Returns:
            情感向量
        """
        # 获取情感概率分布
        emotion_probs = self.predict_emotion(text)

        # 计算加权平均的情感向量
        emotion_vector = np.zeros_like(next(iter(emotion_embeddings.values()))[0])

        for i, emotion in enumerate(self.emotions):
            if emotion in emotion_embeddings:
                # 计算该情绪的平均嵌入向量
                avg_embedding = np.mean(emotion_embeddings[emotion], axis=0)
                # 加权累加
                emotion_vector += emotion_probs[i] * avg_embedding

        return emotion_vector


# 使用示例
def main():
    # 初始化微调器
    tuner = EmotionFineTuner()

    # 准备数据
    dataset = tuner.prepare_data("data/processed_emotion_data.jsonl")

    # 训练模型
    tuner.train(dataset)

    # 预测示例
    test_text = "今天天气很好，心情愉快"
    emotion_probs = tuner.predict_emotion(test_text)

    print("情感概率分布:")
    for emotion, prob in zip(tuner.emotions, emotion_probs):
        print(f"{emotion}: {prob:.4f}")

    # 保存LoRA权重
    tuner.model.save_pretrained("./emotion_lora_weights")


# LoRA参数调优配置
def create_lora_configs():
    """创建不同的LoRA配置用于调参"""

    configs = {
        "small": LoraConfig(
            r=2, lora_alpha=8, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        ),
        "medium": LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj"]
        ),
        "large": LoraConfig(
            r=16, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"]
        )
    }

    return configs


if __name__ == "__main__":
    main()

'''
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
import warnings
from transformers import AutoModelForCausalLM
from transformers.generation import GenerationConfig

from torch.cuda.amp import autocast

import json
warnings.filterwarnings('ignore')

# 导入 front.py 中的 TextTokenizer 和 TextNormalizer
from indextts.utils.front import TextTokenizer, TextNormalizer
from data_t2e_extract import load_and_process_jsonl

class EmotionFineTuner:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", lora_rank=8, lora_alpha=32, vocab_file="checkpoints/bpe.model"):
        """
        初始化情感微调器

        Args:
            model_name: 预训练模型名称
            lora_rank: LoRA秩
            lora_alpha: LoRA alpha参数
            vocab_file: TextTokenizer使用的词表文件路径
        """
        self.model_name = model_name
        self.vocab_file = vocab_file

        # 使用 front.py 中的 TextTokenizer 替换 AutoTokenizer
        text_normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(
            vocab_file=self.vocab_file,
            normalizer=text_normalizer
        )

        # 由于 TextTokenizer 的 pad_token 是只读属性，我们创建一个实例变量来存储 pad_token
        # 使用 eos_token 作为 pad_token
        self.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id

        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        # 配置LoRA参数
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # 针对Qwen模型的注意力模块
        )

        # 应用LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()

        # 定义7种基本情绪
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    def prepare_data(self, dataset_path):
        """
        准备训练数据

        Args:
            dataset_path: 数据集路径，包含文本和对应的情绪分布
        """
        data=load_and_process_jsonl(dataset_path)

        texts = [item["text"] for item in data]
        emotion_probs = [item['emotion_vector'] for item in data]

        # 使用 TextTokenizer 进行编码
        batch_encoded = self.tokenizer.batch_encode(
            texts,
            out_type=int
        )

        # 手动处理padding，因为TextTokenizer没有内置的padding功能
        max_length = max(len(seq) for seq in batch_encoded)

        padded_input_ids = []
        attention_masks = []

        for seq in batch_encoded:
            # 截断或填充序列
            if len(seq) > max_length:
                padded_seq = seq[:max_length]
                attention_mask = [1] * max_length
            else:
                # 使用 eos_token_id 作为填充
                padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
                attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))

            padded_input_ids.append(padded_seq)
            attention_masks.append(attention_mask)

        # 转换为tensor
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        #print(type(input_ids_tensor))
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        #print(type(attention_mask_tensor))
        emotion_tensor = torch.tensor(emotion_probs, dtype=torch.float32)
        #print(type(emotion_tensor))

        dataset = Dataset.from_dict({
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": emotion_tensor
        })

        return dataset

    def compute_loss(self, outputs, emotion_probs):
        """
        计算损失函数 - 对应你提供的数学表达式

        Args:
            outputs: 模型输出
            emotion_probs: 真实的情感概率分布
        """
        # 获取模型的logits
        logits = outputs.logits

        # 取最后一个token的隐藏状态作为序列表示
        last_hidden_state = logits[:, -1, :]

        # 通过一个线性层将隐藏状态映射到7个情绪类别
        emotion_logits = nn.Linear(last_hidden_state.size(-1), len(self.emotions))(last_hidden_state)

        # 使用交叉熵损失
        # 注意：这里使用KL散度可能更合适，因为是多标签概率分布
        loss = nn.CrossEntropyLoss()(emotion_logits, emotion_probs)

        return loss



    def train(self, dataset, batch_size=4, num_epochs=10, learning_rate=1e-4):
        """
        训练模型

        Args:
            dataset: 训练数据集
            batch_size: 批次大小
            num_epochs: 训练轮数
            learning_rate: 学习率
        """



        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
                "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
                "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.float32)
            }

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        print("start training")
        for epoch in range(num_epochs):
            total_loss = 0
            for i,batch in enumerate(dataloader):
                optimizer.zero_grad()

                # 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
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
        """
        预测文本的情感分布

        Args:
            text: 输入文本

        Returns:
            情感概率分布
        """
        self.model.eval()

        with torch.no_grad():
            # 使用 TextTokenizer 编码文本
            input_ids = self.tokenizer.encode(text, out_type=int)

            # 转换为tensor并添加batch维度
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long)
            }

            outputs = self.model(**inputs)

            # 获取预测结果并转换为概率分布
            logits = outputs.logits[:, -1, :]
            emotion_logits = nn.Linear(logits.size(-1), len(self.emotions))(logits)
            emotion_probs = torch.softmax(emotion_logits, dim=-1)

            return emotion_probs.cpu().numpy()[0]

    def compute_emotion_vector(self, text, emotion_embeddings):
        """
        计算情感向量 - 对应你提供的数学表达式

        Args:
            text: 输入文本
            emotion_embeddings: 情感嵌入字典 {emotion: [embedding_vectors]}

        Returns:
            情感向量
        """
        # 获取情感概率分布
        emotion_probs = self.predict_emotion(text)

        # 计算加权平均的情感向量
        emotion_vector = np.zeros_like(next(iter(emotion_embeddings.values()))[0])

        for i, emotion in enumerate(self.emotions):
            if emotion in emotion_embeddings:
                # 计算该情绪的平均嵌入向量
                avg_embedding = np.mean(emotion_embeddings[emotion], axis=0)
                # 加权累加
                emotion_vector += emotion_probs[i] * avg_embedding

        return emotion_vector


# 使用示例
def main():
    # 初始化微调器
    tuner = EmotionFineTuner()

    # 准备数据（假设你已经有数据集）
    dataset = tuner.prepare_data("data/processed_emotion_data.jsonl")
    # print(dataset)
    # print(type(dataset['input_ids'][0]))
    # print(type(dataset['attention_mask'][0]))
    # print(type(dataset['labels'][0]))
    #exit(0)


    # 训练模型
    tuner.train(dataset)

    # 预测示例
    test_text = "今天天气很好，心情愉快"
    emotion_probs = tuner.predict_emotion(test_text)

    print("情感概率分布:")
    for emotion, prob in zip(tuner.emotions, emotion_probs):
        print(f"{emotion}: {prob:.4f}")

    # 计算情感向量（需要提前准备情感嵌入）
    # emotion_embeddings = {
    #     "anger": [np.random.rand(768) for _ in range(10)],  # 示例嵌入
    #     "joy": [np.random.rand(768) for _ in range(10)],
    #     # ... 其他情绪
    # }
    # emotion_vector = tuner.compute_emotion_vector(test_text, emotion_embeddings)

    # 保存LoRA权重
    # tuner.model.save_pretrained("./emotion_lora_weights")


# LoRA参数调优配置
def create_lora_configs():
    """创建不同的LoRA配置用于调参"""

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
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"]
        )
    }

    return configs


if __name__ == "__main__":
    main()
'''


