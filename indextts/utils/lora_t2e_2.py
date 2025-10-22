import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from datasets import Dataset


class EmotionFineTuner:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", lora_rank=8, lora_alpha=32):
        """
        初始化情感微调器 - 针对 Qwen2.5-1.7B
        """
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"加载tokenizer失败: {e}")
            # 尝试使用备用模型
            self.model_name = "Qwen/Qwen2.5-1.7B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Qwen2.5 tokenizer 设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"加载模型失败: {e}")
            # 尝试不使用 device_map
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        # 配置LoRA参数
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )

        # 应用LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()

        # 定义情绪类别
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

        # 情绪分类器
        self.emotion_classifier = nn.Linear(
            self.base_model.config.hidden_size,
            len(self.emotions)
        ).to(self.base_model.device)

    def prepare_emotion_data(self, dataset_path):
        """
        准备情感分类数据
        """
        # 示例数据格式
        sample_data = [
            {
                "text": "今天天气真好，心情特别愉快",
                "emotion_probs": [0.0, 0.0, 0.0, 0.9, 0.0, 0.1, 0.0]
            },
            {
                "text": "这件事让我感到非常生气",
                "emotion_probs": [0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
            }
        ]

        # 如果你有自己的数据文件，取消下面的注释
        # with open(dataset_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)

        texts = []
        emotion_probs = []

        for item in sample_data:
            # 构建分类提示
            prompt = f"分析文本情感: {item['text']}\n情感分布:"
            texts.append(prompt)
            emotion_probs.append(item["emotion_probs"])

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(emotion_probs, dtype=torch.float32)
        })

        return dataset

    def custom_loss(self, model_outputs, emotion_labels):
        """
        自定义损失函数
        """
        # 获取最后一个token的隐藏状态
        last_hidden_state = model_outputs.logits[:, -1, :]

        # 通过分类器得到情感logits
        emotion_logits = self.emotion_classifier(last_hidden_state)

        # 使用KL散度损失
        predicted_probs = torch.softmax(emotion_logits, dim=-1)
        loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(predicted_probs + 1e-8),
            emotion_labels
        )

        return loss

    def train_model(self, dataset, output_dir="./emotion_model", epochs=3, learning_rate=2e-4):
        """
        训练模型
        """
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to=None,
            dataloader_pin_memory=False
        )

        # 自定义Trainer处理自定义损失
        class EmotionTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                # 前向传播
                outputs = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    return_dict=True
                )

                # 计算自定义损失
                loss = self.model.compute_custom_loss(outputs, inputs.get("labels"))

                return (loss, outputs) if return_outputs else loss

        # 添加自定义损失计算方法到模型
        self.model.compute_custom_loss = lambda outputs, labels: self.custom_loss(outputs, labels)

        trainer = EmotionTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        return trainer

    def predict_emotion(self, text):
        """
        预测情感分布
        """
        self.model.eval()

        prompt = f"分析文本情感: {text}\n情感分布:"

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            # 获取最后一个token的隐藏状态
            last_hidden_state = outputs.logits[:, -1, :]

            # 通过分类器
            emotion_logits = self.emotion_classifier(last_hidden_state)
            emotion_probs = torch.softmax(emotion_logits, dim=-1)

            return emotion_probs.cpu().numpy()[0]


# 简化版本 - 如果上述仍有问题
class SimpleEmotionTuner:
    def __init__(self, model_name="Qwen/Qwen2.5-1.7B"):
        """简化版本的情感调优器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()


def check_environment():
    """检查环境配置"""
    import importlib.metadata as metadata

    try:
        transformers_version = metadata.version('transformers')
        peft_version = metadata.version('peft')
        torch_version = metadata.version('torch')

        print(f"Transformers version: {transformers_version}")
        print(f"PEFT version: {peft_version}")
        print(f"PyTorch version: {torch_version}")

        # 检查CUDA
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name()}")

    except metadata.PackageNotFoundError as e:
        print(f"Package not found: {e}")


if __name__ == "__main__":
    # 检查环境
    check_environment()

    try:
        # 尝试初始化模型
        tuner = EmotionFineTuner()
        print("模型初始化成功!")

        # 准备示例数据
        dataset = tuner.prepare_emotion_data("sample_data.json")
        print("数据准备完成!")

        # 测试预测
        test_text = "今天心情很好"
        probs = tuner.predict_emotion(test_text)
        print("情感预测:", dict(zip(tuner.emotions, probs)))

    except Exception as e:
        print(f"错误: {e}")
        print("尝试使用简化版本...")

        simple_tuner = SimpleEmotionTuner()
        print("简化版本初始化成功!")