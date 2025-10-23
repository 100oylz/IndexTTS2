import json


class JsonlDecoder():
    EMOTIONS = ["Anger", "Happiness", "Fear", "Disgust", "Sadness", "Surprise", "Neutral"]
    EPS = 1e-6

    def __init__(self, file_path):
        self.data = self._load_jsonl(file_path)
        self.text = None
        self.emotions = None
        self.length = len(self.data)
        self.get_text()
        self.get_emotion()

    def _load_jsonl(self, file_path):
        """专门加载JSON Lines文件的方法：按行读取并解析每个JSON对象"""
        data = []
        with open(file_path, 'r', encoding="utf8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()#去除首尾空白字符
                if not line:  # 跳过空行
                    continue
                try:
                    # 解析每行的JSON对象
                    data.append(json.loads(line))#试一下这句话如果发生错误则转入except并raise清晰错误原因
                except json.JSONDecodeError as e:#错误对象赋值给e并抛出清晰说明raise后
                    raise ValueError(f"第{line_num}行JSON格式错误: {e}")
        return data

    def get_text(self):
        self.text = []
        for item in self.data:
            assert 'sentence' in item.keys(), f"'sentence' not in {item.keys()}"#若assert后条件不满足抛出f-str格式的错误
            self.text.append(item['sentence'])

    def get_emotion(self):
        self.emotions = []
        for item in self.data:
            assert "emotion_distribution" in item.keys(), f"'emotion_distribution' not in {item.keys()}"
            emotion_distribution = item["emotion_distribution"]
            if isinstance(emotion_distribution, dict):#以下if=elif语句处理又有字典又有列表格式的数据
                emotion = [emotion_distribution[k] for k in JsonlDecoder.EMOTIONS]
            elif isinstance(emotion_distribution, list):
                emotion_distribution_1 = {}
                for e in emotion_distribution:
                    assert "emotion" in e.keys()
                    assert "probability" in e.keys()
                    emotion_distribution_1[e["emotion"]] = e["probability"]
                emotion = [emotion_distribution_1[k]
                           for k in JsonlDecoder.EMOTIONS]

            else:
                raise NotImplementedError
            s = sum(emotion)
            assert s - 1 < JsonlDecoder.EPS, f"{s}!=1"
            self.emotions.append(emotion)






#以下旧
import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset
import pandas as pd
import tokenizers
from collections import OrderedDict
emotion_distribution_example=OrderedDict({'Anger': 0.85, 'Happiness': 0.01, 'Fear': 0.05, 'Disgust': 0.03, 'Sadness': 0.02, 'Surprise': 0.02, 'Neutral': 0.02})
class JsonlDataProcessor:
    def __init__(self, file_path, tokenizer, max_length=256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.emotions = list(emotion_distribution_example.keys())

    def load_data(self):
        """加载 .jsonl 文件"""
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"JSON 解析错误: {e}, 行内容: {line[:100]}...")

        print(f"成功加载 {len(data)} 条数据")
        return data

    def preprocess_data(self, data):
        """预处理数据"""
        texts = []
        emotion_probs = []

        for item in data:
            # 检查数据格式
            if 'sentence' not in item.keys() or 'emotion_distribution' not in item.keys():
                print(f"跳过无效数据: {item}")
                continue

            # 构建分类提示
            prompt = f"分析文本情感: {item['sentence']}\n情感分布:(情感分布七个特征为:{self.emotions})"
            texts.append(prompt)
            probs=[value for _,value in OrderedDict(item['emotion_distribution']).items()]
            emotion_probs.append(probs)

        # Tokenize 文本
        tokenized = self.tokenizer.batch_encode(
            texts,
            out_type=int,
        )
        print(emotion_probs)
        print(len(emotion_probs))
        # 创建数据集
        dataset = Dataset.from_dict({
            "input_ids": tokenized,
            # "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(emotion_probs, dtype=torch.float32)
        })

        return dataset

    def train_test_split(self, dataset, test_size=0.2, random_seed=42):
        """划分训练集和测试集"""
        dataset = dataset.train_test_split(
            test_size=test_size,
            seed=random_seed
        )
        return dataset['train'], dataset['test']

    def analyze_data(self, data):
        """分析数据分布"""
        print("\n数据分布分析:")
        print(f"总样本数: {len(data)}")

        # 统计每个情绪的平均概率
        emotion_stats = {emotion: [] for emotion in self.emotions}

        for item in data:
            if 'emotion_probs' in item:
                probs = item['emotion_probs']
                for i, emotion in enumerate(self.emotions):
                    if i < len(probs):
                        emotion_stats[emotion].append(probs[i])

        print("各情绪平均概率:")
        for emotion in self.emotions:
            if emotion_stats[emotion]:
                avg_prob = sum(emotion_stats[emotion]) / len(emotion_stats[emotion])
                print(f"  {emotion}: {avg_prob:.4f}")


if __name__ == '__main__':
    files="example.jsonl"
    from indextts.utils.front import TextTokenizer,TextNormalizer
    normalizer=TextNormalizer()
    tokenizer=TextTokenizer(vocab_file="checkpoints/bpe.model",normalizer=normalizer)
    max_length = 10240
    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    processer:JsonlDataProcessor=JsonlDataProcessor(file_path=files,tokenizer=tokenizer,max_length=max_length)
    data=processer.load_data()
    # print(data)
    preprocessed_data=processer.preprocess_data(data)

    # print(preprocessed_data)