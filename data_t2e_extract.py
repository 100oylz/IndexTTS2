import json


def process_multiple_entries(data_list):
    """
    处理多个数据条目
    """
    all_results = []
    emotion_order = ['Anger', 'Happiness', 'Fear', 'Disgust', 'Sadness', 'Surprise', 'Neutral']

    for data in data_list:
        for pair in data['text_emotion_pairs']:
            text = pair['text']
            emotion_dict = pair['emotion_distribution']

            emotion_vector = [emotion_dict[emotion] for emotion in emotion_order]

            all_results.append({
                'text': text,
                'emotion_vector': emotion_vector,
                'original_id': data.get('id'),
                'timestamp': data.get('timestamp')
            })

    return all_results

# 如果有多个数据条目的示例用法
# multiple_data = [data1, data2, data3]  # 你的多个数据
# results = process_multiple_entries(multiple_data)

import json


def load_and_process_jsonl(file_path):
    """
    加载JSONL文件并提取text和emotion_distribution
    """
    results = []
    emotion_order = ['Anger', 'Happiness', 'Fear', 'Disgust', 'Sadness', 'Surprise', 'Neutral']

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # 解析JSON行
                    data = json.loads(line)

                    # 检查是否有text_emotion_pairs字段
                    if 'text_emotion_pairs' in data:
                        for pair in data['text_emotion_pairs']:
                            text = pair.get('text', '')
                            emotion_dict = pair.get('emotion_distribution', {})

                            # 转换为7维向量
                            emotion_vector = [emotion_dict.get(emotion, 0.0) for emotion in emotion_order]

                            # 验证总和
                            total = sum(emotion_vector)
                            if abs(total - 1.0) > 1e-10:
                                print(f"警告: 第{line_num}行情绪分布总和不为1, 当前总和: {total}")

                            results.append({
                                'text': text,
                                'emotion_vector': emotion_vector,
                                'original_id': data.get('id'),
                                'timestamp': data.get('timestamp')
                            })
                    else:
                        print(f"警告: 第{line_num}行缺少text_emotion_pairs字段")

                except json.JSONDecodeError as e:
                    print(f"错误: 第{line_num}行JSON解析失败 - {e}")
                    continue

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return []

    return results


# 使用示例
if __name__ == "__main__":
    file_path = "./data/processed_emotion_data.jsonl"  # 替换为你的JSONL文件路径

    # 加载和处理数据
    extracted_data = load_and_process_jsonl(file_path)

    print(f"成功处理 {len(extracted_data)} 条数据")

    # 显示前几条结果
    for i, item in enumerate(extracted_data[:3]):  # 只显示前3条
        print(f"\n条目 {i + 1}:")
        print(f"文本: {item['text'][:50]}...")  # 只显示前50个字符
        print(f"情绪向量: {item['emotion_vector']}")
        print(f"向量总和: {sum(item['emotion_vector'])}")