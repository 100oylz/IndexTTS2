import random
from xpinyin import Pinyin


def random_replace_pinyin(text, replacement_ratio=0.3):
    """
    将文本中的一部分汉字随机替换为带数字声调的拼音

    Args:
        text (str): 输入的中文文本
        replacement_ratio (float): 替换比例，默认为0.3（30%的汉字会被替换）

    Returns:
        str: 替换后的文本
    """
    # 初始化拼音转换器
    p = Pinyin()

    result_chars = []

    for char in text:
        # 判断是否为中文字符
        if '\u4e00' <= char <= '\u9fff':
            # 随机决定是否替换
            if random.random() < replacement_ratio:
                # 转换为带数字声调的拼音
                pinyin_char = p.get_pinyin(char, tone_marks='numbers', splitter='')
                result_chars.append(pinyin_char)
            else:
                result_chars.append(char)
        else:
            # 非中文字符直接保留
            result_chars.append(char)

    return ''.join(result_chars)


# 测试示例
if __name__ == "__main__":
    # 示例文本
    test_text = "今天天气很好，我们一起去公园玩吧。"

    # 设置随机种子以便结果可重现（实际使用时可以去掉）
    random.seed(42)

    # 进行替换，替换比例为30%
    result = random_replace_pinyin(test_text, 0.3)

    print(f"原始文本: {test_text}")
    print(f"替换后: {result}")

    # 多次运行可以看到随机替换的效果
    print("\n多次运行示例:")
    for i in range(3):
        result = random_replace_pinyin(test_text, 0.3)
        print(f"第{i + 1}次: {result}")