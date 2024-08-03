import math
import pandas as pd

def calculate_simpson_index(word_counts):
    total_words = sum(word_counts.values())
    numerator = sum(count * (count - 1) for count in word_counts.values())
    denominator = total_words * (total_words - 1)
    
    if denominator == 0:
        return 0  # 处理异常情况，如空文本
    
    simpson_index = 1 - numerator / denominator
    return simpson_index


word_counts_df = pd.read_csv(r"DataPipeline/output/dialog/image/p2_word_frequency.csv", encoding='utf-8')
word_counts = dict(zip(word_counts_df['word'], word_counts_df['frequency']))

# print(word_counts)

# 计算Simpson多样性指数
simpson_index = calculate_simpson_index(word_counts)
print("Simpson多样性指数:", simpson_index)
