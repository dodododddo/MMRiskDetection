# coding=utf-8
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def has_digit_or_letter(s):
    return any(c.isalpha() or c.isdigit() for c in s)


def cut_word_single(text):
    words = jieba.lcut(text)  # 使用lcut函数返回一个列表，列表中的每个元素都是一个分词后的词语
    print("分词结果：", words)
    word_counts = Counter(words)
    print("词频统计：")
    for word, count in word_counts.items():
        print(f"{word}: {count}")


def cut_word(text):
    words = jieba.lcut(text)  # 使用lcut函数返回一个列表，列表中的每个元素都是一个分词后的词语
    # print("分词结果：", words)
    filtered_word_counts = Counter(word for word in words if (len(word) > 1 and len(word) < 5))
    # print("\n过滤单字后的词频统计：")
    # print(filtered_word_counts)
    return filtered_word_counts


def count_data(data_path):
    df_dataset = pd.read_csv(data_path, encoding='utf-8')
    word_freq = pd.DataFrame({'word': [], 'frequency': []})
    for i, text in enumerate(df_dataset['text']):
        dictionary = cut_word(text)
        for key, value in dictionary.items():
            if not has_digit_or_letter(key):
                continue
            if key in word_freq['word'].values:
                index = word_freq[word_freq['word'] == key].index[0]
                word_freq.at[index, 'frequency'] += value
            else:
                word_freq = word_freq._append({'word': key, 'frequency': value}, ignore_index=True)
        print(i)
    word_freq.to_csv("DataPipeline/output/dialog/image/p2_wf.csv", index=False, mode='w')

    return word_freq


def paint_histogram(data_frequency):
    df = pd.read_csv(data_frequency)
    sorted_df = df.sort_values(by='frequency', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_df.index, sorted_df['frequency'], color='skyblue')
    plt.xlabel('Vocabulary')
    plt.ylabel('Frequency')
    plt.ylim([0, 120000])

    plt.grid(True)
    plt.show()
    plt.savefig("DataPipeline/output/dialog/image/1.png")

def remove_stopwords(data_frequency):
    stopwords_path = "DataPipeline/stopwords.txt"
    stop_words = []
    with open(stopwords_path,'r') as f:
        for line in f:
            stop_words.append(line.strip())
    # stop_words = pd.read_csv(stopwords_path, encoding='utf-8')
    df = pd.read_csv(data_frequency)
    # print(stop_words)
    print(len(df))
    for index, row in df.iterrows():
        # print(row['word'])
        if row['word'] in stop_words:
            df.drop(index, inplace=True)
            continue
        if len(row['word']) > 4:
            df.drop(index, inplace=True)
            continue
        if " " in row['word'] or "." in row['word'] or "%" in row['word']:
            df.drop(index, inplace=True)
            continue
        if row['word'].isdigit():
            df.drop(index, inplace=True)
            continue
        # if has_digit_or_letter(str(row['word'])):
        #     df.drop(index, inplace=True)
        #     continue
    print(df)
    df.to_csv("DataPipeline/output/dialog/image/p1_word_frequency.csv", index=False, mode='w')


if __name__ == '__main__':
    data_path = "DataPipeline/output/dialog/prompt2_generate_8w.csv"
    data_frequency = "DataPipeline/output/dialog/image/p1_wf.csv"
    # count_data(data_path)
    remove_stopwords(data_frequency)
    