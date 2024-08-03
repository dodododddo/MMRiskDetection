# %%
from FlagEmbedding import FlagModel
from FlagEmbedding import FlagReranker
from scipy.spatial.distance import cosine
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

def draw():
    plt.figure()
    plt.bar(x - width/2, scores_1, width, label='Prompt 1', color='blue')
    plt.bar(x + width/2, scores_2, width, label='Prompt 2', color='orange')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

def list_sorted(nums):
    indexes = list(range(len(nums)))
    indexed_nums = list(zip(nums, indexes))
    sorted_indexes = [index for value, index in sorted(indexed_nums)]
    return sorted_indexes

def encode_correlation(question, data_path):
    model = FlagModel("./TextPipeline/model/bge-reranker-large")
    df_dataset = pd.read_csv(data_path, encoding='utf-8')
    df_dataset = df_dataset[:4]
    data_texts = df_dataset['text'].tolist()
    print(df_dataset['text'])
    question_embeddings = model.encode([question])[0]
    data_embeddings = model.encode(data_texts)
    similarities = [1 - cosine(question_embeddings, emb) for emb in data_embeddings]
    print("问题与每条数据的相似度：", similarities)

def doc_correlation(question, data_path_1, data_path_2, k):
    reranker = FlagReranker('./TextPipeline/model/bge-reranker-large', use_fp16=True)
    def data_group(data_path, k):
        df_dataset = pd.read_csv(data_path, encoding='utf-8')
        # df_dataset = df_dataset[:k]
        # df_dataset['text'] = df_dataset['index'].astype(str) + df_dataset['text']
        results = df_dataset.groupby('index')['text'].agg(list).reset_index()
        results = results['text'].values
        result_list = [[question,'\t'.join(result)] for result in results]
        return df_dataset, result_list

    def data_group_short(data_path, k):
        df_dataset = pd.read_csv(data_path, encoding='utf-8')
        # df_dataset = df_dataset[:k]
        result_list = [[question, df_dataset.iloc[i]['text'] + "，这是它对应的风险类型：" + df_dataset.iloc[i]['riskType']] for i in range(len(df_dataset))]
        return df_dataset, result_list

    df_dataset1, result_list_1 = data_group_short(data_path_1, k)
    df_dataset2, result_list_2 = data_group_short(data_path_2, k)
    # print(result_list_1)

    
    
    scores_1 = reranker.compute_score(result_list_1, normalize=True)
    scores_2 = reranker.compute_score(result_list_2, normalize=True)
    # scores = reranker.compute_score(result_list)
    
    print(question)
    # print(scores_1)
    # print(scores_2)
    # # print()
    mean1 = np.mean(scores_1)
    mean2 = np.mean(scores_2)

    # 计算峰峰值
    peak_to_peak1 = np.ptp(scores_1)
    peak_to_peak2 = np.ptp(scores_2)

    # 输出结果
    print(f"prompt1的平均值为 {mean1}, 峰峰值为 {peak_to_peak1}")
    print(f"prompt2的平均值为 {mean2}, 峰峰值为 {peak_to_peak2}")

    sorted_scores_1 = sorted(scores_1, reverse=True)[:int(0.5*len(scores_1))]
    sorted_mean1 = np.mean(sorted_scores_1)
    sorted_scores_2 = sorted(scores_2, reverse=True)[:int(0.5*len(scores_2))]
    sorted_mean2 = np.mean(sorted_scores_2)

    print(f"筛选后prompt1的平均值为 {sorted_mean1}")
    print(f"筛选后prompt2的平均值为 {sorted_mean2}")


    
    def write_file(df_dataset, scores_1, result_list_1, p):
        # score_selected = sorted(scores_1, reverse=True)[:int(0.5*len(scores_1))]
        # list_selected = [i for i, score in enumerate(scores_1) if score in score_selected]

        sorted_indices = sorted(range(len(scores_1)), key=lambda i: scores_1[i], reverse=True)[:int(0.5*len(scores_1))]
        # for i in sorted_indices:
        #     df_dataset.iloc[2*i].to_csv(f'DataPipeline/output/dialog/table_new/{p}_corr_select.csv',mode='a',header=False,index=False)
        #     df_dataset.iloc[2*i+1].to_csv(f'DataPipeline/output/dialog/table_new/{p}_corr_select.csv',mode='a',header=False,index=False)
            

        # with open(f'DataPipeline/output/dialog/table_new/{p}_corr.txt', 'w') as file:
        #     for i, item in enumerate(result_list_1):
        #         position = item[1].find('【')
        #         index = item[1][:position]
        #         # print("data "+str(i))
        #         file.write(f"{index}\n")
        #         df_dataset.iloc[int(index)].to_csv(f'DataPipeline/output/dialog/table_new/{p}_corr_select.csv',header=False,index=False)

    # write_file(df_dataset1, scores_1, result_list_1, 'p1')
    # write_file(df_dataset2, scores_2, result_list_2, 'p2')
    
    def write_low_file():
        list1 = [i for i, score in enumerate(scores_1) if score < mean1]
        text_data = [result_list_1[i][1] for i in list1]
        with open('DataPipeline/output/dialog/corr.txt', 'w') as file:
            for item in text_data:
                file.write(f"{item}\n")
        list2 = [i for i, score in enumerate(scores_2) if score < mean2]
        text_data = [result_list_1[i][1] for i in list2]
        with open('DataPipeline/output/dialog/corr.txt', 'a') as file:
            file.write(f"\n")
            for item in text_data:
                file.write(f"{item}\n")
                
    # write_low_file()

def write(question):
    k = 80000
    data_path_1 = "DataPipeline/output/dialog/prompt1_generate_8w.csv"
    data_path_2 = "DataPipeline/output/dialog/prompt2_generate_8w.csv"
    def data_group(data_path, k):
        df_dataset = pd.read_csv(data_path, encoding='utf-8')
        df_dataset = df_dataset[:k]
        results = df_dataset.groupby('index')['text'].agg(list).reset_index()
        results = results['text'].values
        result_list = [[question,'\t'.join(result)] for result in results]
        return result_list

    result_list_1 = data_group(data_path_1, k)
    result_list_2 = data_group(data_path_2, k)

    file_path1 = "DataPipeline/output/dialog/table/p1_score.txt"
    file_path2 = "DataPipeline/output/dialog/table/p2_score.txt"
    def read(file_path):
        with open(file_path, 'r') as file:
            content = file.read().strip()
        my_list = eval(content)
        return my_list

    def write_file(scores_1, result_list_1, p):
        score_selected = sorted(scores_1, reverse=True)[:int(0.4*len(scores_1))]
        list_selected = [i for i, score in enumerate(scores_1) if score in score_selected]
        with open(f'DataPipeline/output/dialog/table/{p}_corr.txt', 'w') as file:
            for i, item in enumerate(result_list_1):
                print("data "+str(i))
                file.write(f"{item}\n")
                if i >= 16000:
                    break

    scores_1 = read(file_path1)
    scores_2 = read(file_path2)
    write_file(scores_1, result_list_1, 'p1')
    write_file(scores_2, result_list_2, 'p2')

def selected_correlation(data_path_1, data_path_2):
    reranker = FlagReranker('./TextPipeline/model/bge-reranker-large', use_fp16=True)
    
    def read(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        lines = [eval(line.strip()) for line in lines]
        return lines

    result_list_1 = read(data_path_1)
    result_list_2 = read(data_path_2)
    
    scores_1 = reranker.compute_score(result_list_1, normalize=True)
    scores_2 = reranker.compute_score(result_list_2, normalize=True)

    print(len(scores_1))
    print(len(scores_2))
    mean1 = np.mean(scores_1)
    mean2 = np.mean(scores_2)

    # 计算峰峰值
    peak_to_peak1 = np.ptp(scores_1)
    peak_to_peak2 = np.ptp(scores_2)

    # 输出结果
    print(f"prompt1的平均值为 {mean1:.2f}, 峰峰值为 {peak_to_peak1}")
    print(f"prompt2的平均值为 {mean2:.2f}, 峰峰值为 {peak_to_peak2}")

def calculate_score():
    file_path1 = "DataPipeline/output/dialog/table/p1_score.txt"
    file_path2 = "DataPipeline/output/dialog/table/p2_score.txt"
    def read(file_path):
        with open(file_path, 'r') as file:
            content = file.read().strip()
        my_list = eval(content)
        return my_list
    scores_1 = read(file_path1)
    scores_2 = read(file_path2)

    def average_of_top_40_percent(lst):
        top_count = int(len(lst) * 0.4)
        if top_count == 0:
            return None
        top_elements = sorted(lst)[-top_count:]
        avg_top = sum(top_elements) / len(top_elements)
        return avg_top

    mean1_before = sum(scores_1) / len(scores_1)
    mean2_before = sum(scores_2) / len(scores_2)
    print(f"筛选前prompt1的平均值为 {mean1_before}")
    print(f"筛选前prompt2的平均值为 {mean2_before}")

    mean1_after = average_of_top_40_percent(scores_1)
    mean2_after = average_of_top_40_percent(scores_2)
    print(f"筛选后prompt1的平均值为 {mean1_after}")
    print(f"筛选后prompt2的平均值为 {mean2_after}")

def concat():
    input_file = 'DataPipeline/output/dialog/table_new/p2_corr_select.csv'  # 替换为你的 CSV 文件路径
    output_file = 'DataPipeline/output/dialog/table_new/p2_corr_select_new.csv'  # 输出文件路径

    df = pd.read_csv(input_file)
    new_rows = []

    # 每5行组成新的一行
    for i in range(0, len(df), 4):
        row_slice = df.iloc[i:i+4]
        concatenated_row = ','.join(row_slice.astype(str).values.flatten())
        new_rows.append(concatenated_row)

    # 将新的数据写入输出文件
    with open(output_file, 'w') as f:
        for row in new_rows:
            f.write(row + '\n')

    print(f"已将每5行合并为新行的结果写入到 {output_file}")

'''输入输出示例
score = reranker.compute_score(['query', 'passage'])
print(score)
scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
'''

if __name__ == '__main__':
    question_text_1 = "诈骗对话"
    question_text_2 = '''给出两段通话记录，以'\t'进行分隔。
    第一段对话：【坐席】是否试图用冒充身份或诱导操作的方式，向【客户】谋取个人财产或隐私信息？
    第二段对话：第二段对话是否与第一段对话类似，但是是一段正常的无风险对话？'''
    question_text_3 = '''
            给出两段通话记录，以'\\t'进行分隔。第一段对话中【坐席】试图用冒充身份和诱导转账的方法进行诈骗，谋取【客户】的个人财产和信息；
            第二段对话与第一段对话语境类似，但是一段正常的无风险对话。
    '''

    # question_text_4 = "你可以给我找一段诈骗文本吗？我想用它来做反诈测试，所以请你把这条风险文本对应的风险类型也给我，方便我判断诈骗文本的类型，谢谢！"
    question_text_4 = "请给出一段用于反诈测试的通话记录，并给出这条风险文本对应的风险类型。对话在【坐席】和【客户】之间进行，若对话中含有一定的诈骗风险，则【坐席】试图用冒充身份和诱导转账的方法进行诈骗，谋取【客户】的个人财产和信息。"
    # question_text_4 = "请给出一段用于反诈测试的通话记录，对话在【坐席】和【客户】之间进行，并判断出这条风险文本对应的风险类型。"
    
    data_path_1 = "DataPipeline/output/dialog/prompt1_generate_8w.csv"
    data_path_2 = "DataPipeline/output/dialog/prompt2_generate_8w.csv"

    # data_path_1 = "DataPipeline/output/dialog/table_new/p1_selected copy.csv"
    # data_path_2 = "DataPipeline/output/dialog/table_new/p2_selected copy.csv"
    # correlation(question_text, data_path)
    # doc_correlation(question_text_1, data_path)
    # doc_correlation(question_text_2, data_path_1, data_path_2, 50)
    doc_correlation(question_text_4, data_path_1, data_path_2, 50)

    # write(question_text_3)
    data_path_1 = "DataPipeline/output/dialog/table/p1_corr.txt"
    data_path_2 = "DataPipeline/output/dialog/table/p2_corr.txt"
    # selected_correlation(data_path_1, data_path_2)

    # calculate_score()
    # concat()


    max = '''
            给出两段通话记录，以'\\t'进行分隔。第一段对话中【坐席】试图用冒充身份和诱导转账的方法进行诈骗，谋取【客户】的个人财产和信息；
            第二段对话与第一段对话语境类似，但是一段正常的无风险对话。 64  48
    '''