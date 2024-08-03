import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from utils import batch_iterator
import statistics
import numpy as np

    
model = AutoModel.from_pretrained(
    "/data1/home/jrchen/MMRiskDetection/DataPipeline/reward_model/internlm2-7b-reward", 
    device_map="cuda", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("/data1/home/jrchen/MMRiskDetection/DataPipeline/reward_model/internlm2-7b-reward", trust_remote_code=True)

def average(source_file):
    with open(source_file) as f:
        dataset = json.load(f)
        dataset = [x['messages'] for x in dataset]


    all_scores = []
    for batch in batch_iterator(dataset, 20):
        scores = model.get_scores(tokenizer, batch)
        all_scores += scores

    # print(scores)      #列表
    print(statistics.mean(all_scores))     #平均值

def select(source_file,dst_file):
    with open(source_file) as f:
        dataset = json.load(f)
        dataset = [x['messages'] for x in dataset]

    all_scores = []
    for batch in batch_iterator(dataset, 20):
        scores = model.get_scores(tokenizer, batch)
        all_scores += scores

    dataset1 = [[x[1]['content'],x[2]['content']] for x in dataset]

    scores_column = np.expand_dims(all_scores, axis=-1)  # 转换成列向量
    data_score = np.concatenate((dataset1, scores_column), axis=-1)
    data_score = data_score.tolist()
    # print(data_score)
    print(len(data_score))

    data_sorted = np.array(sorted(data_score, key=lambda x: x[2]))  # 按每个子列表的第3个元素排序，并转换为numpy数组

    index_30_percent = int(len(data_sorted) * 0.5)    

    data_sorted = data_sorted[index_30_percent:]  # 这将保留索引 idx 及之后的所有元素

    new_data = []
    all_score = []
    for item in data_sorted:
        newdata = {}
        newdata['文本'] = item[0]
        newdata['风险类别'] = item[1]
        newdata['评价'] = item[2]
        all_score.append(float(item[2]))
        new_data.append(newdata)
    print("平均值为" + str(statistics.mean(all_score)))

    # new_data = [item[0] for item in data_sorted]

    # 将修改后的数据写入新的JSON文件
    with open(dst_file, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    source_file = 'DataPipeline/output/dialog/table_new/p2_selected_reward.json'
    dst_file = 'DataPipeline/output/dialog/table_new/p2_selected.json'
    # select(source_file,dst_file)
    average(source_file)

