from transformers import AutoTokenizer
from FlagEmbedding import FlagModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import torch
import argparse
import os

def compute_matrix_cosine_similarity(embeddings):
    # x1 = embeddings1.unsqueeze(1)  # 变成(m, 1, n)
    # x2 = embeddings2.unsqueeze(0)  # 变成(1, m, n)
    # cos_sim = torch.nn.functional.cosine_similarity(x1, x2, dim=2)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    cos_sim = embeddings @ embeddings.T
    # print(cos_sim.shape)
    return cos_sim.cpu()

def compute_mean_cosine_similarity(embeddings):
    mean_embedding = torch.mean(embeddings, dim=0)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings, mean_embedding.unsqueeze(0), dim=1)
    mean_cos_sim = torch.mean(cos_sim,dim=0)
    return mean_cos_sim

def select_least_similar(data, labels, m, n_sample):
    unique_labels = torch.unique(labels)
    selected_indices = [[] for _ in unique_labels]

    # 计算每个类别的中心
    class_centers = []
    for label in unique_labels:
        indices = torch.nonzero(labels == label, as_tuple=False).squeeze()
        indices = indices[indices<=n_sample]
        class_center = torch.mean(data[indices[indices<n_sample]], dim=0)
        class_centers.append(class_center)
    class_centers_tensor = torch.stack(class_centers)

    for i, label in enumerate(unique_labels):
        # 该类所有数据在data中的索引
        indices = torch.nonzero(labels == label, as_tuple=False).squeeze()
        mean_embedding = class_centers_tensor[i]
        # 该类所有数据与该类中心计算余弦相似度
        similarities_to_center = compute_matrix_cosine_similarity(data[indices[indices<n_sample]], mean_embedding)
        # 数据按余弦相似度大小排列 及 原索引的排列
        selected_cosine_data, sorted_indices = torch.topk(similarities_to_center, len(similarities_to_center), largest=False)
        j = 0
        
        m = len(sorted_indices)
        if len(sorted_indices) <= m-1:
            selected_indices[i] = sorted_indices
        else:
            c = 0
            while len(selected_indices[i]) < m and j < len(sorted_indices):
            # while len(selected_indices[i]) < m:
                idx = indices[sorted_indices[j]].item()
                # print(f"len(selected_indices[i]):{len(selected_indices[i])}     j:{j}")
                selected_single_data = data[idx]
                similarities_to_all_centers = compute_matrix_cosine_similarity(class_centers_tensor, selected_single_data)
                maxarg_similarities_to_center = torch.argmax(similarities_to_all_centers)
                if unique_labels[maxarg_similarities_to_center] != label:
                    c+=1
                    j += 1
                    continue
                else:
                    selected_indices[i].append(idx)
                    j += 1
        #     print(c)
        # print(len(selected_indices[i]))
    return selected_indices

def select_least_similar_sample(df_dataset, embeddings, riskType, m, n_sample):
    tensor = torch.tensor(riskType.values)
    selected_data_index = select_least_similar(embeddings, tensor, m, n_sample)
    index_list = []
    for i, index in enumerate(selected_data_index):
        index = torch.tensor(index)
        index_list = index_list + index.tolist()
    selected_df = df_dataset.iloc[index_list]
    selected_df.to_csv("DataPipeline/output/dialog/prompt2_unrepeated_selected2.csv", encoding='utf-8')

def select_represent_sample(model, data_path, embedding_store_path, sample_store_path, category_labels, n_sample, top_k):
    df_dataset = pd.read_csv(data_path, encoding='utf-8')
    def find_index(x):
        return category_labels.index(x) if x in category_labels else None
    
    df_dataset['type_num'] = df_dataset['riskType'].apply(find_index)

    raw_dataset = df_dataset.to_dict(orient='records')
    # with open(data_path, 'r') as f:
    #     raw_dataset = json.load(f)
    # if n_sample > 0:
    #     raw_dataset = raw_dataset[:n_sample]
    
    dataset = [[] for i in range(len(category_labels))]
    for data in raw_dataset:
        index = category_labels.index(data['riskType'])
        if index != -1:
            # data.pop('f_index')
            dataset[index].append(data['text'])

    def all_data():
        texts = []
        for docs in tqdm(dataset):  # 分类
            texts = texts + [str(x) for x in docs]
        embeddings = model.encode(texts)
        embeddings = torch.tensor(embeddings).to(torch.float32)
        mean_cosine = compute_mean_cosine_similarity(embeddings)
        print("total: "+str(len(texts))+"条, "+str(mean_cosine))
        # cosine = compute_matrix_cosine_similarity(embeddings,embeddings)
        # print(cosine.shape)
        # df = pd.DataFrame(cosine.numpy())
        # df.to_csv("DataPipeline/output/dialog/visualization.csv")
        # select_least_similar_sample(df_dataset, embeddings, df_dataset['type_num'], 100, n_sample)

    def sum_by_row(embeddings, k, i):
        cosine = compute_matrix_cosine_similarity(embeddings)
        df = pd.DataFrame(cosine.numpy())
        df['sum'] = df.sum(axis=1)
        sorted_index = df['sum'].sort_values().index
        top_k_index = sorted_index[:k]
        # print(top_k_index)
        top_k_df = df_dataset[df_dataset['riskType']==category_labels[i]].iloc[top_k_index]
        # print(top_k_df)
        top_k_df.to_csv(f"DataPipeline/output/dialog/prompt1/sum_by_row_selected_8w_0.4.csv",index=False,encoding="utf-8",mode='a')
        return top_k_df
    
    def greedy_select(embeddings, k, i):
        embeddings=embeddings.cuda()
        cosine = compute_matrix_cosine_similarity(embeddings)
        print(cosine.shape)
        df = pd.DataFrame(cosine.numpy())
        df = df.fillna(0)
        selected_indices = []
        # while len(set(selected_indices)) < k:
        # min_value = np.min(df.values[np.triu_indices(df.shape[0], k=1)])
        # min_indices = np.argwhere(df.values == min_value)[0]
        # selected_indices.append(min_indices[0])
        # selected_indices.append(min_indices[1])
        max_indices_s = np.argwhere(df.values<0.324)
        print(len(max_indices_s))
        for max_indices in max_indices_s:
            if max_indices[0] in selected_indices and max_indices[0] in selected_indices:
                continue
            if max_indices[0] not in selected_indices:
                selected_indices.append(max_indices[0])
                # print(df_dataset[df_dataset['riskType']==category_labels[i]].iloc[max_indices[0]])
            if max_indices[1] not in selected_indices:
                selected_indices.append(max_indices[1])
                # print(df_dataset[df_dataset['riskType']==category_labels[i]].iloc[max_indices[1]])
            print(len(selected_indices))
        print(selected_indices)
        print(len(max_indices_s))
        print(len(selected_indices))
        
        top_k_df = df_dataset[df_dataset['riskType']==category_labels[i]].iloc[selected_indices]
        top_k_df.to_csv(f"DataPipeline/output/dialog/prompt2/greedy_selected_8w_0.4 copy.csv",index=False,encoding="utf-8",mode='w')
        return top_k_df

    def groupby_data():
        texts = []
        for i, docs in enumerate(dataset):  # 分类
            texts = [str(x) for x in docs]
            if len(texts)==0:
                continue
            embeddings = model.encode(texts)
            embeddings = torch.tensor(embeddings).to(torch.float32)
            mean_cosine = compute_mean_cosine_similarity(embeddings)
            print(category_labels[i]+": "+str(len(texts))+"条, "+str(mean_cosine))

            if i == 0:
                k = int(len(texts) * 0.4)
                # df_sum_by_row = sum_by_row(embeddings, k, i)
                df_greedy_select = greedy_select(embeddings, k, i)
                break
    
    # all_data()
    groupby_data()

def main(args):
    model = FlagModel("./TextPipeline/model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示",
                  use_fp16=True)
    
    category_labels = [ '无风险', '虚假购物、服务类', '冒充电商物流客服类', '冒充领导、熟人类',
                        '虚假信用服务类', '冒充公检法及政府机关类',  '虚假网络投资理财类', 
                        '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件'] # 未完成
    # '''
    embedding_store_path = os.path.join(args.embedding_store_root_path, f"represent_embedding_{args.n_sample if args.n_sample > 0 else 'all'}_top_{args.top_k if args.top_k > 0 else 'all'}.pth")
    sample_store_path = os.path.join(args.sample_store_root_path, f"represent_sample_{args.n_sample if args.n_sample > 0 else 'all'}_top_{args.top_k if args.top_k > 0 else 'all'}.json")
    select_represent_sample(model, args.data_path, embedding_store_path, sample_store_path, category_labels, args.n_sample, args.top_k)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process embeddings and samples.')

    # 设置默认值
    parser.add_argument('--embedding_store_root_path', type=str, default='dialog/',
                        help='Path to the embedding storage file.')
    
    parser.add_argument('--data_path', type=str, default='./DataPipeline/output/dialog/prompt2_generate_8w.csv',
                        help='Path to the data file.')
    
    parser.add_argument('--sample_store_root_path', type=str, default='dialog/',
                        help='Path to the sample storage file.')
    
    parser.add_argument('--n_sample', type=int, default=1200,
                        help='Number of samples.-1 means use all sample')
    
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top K value. -1 means use all sample')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)


