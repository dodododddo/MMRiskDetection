from transformers import AutoTokenizer
from FlagEmbedding import FlagModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import torch
import argparse
import os

# def compute_matrix_euclidean_distance(tensor1, tensor2):
#     tensor1 = tensor1.unsqueeze(1)  # 变成 (m, 1, n)
#     tensor2 = tensor2.unsqueeze(0)  # 变成 (1, m, n)
#     euclidean_distances = torch.norm(tensor1 - tensor2, p=2, dim=2)
#     return euclidean_distances

def compute_matrix_euclidean_distance(A, B):
    A_sum = (A**2).sum(dim=1).unsqueeze(1)
    B_sum = (B**2).sum(dim=1).unsqueeze(0)
    # 计算行向量之间的平方差
    distances = A_sum + B_sum - 2 * torch.mm(A, B.t())
    
    # 计算欧氏距离
    distances = torch.sqrt(distances)
    
    return distances

def compute_mean_euclidean_distance(embeddings):
    mean_embedding = torch.mean(embeddings, dim=0)
    euclidean_dist = torch.norm(embeddings - mean_embedding, p=2, dim=1)
    mean_euclidean_dist = torch.mean(euclidean_dist)
    return mean_euclidean_dist

def select_farthest_distance(data, labels, m, n_sample):
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
        distances_to_center = compute_matrix_euclidean_distance(data[indices[indices<n_sample]], mean_embedding)
        # 数据按余弦相似度大小排列 及 原索引的排列
        selected_distance_data, sorted_indices = torch.topk(distances_to_center, len(distances_to_center), largest=False)
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
                distances_to_all_centers = compute_matrix_euclidean_distance(class_centers_tensor, selected_single_data)
                maxarg_distances_to_center = torch.argmax(similarities_to_all_centers)
                if unique_labels[maxarg_distances_to_center] != label:
                    c+=1
                    j += 1
                    continue
                else:
                    selected_indices[i].append(idx)
                    j += 1
    return selected_indices

def select_farthest_distance_sample(df_dataset, embeddings, riskType, m, n_sample):
    tensor = torch.tensor(riskType.values)
    selected_data_index = select_farthest_distance(embeddings, tensor, m, n_sample)
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
        mean_distance = compute_mean_euclidean_distance(embeddings)
        print("total: "+str(len(texts))+", "+str(mean_distance))
        # cosine = compute_matrix_euclidean_distance(embeddings,embeddings)
        # print(cosine.shape)
        # df = pd.DataFrame(cosine.numpy())
        # df.to_csv("DataPipeline/output/dialog/visualization.csv")
        # select_least_similar_sample(df_dataset, embeddings, df_dataset['type_num'], 100, n_sample)

    def sum_by_row(embeddings, k, i):
        # embeddings=embeddings.cuda()
        cosine = compute_matrix_euclidean_distance(embeddings,embeddings)
        df = pd.DataFrame(cosine.numpy())
        df['sum'] = df.sum(axis=1)
        sorted_index = df['sum'].sort_values(ascending=False).index
        if k < 10:
            top_k_index = sorted_index
        else:
            top_k_index = sorted_index[:k]
        top_k_df = df_dataset[df_dataset['riskType']==category_labels[i]].iloc[top_k_index]
        write_path = r"DataPipeline/output/dialog/table_new/p2_E_sum.csv"
        if i==0:
            top_k_df.to_csv(write_path,index=False,encoding="utf-8",mode='a')
        else:
            top_k_df.to_csv(write_path,index=False,encoding="utf-8",mode='a',header=False)
        return top_k_df
    
    def greedy_select(embeddings, k, i):
        # print("enter")
        embeddings=embeddings.cuda()
        cosine = compute_matrix_euclidean_distance(embeddings,embeddings)
        # print(cosine.shape)
        nan_mask = torch.isnan(cosine)
        cosine[nan_mask] = 0
        selected_indices = []
        if k < 10:
            top_k_df = df_dataset[df_dataset['riskType']==category_labels[i]]
        else:
            while len(selected_indices) < k:
                max_value = torch.max(cosine)
                max_indices = torch.argmax(cosine).item()
                max_row = max_indices // cosine.shape[1]
                max_col = max_indices % cosine.shape[1]
                # print(f'{max_row}   {max_col}')
                # cosine[max_row, max_col] = 0 # 将最大值设为0 compare2
                cosine[max_row, :] = 0  # 将最大值所在的行设为0 compare1
                cosine[:, max_col] = 0  # 将最大值所在的列设为0
                if max_row in selected_indices and max_col in selected_indices:
                    continue
                if max_row not in selected_indices:
                    selected_indices.append(max_row)
                if max_col not in selected_indices:
                    selected_indices.append(max_col)
            #     print(len(selected_indices))
            # print(selected_indices)

        top_k_df = df_dataset[df_dataset['riskType']==category_labels[i]].iloc[selected_indices]
        write_path = r"DataPipeline/output/dialog/table_new/p2_corr_Cos_Ed.csv"
        if i==0:
            top_k_df.to_csv(write_path,index=False,encoding="utf-8",mode='a')
        else:
            top_k_df.to_csv(write_path,index=False,encoding="utf-8",mode='a',header=False)
        return top_k_df

        return top_k_df

    def groupby_data():
        texts = []
        for i, docs in enumerate(dataset):  # 分类
            texts = [str(x) for x in docs]
            if len(texts)==0:
                continue
            embeddings = model.encode(texts)
            embeddings = torch.tensor(embeddings).to(torch.float32)
            mean_distance = compute_mean_euclidean_distance(embeddings)
            print(category_labels[i]+": "+str(len(texts))+", "+str(mean_distance))

            
            k = int(len(texts) * 0.5)
            # df_sum_by_row = sum_by_row(embeddings, k, i)
            df_greedy_select = greedy_select(embeddings, k, i)
    
    all_data()
    groupby_data()

def main(args):
    model = FlagModel("./TextPipeline/model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示",
                  use_fp16=True)
    '''
    category_labels = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                        '虚假征信类','虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
                        '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                        '网黑案件','无风险']
    '''
    category_labels = [ '虚假购物、服务类', '虚假信用服务类', '冒充电商物流客服类',
                        '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件',
                        '冒充公检法及政府机关类', '冒充领导、熟人类', 
                        '虚假网络投资理财类', '无风险']
    # '''
    embedding_store_path = os.path.join(args.embedding_store_root_path, f"represent_embedding_{args.n_sample if args.n_sample > 0 else 'all'}_top_{args.top_k if args.top_k > 0 else 'all'}.pth")
    sample_store_path = os.path.join(args.sample_store_root_path, f"represent_sample_{args.n_sample if args.n_sample > 0 else 'all'}_top_{args.top_k if args.top_k > 0 else 'all'}.json")
    select_represent_sample(model, args.data_path, embedding_store_path, sample_store_path, category_labels, args.n_sample, args.top_k)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process embeddings and samples.')

    # 设置默认值
    parser.add_argument('--embedding_store_root_path', type=str, default='dialog/',
                        help='Path to the embedding storage file.')
    
    parser.add_argument('--data_path', type=str, default='./DataPipeline/output/dialog/table_new/p2_selected copy.csv',
                        help='Path to the data file.')
    
    parser.add_argument('--sample_store_root_path', type=str, default='dialog/',
                        help='Path to the sample storage file.')
    
    parser.add_argument('--n_sample', type=int, default=80000,
                        help='Number of samples.-1 means use all sample')
    
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top K value. -1 means use all sample')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)


