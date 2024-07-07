from transformers import AutoTokenizer
from FlagEmbedding import FlagModel
from tqdm import tqdm
import json
import torch
import argparse
import os

def get_represent_indices(embeddings, k):
    mean_embedding = torch.mean(embeddings, dim=0)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings, mean_embedding.unsqueeze(0), dim=1)
    _, top_k_indices = torch.topk(cos_sim, k)
    return top_k_indices


def select_represent_sample(model, data_path, embedding_store_path, sample_store_path, category_labels, n_sample, top_k):
    with open(data_path, 'r') as f:
        raw_dataset = json.load(f)
        if n_sample > 0:
            raw_dataset = raw_dataset[:n_sample]


    dataset = [[] for i in range(len(category_labels))]
    for data in raw_dataset:
        index = category_labels.index(data['案件类别'])
        if index != -1:
            data.pop('案件编号')
            dataset[index].append(data)

    represent_samples = []
    represent_embeddings = []

    for docs in tqdm(dataset):
        texts = [str(x) for x in docs]
        embeddings = model.encode(texts)
        embeddings = torch.tensor(embeddings).to(torch.float32)
        if top_k > 0:
            represent_indices = get_represent_indices(embeddings, top_k)
            represent_embeddings.append(embeddings[represent_indices])
            represent_samples += [docs[i] for i in list(represent_indices)]
        else:
            represent_embeddings.append(embeddings)
            represent_samples += docs
        
        
    represent_embeddings = torch.cat(represent_embeddings, dim=0)
    print(represent_embeddings.shape)
    torch.save(represent_embeddings, embedding_store_path)
        
    with open(sample_store_path, 'w') as f:
        json.dump(represent_samples, f, indent=4, ensure_ascii=False)

def main(args):
    model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)

    category_labels = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                        '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                        '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                        '网黑案件']
    
    embedding_store_path = os.path.join(args.embedding_store_root_path, f"represent_embedding_{args.n_sample if args.n_sample > 0 else 'all'}_top_{args.top_k if args.top_k > 0 else 'all'}.pth")
    sample_store_path =  os.path.join(args.sample_store_root_path, f"represent_sample_{args.n_sample if args.n_sample > 0 else 'all'}_top_{args.top_k if args.top_k > 0 else 'all'}.json")
    select_represent_sample(model, args.data_path, embedding_store_path, sample_store_path, category_labels, args.n_sample, args.top_k)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process embeddings and samples.')

    # 设置默认值
    parser.add_argument('--embedding_store_root_path', type=str, default='embedding/',
                        help='Path to the embedding storage file.')
    
    parser.add_argument('--data_path', type=str, default='../DataPipeline/dataset/ccl_2023_eval_6_train_rag_or_finetuning_split.json',
                        help='Path to the data file.')
    
    parser.add_argument('--sample_store_root_path', type=str, default='dataset/',
                        help='Path to the sample storage file.')
    
    parser.add_argument('--n_sample', type=int, default=20000,
                        help='Number of samples.-1 means use all sample')
    
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top K value. -1 means use all sample')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)


