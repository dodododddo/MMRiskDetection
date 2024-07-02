import json
import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import pipeline
from FlagEmbedding import FlagModel

def get_nearest_indices(embeddings, ref_vectors, k)->torch.Tensor:
    cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), ref_vectors, dim=2)
    _, top_k_indices = torch.topk(cos_sim, k)
    return top_k_indices

if __name__ == '__main__':
    test_path = "./dataset/test.json"
    with open(test_path, 'r') as f:
        all_data = json.load(f)
        
    model_path = './model/llama3-chat-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto"
        )
    embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)

    risk_label = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                    '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                    '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                    '网黑案件', '无风险']
        
    prompt = f'''你是一个风险判断的专家，你将接受一些诈骗案件示例作为参考，然后接受一个短信文本，请按输出格式完成以下两个任务：
                    输出格式：{{'风险类别':'...', '风险点':'...'}}
                    1. 风险类型包括{str(risk_label)}，若认为短信是无风险文本则风险类型为无风险。
                    2. 风险点可用短语概括，若可用原文中词句可直接使用
                    注意：除该字典外不要输出任何其他内容，确保你仅仅输出一个字典'''
                    
    ref_vectors = torch.load('embedding/represent_mean_top100_1200.pth')
    with open('dataset/represent_mean_top100_1200.json') as f:
        ref_docs = json.load(f)

    datas = [str(x['文本']) for x in all_data[:5]]
    pipe = pipeline(model, tokenizer)
    embeddings = embedding_model.encode(datas)
    embeddings = torch.tensor(embeddings).to(torch.float32)
    indices = get_nearest_indices(embeddings,ref_vectors, 3)

    for idx, data in enumerate(datas):
        raw_examples = [ref_docs[x] for x in list(indices[idx])]
        examples = " ".join([f'案例{i}: {example}' for i, example in enumerate(raw_examples)])
        user_text = examples + '短信文本：' + data
        text = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': user_text}]
        resp = pipe(text)
        print('\n')
        print(f'短信内容:{all_data[idx]}')
        print(f'模型回复:{resp}')
        print(f'参考案例:{examples}')