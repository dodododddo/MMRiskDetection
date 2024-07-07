import json
import ast
import torch
import argparse
from FlagEmbedding import FlagModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import pipeline
from rag import get_nearest_indices, LLM, RAGPipeline, RAGResp

    
def eval_zero_shot(model_path, sample_num=100, max_regenerate_num = 5, batch_size=2):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
    # Load model directly
    llm = LLM(model, tokenizer)

    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
    risk_label = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                  '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                  '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                  '网黑案件', '无风险']
    
    prompt = f'''你是一个风险判断的专家，你将接受一个短信文本，请按输出格式完成以下两个任务：
                 输出格式：{{'风险类别':'...', '风险点':'...'}}
                 1. 风险类型包括{str(risk_label)}，若认为短信是无风险文本则风险类型为无风险。
                 2. 风险点可用短语概括，若可用原文中词句可直接使用
                 注意：除该字典外不要输出任何其他内容，确保你仅仅输出一个字典'''
               
    test_path = "./dataset/test.json"
    with open(test_path, 'r') as f:
        all_data = json.load(f)[:sample_num]
    
    all_data = [all_data[i:i + batch_size] for i in range(0, len(all_data), batch_size)]
    
    tp = 0
    fp = 0
    fn = 0
    tp_cls = 0
    fp_cls = 0
    parse_error = 0
    
    for idx, batch_data in enumerate(tqdm(all_data)):
        batch_text = [data['文本'] for data in batch_data]
        for _ in range(max_regenerate_num):
            try:
                raw_resps = llm(batch_text, prompt)
                resps = [ast.literal_eval(raw_resp) for raw_resp in raw_resps]
                preds = [resp['风险类别'] for resp in resps]
                targets = [data['风险类别'] for data in batch_data]
                break
            except(SyntaxError, KeyError, TypeError, ValueError):
                pass
        else:
            print(f'\n {idx + 1}: parse error happened in this batch: ', raw_resps)
            parse_error += 1
            continue
            
            
        for target_cls, pred_cls in zip(preds, targets):
            if(target_cls == '无风险' and pred_cls == target_cls):
                tp_cls += 1
                
            if(target_cls != '无风险' and pred_cls == target_cls):
                tp += 1
                tp_cls += 1
            
            if(target_cls == '无风险' and pred_cls != target_cls):
                fp_cls += 1
                fp += 1
                
            if(target_cls != '无风险' and pred_cls != target_cls):
                if(pred_cls == '无风险'):
                    fn += 1
                    fp_cls += 1
                else:
                    fp_cls += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    precision_cls = tp_cls / (tp_cls + fp_cls)
    
    print(f'precision:{precision}, recall:{recall}, f1:{f1}, precision_cls:{precision_cls}, parse_error:{parse_error}') 


def eval_rag(model_path, sample_num=100, max_regenerate_num = 5, batch_size = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
    # Load model directly
    
    embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True)

    risk_label = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                  '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                  '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                  '网黑案件', '无风险']
    
            
    # ref_vectors = torch.load('embedding/represent_embedding_all_top_all.pth')
    # with open('dataset/represent_sample_all_top_all.json') as f:
    #     ref_docs = json.load(f)
    ref_vectors = torch.load('embedding/represent_mean_top100_1200.pth')
    with open('dataset/represent_mean_top100_1200.json') as f:
        ref_docs = json.load(f)
    
    prompt = f'''你是一个风险判断的专家，你将接受一些诈骗案件示例作为参考，然后接受一个短信文本，请按输出格式完成以下两个任务：
                    输出格式：{{'风险类别':'...', '风险点':'...'}}
                    1. 风险类型包括{str(risk_label)}，若认为短信是无风险文本则风险类型为无风险。
                    2. 风险点可用短语概括，若可用原文中词句可直接使用
                    注意：除该字典外不要输出任何其他内容，确保你仅仅输出一个字典'''
    
    llm = LLM(model, tokenizer)
    rag_pipeline = RAGPipeline(llm, embedding_model, None, ref_vectors, ref_docs)
    test_path = "./dataset/test.json"
    with open(test_path, 'r') as f:
        all_data = json.load(f)[: sample_num]
        
        
    tp = 0
    fp = 0
    fn = 0
    tp_cls = 0
    fp_cls = 0
    parse_error = 0
    
    all_data = [all_data[i:i+batch_size] for i in range(len(all_data))]
    
    for idx, batch_data in enumerate(tqdm(all_data)):
        batch_text = [data['文本'] for data in batch_data]
        for _ in range(max_regenerate_num):
            try:
                raw_resps = rag_pipeline(batch_text, prompt)
                resps = [ast.literal_eval(raw_resp.answer) for raw_resp in raw_resps]
                preds = [resp['风险类别'] for resp in resps]
                targets = [data['风险类别'] for data in batch_data]
                break
            except(SyntaxError, KeyError, TypeError, ValueError):
                pass
        else:
            print(f'\n {idx + 1}: parse error happened in this batch: ', raw_resps)
            parse_error += 1
            continue
        
        for target_cls, pred_cls in zip(preds, targets):
            if(target_cls == '无风险' and pred_cls == target_cls):
                tp_cls += 1
                
            if(target_cls != '无风险' and pred_cls == target_cls):
                tp += 1
                tp_cls += 1
            
            if(target_cls == '无风险' and pred_cls != target_cls):
                print(f'第{idx+1}短信判断错误，pred: {pred_cls}, except: {target_cls}')
                fp_cls += 1
                fp += 1
                
            if(target_cls != '无风险' and pred_cls != target_cls):
                print(f'第{idx+1}条短信判断错误，pred: {pred_cls}, except: {target_cls}')
                if(pred_cls == '无风险'):
                    fn += 1
                    fp_cls += 1
                else:
                    fp_cls += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    precision_cls = tp_cls / (tp_cls + fp_cls)
    
    print(f'precision:{precision:.3f}, recall:{recall:.3f}, f1:{f1:.3f}, precision_cls:{precision_cls:.3f}, parse_error:{parse_error}')
            
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument parser for evaluation.')

    # 添加命令行参数
    parser.add_argument('--model_path', type=str, default='model/llama3-chat-chinese', help='Evaluate using zero shot.')
    parser.add_argument('--zero_shot', action='store_true', help='Evaluate using zero shot.')
    parser.add_argument('--rag', action='store_true', help='Evaluate using RAG.')
    parser.add_argument('--sample_num', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.rag:
        eval_rag(args.model_path, args.sample_num)
    else:
        eval_zero_shot(args.model_path, args.sample_num)
    