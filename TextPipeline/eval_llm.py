import json
import ast
import torch
import argparse
from FlagEmbedding import FlagModel
from tqdm import tqdm
from rag import LLMWrapper, RAGPipeline, VllmConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftConfig, PeftModel
    
def eval(model_path, test_dataset_path, use_rag, sample_num=100, max_regenerate_num = 5, batch_size=16):
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
    risk_label = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                  '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                  '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                  '网黑案件', '无风险']
    
    prompt = f'''你是一个风险判断的专家，你将接受一个短信文本，请给出短信文本的风险类别和风险点：
                 输出格式：{{'风险类别':'...', '风险点':'...'}}
                 注意点：
                 1. 风险类型包括{str(risk_label)}，若认为短信是无风险文本则风险类型为无风险。
                 2. 风险点可用短语概括，若可用原文中词句可直接使用
                 3. 除该字典外不要输出任何其他内容，确保你仅仅输出一个字典
              '''
               
    with open(test_dataset_path, 'r') as f:
        all_data = json.load(f)[:sample_num]
    
    # Load model directly
    if use_rag:
        embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True)
        ref_vectors = torch.load('embedding/represent_mean_top100_1200.pt')
        with open('dataset/represent_mean_top100_1200.json') as f:
            ref_docs = json.load(f)
        llm = LLMWrapper(model_path, use_vllm=False)
        pipe = RAGPipeline(llm, embedding_model, None, ref_vectors, ref_docs)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        llm = LLMWrapper(model, tokenizer)
        # llm = LLMWrapper(model_path, use_vllm=True, vllm_config=VllmConfig(2, 0.2))
        pipe = RAGPipeline(llm)
        
    tp = 0
    fp = 0
    fn = 0
    tp_cls = 0
    fp_cls = 0
    parse_error = 0
    
    progress_bar = tqdm(total=len(all_data))
    generate_num = max_regenerate_num * len(all_data) / batch_size
    
    while all_data and max_regenerate_num > 0:
        batch_data = all_data[:batch_size]
        all_data = all_data[batch_size:]
        
        batch_text = [data['文本'] for data in batch_data]
        raw_resps = pipe(batch_text, prompt, use_rag=use_rag)
        generate_num -= 1
        for idx, raw_resp in enumerate(raw_resps):
            try:
                resp = ast.literal_eval(raw_resp.answer)
                pred = resp['风险类别']
                target = batch_data[idx]['风险类别']
                progress_bar.update(1)
                if(target == '无风险' and pred == target):
                    tp_cls += 1
                    
                if(target != '无风险' and pred == target):
                    tp += 1
                    tp_cls += 1
                
                if(target == '无风险' and pred != target):
                    fp_cls += 1
                    fp += 1
                    
                if(target != '无风险' and pred != target):
                    if(pred == '无风险'):
                        fn += 1
                        fp_cls += 1
                    else:
                        fp_cls += 1
            except(SyntaxError, KeyError, TypeError, ValueError):
                print(f'解析失败：{raw_resp.answer}')
                all_data.append(batch_data[idx])
                parse_error += 1
                
    progress_bar.close()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    precision_cls = tp_cls / (tp_cls + fp_cls)
    
    print(f'precision:{precision:.3f}, recall:{recall:.3f}, f1:{f1:.3f}, precision_cls:{precision_cls:.3f}, parse_error:{parse_error}') 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument parser for evaluation.')

    # 添加命令行参数
    parser.add_argument('--model_path', type=str, default='model/llama3-chat-chinese', help='Evaluate.')
    parser.add_argument('--test_dataset_path', type=str, default='dataset/test.json', help='Evaluate.')
    parser.add_argument('--zero_shot', action='store_true', help='Evaluate.')
    parser.add_argument('--rag', action='store_true', help='Evaluate using RAG.')
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    eval(args.model_path, args.test_dataset_path, args.rag, args.sample_num, batch_size=args.batch_size)
    