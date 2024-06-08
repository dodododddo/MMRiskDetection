import json
import ast
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import pipeline

    
def eval_zero_shot(sample_num=100, max_regenerate_num = 5):
    model_path = './model/llama3-chat-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
    risk_label = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                  '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                  '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                  '网黑案件', '无风险']
    
    prompt = f'''你是一个风险判断的专家，你将接受一个短信文本，请按输出格式完成以下两个任务：
                 输出格式：{{'风险类别':'...', '风险点':'...'}}
                 1. 风险类型包括{str(risk_label)}，若认为短信是无风险文本则风险类型为无风险。
                 2. 风险点可用短语概括，若可用原文中词句可直接使用
                 注意：除该字典外不要输出任何其他内容，确保你仅仅输出一个字典'''
               
    pipe = pipeline(model, tokenizer)
    test_path = "./dataset/test.json"
    with open(test_path, 'r') as f:
        all_data = json.load(f)[:sample_num]
        
    tp = 0
    fp = 0
    fn = 0
    tp_cls = 0
    fp_cls = 0
    parse_error = 0
    
    for idx, data in enumerate(tqdm(all_data)):
        text = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': data}]
        resp = pipe(text)
        
        for _ in range(max_regenerate_num):
            try:
                resp = ast.literal_eval(resp)
                assert(isinstance(resp, dict))
                pred_cls = resp['风险类别']
                target_cls = data['风险类别']
                break
            except(SyntaxError, KeyError, TypeError):
                pass
        else:
            print(f'\n {idx + 1}: parse error ' + resp)
            parse_error += 1
        
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
                
            

if __name__ == '__main__':
    
    eval_zero_shot(sample_num=200)