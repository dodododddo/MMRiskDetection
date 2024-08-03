import numpy as np
from tqdm import tqdm

def pipeline(model, tokenizer):
    def process(input_text, max_new_token=8192, temperature=0.6):

        # messages = [
        #     {"role": "user", "content": input_text},
        # ]

        input_ids = tokenizer.apply_chat_template(
            input_text, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_token,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)
    return process

def select_topk_with_threshold(examples, scores, threshold, k):
    scores = np.array(scores)
    
    # 找出大于阈值的值及其对应的索引
    indices_above_threshold = np.where(scores > threshold)[0]
    values_above_threshold = scores[indices_above_threshold]
    
    topk_indices_sorted = np.argsort(values_above_threshold)[-k:][::-1]
    
    # 取得对应的原始索引
    topk_indices = indices_above_threshold[topk_indices_sorted]
    
    # 使用这些索引在A中选择对应的元素
    selected_examples = [examples[i] for i in topk_indices]
    
    return selected_examples

def batch_iterator(dataset, batch_size):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset[i:i + batch_size]

def process_string(s):
    # 如果字符串被单引号包围，则去掉单引号
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
        
    if s.startswith("\"") and s.endswith("\""):
        s = s[1:-1]
        
    # 如果字符串中存在逗号，则只保留第一个逗号前的内容
    if ',' in s:
        s = s.split(',')[0]
    
    return s
