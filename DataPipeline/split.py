import json

with open('dataset/ccl_2023_eval_6_train.json', 'r') as f:
    data = json.load(f)
    
with open('dataset/ccl_2023_eval_6_train_rag_or_finetuning_split.json', 'w') as f:
    json.dump(data[-40000:], f,indent=4, ensure_ascii=False)
    
with open('dataset/ccl_2023_eval_6_train_trans_and_eval_split.json', 'w') as f:
    json.dump(data[:-40000], f,indent=4, ensure_ascii=False)
    
