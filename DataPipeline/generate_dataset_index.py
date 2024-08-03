import json

src_path = "DataPipeline/dataset/ccl_2023_eval_6_train_rag_or_finetuning_split.json"
dst_path = "DataPipeline/dataset/train_rag_or_finetuning_split_index.json"


with open(src_path, 'r') as f:
    source = json.load(f)
data = []
n = 0
for i in source:
    i['序号'] = n
    n = n+1
    # print(i)
    data.append(i)

with open(dst_path, 'w') as g:
    json.dump(data, g, indent=4, ensure_ascii=False)