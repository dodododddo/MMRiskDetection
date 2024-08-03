import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from utils import batch_iterator

    
model = AutoModel.from_pretrained(
    "/data1/home/jrchen/MMRiskDetection/DataPipeline/reward_model/internlm2-7b-reward", 
    device_map="cuda", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("/data1/home/jrchen/MMRiskDetection/DataPipeline/reward_model/internlm2-7b-reward", trust_remote_code=True)

with open('DataPipeline/output/message/useful/train_eval.json') as f:
    dataset = json.load(f)
    dataset = [x['messages'] for x in dataset]

all_scores = []
for batch in batch_iterator(dataset, 16):
    scores = model.get_scores(tokenizer, batch)
    all_scores += scores

print(scores)
print(all_scores)