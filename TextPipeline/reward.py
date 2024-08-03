import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from utils import batch_iterator
import torch


model = AutoModel.from_pretrained(
    "/data1/home/jrchen/MMRiskDetection/DataPipeline/reward_model/internlm2-7b-reward", 
    device_map="cuda", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("/data1/home/jrchen/MMRiskDetection/DataPipeline/reward_model/internlm2-7b-reward", trust_remote_code=True)

with open('summary/message_4000_lora.json') as f:
    answers = json.load(f)[:4000]

with open('dataset/message/sft_eval.json') as g:
    questions = json.load(g)[:4000]
    messages = [x['messages'][1]['content'] for x in questions]
    
prompt = '''
        你是一个风险内容识别专家，你将接受不同来源、从不同模态转换而来的文本，这些文本包含了原模态内容的主要信息，请判断输入文本是否包含风险内容（如诈骗、色情、暴力、煽动、谣言等），并对风险内容进行识别、分析，最后提供防范建议。

        请按照以下格式输出结果：
        1. 是否存在风险内容：是/否
        2. 风险摘要：
        3. 识别的风险类型：{风险类型1, 风险类型2, ...}
        4. 风险内容分析：
        - {风险类型1}：{具体分析}
        - {风险类型2}：{具体分析}
        ...
        5. 防范建议：
        - {风险类型1}：{具体建议}
        - {风险类型2}：{具体建议}
        ...

        风险类型包括但不限于以下几种：
        1. 诈骗：涉及财务信息的不合理要求，试图诱导用户提供个人信息或进行转账。
        2. 色情：包含露骨的性内容或诱导性的语言。
        3. 暴力：鼓励或描述暴力行为，威胁他人安全。
        4. 煽动：使用煽动性语言，试图引发群体对特定事件的不满和抗议。
        5. 谣言：传播虚假或误导性的信息，可能引发恐慌或误解。
    '''

dataset = [[{'role':'system', 'content': prompt}, {'role':'user', 'content': message}, {'role': 'assistant', 'content': answer}] for message, answer in zip(messages, answers)]
all_scores = []
for batch in batch_iterator(dataset, 32):
    scores = model.get_scores(tokenizer, batch)
    all_scores += scores
    
print(torch.mean(torch.tensor(all_scores)).item())

