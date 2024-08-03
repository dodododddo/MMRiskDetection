from openai import OpenAI
import gradio as gr
from FlagEmbedding import FlagModel, FlagReranker
from rag import get_nearest_indices
import torch
import json
from utils import select_topk_with_threshold
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True)
ref_vectors = torch.load('embedding/represent_mean_top100_1200.pt')
with open('dataset/represent_mean_top100_1200.json') as f:
    ref_docs = json.load(f)


reranker = FlagReranker('./model/bge-reranker-large', use_fp16=True)

bert_model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model/ernie-3.0-xbase-zh-finetune-message", num_labels=10)
tokenizer = AutoTokenizer.from_pretrained("model/ernie-3.0-xbase-zh")
label_cat = [ '冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
              '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
              '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险']

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
    
example_input = "来源:文本,文本内容:我们提供最新的投资机会，只需提供您的银行账户信息，即可获得丰厚回报。不要错过这一机会，立即行动！此外，有人说政府计划削减养老金，这是一场针对老年人的巨大阴谋。"
example_answer = '''
         1. 是否存在风险内容：是
         2. 风险摘要：该文本通过提供虚假的投资机会，诱惑受害者提供个人信息；还散播了政府削减养老金的不实信息,存在诈骗和谣言的风险，存在涉及财务信息的不合理要求域关于政府计划削减养老金的虚假信息。该文本主要是针对老年人和投资爱好者群体的风险信息。
         3. 识别的风险类型：诈骗, 谣言
         4. 风险内容分析：
         - 诈骗：文本中包含涉及财务信息的不合理要求，试图诱导用户提供个人信息或进行转账。
         - 谣言：文本中传播了关于政府计划削减养老金的虚假信息，可能引发恐慌和误解。
         5. 防范建议：
         - 诈骗：不随意透露个人信息，遇到财务请求时要核实对方身份，保持警惕。
         - 谣言：查证信息来源的可靠性，不轻信未经证实的消息，向官方渠道求证。
    '''

def reply_for_gradio(message, history):
    answer = ""
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": example_input},
            {"role": "assistant", "content": example_answer},
            {"role": "user", "content": message},
        ],
        stream=True,
        max_tokens=512
    )
    
    next(response)
    for resp in response:
        answer += resp.choices[0].delta.content
        yield answer
        
    if answer.find('识别的风险类型：诈骗') != -1:
        data = tokenizer(message, return_tensors="pt")
        cls = label_cat[bert_model(**data).logits.argmax().item()]
        answer +=  f'\n\n细分诈骗类型: {cls}'
        yield answer
            
        embedding = embedding_model.encode(message)
        embedding = torch.tensor(embedding).to(torch.float32)
        indices = get_nearest_indices(embedding, ref_vectors, 10)
        examples = [ref_docs[x]["案情描述"] for x in list(indices)]
        scores = reranker.compute_score([[message, example] for example in examples], normalize=True)
        examples = select_topk_with_threshold(examples, scores, 0.3, 5)
        if len(examples) > 0:
            answer += "\n\n相似案例: "
            yield answer
            for i, example in enumerate(examples):
                answer += f"\n\n案例 {i+1}: {example}" 
                yield answer



if __name__ == '__main__':
    demo = gr.ChatInterface(reply_for_gradio)
    demo.launch()
    

    