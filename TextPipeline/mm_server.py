from openai import OpenAI
import gradio as gr
from FlagEmbedding import FlagModel, FlagReranker
from utils import select_topk_with_threshold
from rag import get_nearest_indices
import torch
import json
import re
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class InputData(BaseModel):
    message: str
    
class OutputData(BaseModel):
    answer:str
    
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

bert_model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model/ernie-3.0-xbase-zh-finetune-message", num_labels=10)
tokenizer = AutoTokenizer.from_pretrained("model/ernie-3.0-xbase-zh")
label_cat = [ '冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
              '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
              '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险']
reranker = FlagReranker('./model/bge-reranker-large', use_fp16=True)
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
    
example_input_1 = "来源:文本,文本内容:我们提供最新的投资机会，只需提供您的银行账户信息，即可获得丰厚回报。不要错过这一机会，立即行动！此外，有人说政府计划削减养老金，这是一场针对老年人的巨大阴谋。"
example_answer_1 = '''
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
    
example_input_3 = '''来源: 视频, 视频内容: 
    [' The image shows a woman with dark hair and makeup, wearing a red top, speaking or looking to her right. ', 
    ' The image shows a woman with dark hair and makeup, likely at an event, with a red dress and smiling expression. ',
    ' The image shows a close-up of a woman with dark hair, looking slightly to the side, and wearing what appears to be a red outfit. '], 
    视频风险内容: [' Yes, these images contain evidence of a woman who appears to be heavily edited or altered, possibly for cosmetic or appearance purposes, 
    which could suggest face-changing techniques. '," Yes, the image appears to contain a face-changing technique as the woman's face appears to be altered in some way, possibly through digital manipulation or makeup. ", 
    ' No, the image does not contain any evidence of pornography, violence, hatred, rumors, or face-changing techniques. It is a photograph of a woman. '], 
    视频中音频内容:  and that they are going to be more the Hispanic community., 
    视频是否换脸：伪造, 声音是否合成：伪造'''

example_answer_3 = '''1. 是否存在风险内容：是         
        2. 风险摘要：该视频内容中存在涉及面部编辑或换脸的风险，可能涉及到使用面部识别技术进行虚假的信息传播。         
        3. 识别的风险类型：面部编辑/换脸         
        4. 风险内容分析：         - 面部编辑/换脸：视频中存在使用面部识别技术进行虚假信息传播的风险，可能涉及到使用换脸技术。         
        5. 防范建议：         - 面部编辑/换脸：提高对视频内容的审查，特别是面部识别技术的使用，确保信息的真实性和合法性。
'''

example_input_2 = '''
        来源: 图片, 图片上文字内容: 我刚做完规划没多久,晚上去小
        金库装赚钱饰
        小金库?
        1
        对呀,期指里面的钱就是我的小
        金库

        晚上开始?
        嗯呢,晚上,要不要尝试一下大
        宝宝
        第一次,你也不熟悉,准备个五
        千吧
        我还说2万呢
        以后机会多的是,第一次就当熟
        悉吧大宝宝,
        图片人脸伪造：未检测
'''

example_answer_2 = '''
    1. 是否存在风险内容：是        
    2. 风险摘要：该图片展示了一段聊天对话，聊天中涉及了潜在的金融投资建议（如期指），且聊天中的一方缺乏投资经验，可能存在被诈骗的风险。       
    3. 识别的风险类型：诈骗、金融投资风险     
    4. 风险内容分析：对话中提到的“期指”和大额投资金额可能暗示了高风险的金融操作。聊天中提到的投资金额（如5000和20000）和对“第一次”的提醒，可能表明对方不熟悉投资情况，增加了被误导或诈骗的风险。
    5. 防范建议：核实信息： 确保所有投资建议来自信誉良好的金融机构或专业顾问。
                警惕诱惑： 对涉及高收益承诺或不明金融产品的建议保持警惕。
                学习投资知识： 了解基本的金融投资知识，避免被不熟悉的投资产品所误导。
                咨询专业意见： 在做出任何投资决策前，咨询专业金融顾问以确保决策的安全性。
'''

# def reply(message):
#     answer = ""
#     response = client.chat.completions.create(
#         model="llama3",
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": example_input},
#             {"role": "assistant", "content": example_answer},
#             {"role": "user", "content": message},
#         ],
#         stream=True,
#         temperature=0
#     )
    
#     next(response)
#     for resp in response:
#         answer += resp.choices[0].delta.content
#         yield resp.choices[0].delta.content
        
#     if answer.find('诈骗') != -1:
#         embedding = embedding_model.encode(message)
#         embedding = torch.tensor(embedding).to(torch.float32)
#         indices = get_nearest_indices(embedding, ref_vectors, 10)
#         examples = [ref_docs[x]["案情描述"] for x in list(indices)]
#         scores = reranker.compute_score([[message, example] for example in examples], normalize=True)
#         examples = select_topk_with_threshold(examples, scores, 0.3, 5)
#         if len(examples) > 0:
#             yield "\n\n相似案例: "
#             for i, example in enumerate(examples):
#                 yield f"\n\n案例 {i+1}: {example}" 

def reply(message):
    answer = ""
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": example_input_1},
            {"role": "assistant", "content": example_answer_1},
            {"role": "user", "content": example_input_2},
            {"role": "assistant", "content": example_answer_2},
            {"role": "user", "content": example_input_3},
            {"role": "assistant", "content": example_answer_3},
            {"role": "user", "content": message},
        ],
        stream=True,
        temperature=0,
        top_p=1.0
    )
    
    next(response)
    for resp in response:
        answer += resp.choices[0].delta.content
        print(resp.choices[0].delta.content)
        yield resp.choices[0].delta.content
    
    if answer.find('识别的风险类型：诈骗') != -1:
        data = tokenizer(message, return_tensors="pt")
        cls = label_cat[bert_model(**data).logits.argmax().item()]
        yield  f'\n\n6.细分诈骗类型：{cls}'
        
        embedding = embedding_model.encode(message)
        embedding = torch.tensor(embedding).to(torch.float32)
        indices = get_nearest_indices(embedding, ref_vectors, 10)
        examples = [ref_docs[x]["案情描述"] for x in list(indices)[:3]]
        # scores = reranker.compute_score([[message, example] for example in examples], normalize=True)
        # examples = select_topk_with_threshold(examples, scores, 0.2, 3)
        if len(examples) > 0:
            yield "\n\n7.相似案例："
            for i, example in enumerate(examples):
                yield f"\n\n案例 {i+1}: {example}" 


app = FastAPI()
@app.post("/generate", response_model=OutputData)
async def chat(request: InputData):
    '''encode: utf-8'''
    response_stream = reply(request.message)
    return StreamingResponse(response_stream, media_type="text/plain")


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=1111)
    

    