from rag import RAGPipeline, LLMWrapper
from FlagEmbedding import FlagModel
import torch
import json
import time
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class InputData(BaseModel):
    message: str
    
class OutputData(BaseModel):
    answer:str
    ref_examples: str

if __name__ == '__main__':
    embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)
    ref_vectors = torch.load('embedding/represent_mean_top100_1200.pt')
    with open('dataset/represent_mean_top100_1200.json') as f:
        ref_docs = json.load(f)

    llm = LLMWrapper('model/llama3-chat-chinese')
    pipe = RAGPipeline(llm, embedding_model, None, ref_vectors, ref_docs)


    prompt = '''
        任务：判断输入文本是否包含风险内容（如诈骗、色情、暴力、煽动、谣言等），并对风险内容进行识别、分析，并提供防范建议。

        输入文本：
        "我们提供最新的投资机会，只需提供您的银行账户信息，即可获得丰厚回报。不要错过这一机会，立即行动！此外，有人说政府计划削减养老金，这是一场针对老年人的巨大阴谋。"

        请按照以下格式输出结果：
        1. 是否存在风险内容：是/否
        2. 识别的风险类型：{风险类型1, 风险类型2, ...}
        3. 风险内容分析：
        - {风险类型1}：{具体分析}
        - {风险类型2}：{具体分析}
        ...
        4. 防范建议：
        - {风险类型1}：{具体建议}
        - {风险类型2}：{具体建议}
        ...

        风险类型包括但不限于以下几种：
        1. 诈骗：涉及财务信息的不合理要求，试图诱导用户提供个人信息或进行转账。
        2. 色情：包含露骨的性内容或诱导性的语言。
        3. 暴力：鼓励或描述暴力行为，威胁他人安全。
        4. 煽动：使用煽动性语言，试图引发群体对特定事件的不满和抗议。
        5. 谣言：传播虚假或误导性的信息，可能引发恐慌或误解。

        示例输出：
        1. 是否存在风险内容：是
        2. 识别的风险类型：诈骗, 谣言
        3. 风险内容分析：
        - 诈骗：文本中包含涉及财务信息的不合理要求，试图诱导用户提供个人信息或进行转账。
        - 谣言：文本中传播了关于政府计划削减养老金的虚假信息，可能引发恐慌和误解。
        4. 防范建议：
        - 诈骗：不随意透露个人信息，遇到财务请求时要核实对方身份，保持警惕。
        - 谣言：查证信息来源的可靠性，不轻信未经证实的消息，向官方渠道求证。

        接下来，你将接受一些相关案例和一个短信文本，请参考案例对短信文本进行分析:

    '''

    app = FastAPI()
    @app.post('/generate', response_model=OutputData)
    def reply(data: InputData):
        print(data.message)
        resp = pipe(data.message, prompt, use_rag=True)
        return OutputData(answer=resp.answer, ref_examples=resp.ref_examples)

    uvicorn.run(app, host="127.0.0.1", port=1111)