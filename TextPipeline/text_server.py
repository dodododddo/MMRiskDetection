from rag import RAGPipeline, LLMWrapper, VllmConfig
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
    
    llm = LLMWrapper('model/llama3-chat-chinese')
    pipe = RAGPipeline(llm)

    prompt = '''
        任务：判断输入文本是否包含风险内容（如诈骗、色情、暴力、煽动、谣言等），并对风险内容进行识别、分析，并提供防范建议。

         请按照以下格式输出结果（请留意风险摘要中的部分内容需要你进行填空）：
         1. 是否存在风险内容：是/否
         2. 风险摘要：该文本<content>,存在<risk_name>的风险，存在<risk_content>。该文本主要是针对<risk_people>的风险信息。
         3. 识别的风险类型：{风险类型1}, {风险类型2}, ...
         4.（如果识别到的风险类型中有“诈骗”则加上此条，没有则忽略）诈骗风险类别：{诈骗风险类别}
         5. 风险内容分析：
         - {风险类型1}：{具体分析}
         - {风险类型2}：{具体分析}
         ...
         6. 防范建议：
         - {风险类型1}：{具体建议}
         - {风险类型2}：{具体建议}
         ...

         其中诈骗风险类别包括{'刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                  '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                  '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                  '网黑案件'}
         诈骗风险类别只能从以上十三种中选择，不要自己编造。但是风险类型和风险类别并不是一个东西，以上的诈骗风险类别仅用于规范第4点。

         在给出的风险摘要格式中，存在<content>,<risk_name>与<risk_content>与<risk_people>的符号，此处需要在正式输出时进行填空。
         请注意，任何一个符号中都不能用空内容填充，一定要保证格式的完整性。
         <content>中请使用"行为+目的"的格式对文本进行概括。
         关于"行为"，可以提供投资机会、冒充客服、冒充熟人、赠送礼物、提供高额利益、提供虚假投资机会等。
         关于"目的"，可以是诱导转账、窃取个人信息、扫描未知二维码、点击未知链接、加入陌生群聊、诈骗财物等。
         <risk_name>中请填写风险类型，如果有多个风险类型请用“和”字连接；<risk_content>部分请结合文本填写一些具体风险内容，如果能用原文则使用原文。<risk_people>为对文本进行分析后得出的潜在风险受骗群体，如果潜在风险受骗群体为某网站或APP用户，请指明具体是什么用户；如果文本明显是两个人的对话内容，比如出现"我是你同学"之类的词汇，则受骗群体为"熟人"。


         示例输出：
         1. 是否存在风险内容：是
         2. 风险摘要：xxx
         3. 识别的风险类型：诈骗, 谣言
         4. 诈骗风险类别：xxx
         4. 风险内容分析：
         - 诈骗：xxx
         - 谣言：xxx
         5. 防范建议：
         - 诈骗：xxx
         - 谣言：xxx

         接下来，你将接受一些相关案例和一个短信文本，请参考案例对短信文本进行分析:
    '''

    app = FastAPI()
    @app.post('/generate', response_model=OutputData)
    def reply(data: InputData):
        resp = pipe(data.message, prompt, use_rag=True)
        return OutputData(answer=resp.answer, ref_examples=resp.ref_examples)

    uvicorn.run(app, host="127.0.0.1", port=1111)