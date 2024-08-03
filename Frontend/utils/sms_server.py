
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import re
from sub_module import sms_text_module
from mark import text_mark

class InputData(BaseModel):
    text: str
    
class OutputData(BaseModel):
    text:str

if __name__ == '__main__':
    app = FastAPI()
    @app.post('/test', response_model=OutputData)
    def reply(data: InputData):
        print(data.text)
        detect_text = sms_text_module(text_mark(data.text))
        # detect_text = '1. 是否存在风险内容：否 2. 风险摘要：该文本提供了外卖取件的具体信息，包括位置和取件方式，不存在风险内容。3. 识别的风险类型：无4. 风险内容分析：无5. 防范建议：无'
        print(detect_text)
        return OutputData(text=detect_text)

    uvicorn.run(app, host="10.249.189.249", port=1931)