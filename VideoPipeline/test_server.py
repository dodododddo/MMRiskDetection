
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

class InputData(BaseModel):
    text: str
    
class OutputData(BaseModel):
    text:str

if __name__ == '__main__':
    app = FastAPI()
    @app.post('/test', response_model=OutputData)
    def reply(data: InputData):
        print(data.text)
        return OutputData(text='连接成功')

    uvicorn.run(app, host="10.249.189.249", port=1931)
