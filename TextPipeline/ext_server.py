import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from extraction import Ext
import uvicorn

class InputData(BaseModel):
    text: str
    
class OutputData(BaseModel):
    ext_text:str

if __name__ == '__main__':
    app = FastAPI()
    @app.post('/ext', response_model=OutputData)
    def reply(data: InputData):
        result = Ext(data.text)
        return OutputData(ext_text=result)

    uvicorn.run(app, host="127.0.0.1", port=1930)