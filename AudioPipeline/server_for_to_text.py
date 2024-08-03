import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from utils import To_Text
import uvicorn

class InputData(BaseModel):
    audio_path: str
    
class OutputData(BaseModel):
    text: str

if __name__ == '__main__':
    audioePipeline_to_text = To_Text()
    app = FastAPI()
    @app.post('/audio', response_model=OutputData)
    def reply(data: InputData):
        print(data.audio_path)
        result = audioePipeline_to_text.process(data.audio_path)
        return OutputData(text=result)

    uvicorn.run(app, host="127.0.0.1", port=9999)

# from fastapi import FastAPI
# from pydantic import BaseModel
# from utils import To_Text  
# import uvicorn

# class InputData(BaseModel):
#     audio_path: str
    
# class OutputData(BaseModel):
#     text: str

# app = FastAPI()

# audioPipeline_to_text = To_Text()

# @app.post('/audio', response_model=OutputData)
# def reply(data: InputData):
#     print(data.audio_path)
#     result = audioPipeline_to_text.process(data.audio_path)
#     return OutputData(text=result)

# if __name__ == '__main__':
#     uvicorn.run(app, host="127.0.0.1", port=9999)
