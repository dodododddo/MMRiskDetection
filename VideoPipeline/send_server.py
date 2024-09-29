import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from video_to_image_audio import video_to_image_audio
import uvicorn
import os

class InputData(BaseModel):
    video_path: str
    
class OutputData(BaseModel):
    image: str
    audio: str

if __name__ == '__main__':
    app = FastAPI()
    @app.post('/video_extract', response_model=OutputData)
    def reply(data: InputData):
        print(data.video_path)
        
        result = video_to_image_audio(data.video_path)
        print(result.image, result.audio)
        return OutputData(image=result.image, audio=result.audio)

    uvicorn.run(app, host="127.0.0.1", port=1928)