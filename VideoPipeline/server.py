import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from video_pipeline import VideoPipeline
import uvicorn
import os

class InputVideoData(BaseModel):
    video_path: str
    
class OutputVideoData(BaseModel):
    describe:list
    risk:list
    deepfake_detection:bool

class InputImageData(BaseModel):
    image_path: str
    prompt: str
    
class OutputImageData(BaseModel):
    text: str

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    videoPipeline = VideoPipeline()
    app = FastAPI()
    @app.post('/video', response_model=OutputVideoData)
    def reply(data: InputVideoData):
        print(data.video_path)
        result = videoPipeline(data.video_path)
        return OutputVideoData(describe=result.describe, risk=result.risk, deepfake_detection=result.deepfake_detection)
    
    @app.post('/image', response_model=OutputImageData)
    def reply(data: InputImageData):
        print(data.image_path, data.prompt)
        result = videoPipeline.image_to_text(data.image_path,data.prompt)
        return OutputImageData(text = result)
    
    uvicorn.run(app, host="127.0.0.1", port=1927)
