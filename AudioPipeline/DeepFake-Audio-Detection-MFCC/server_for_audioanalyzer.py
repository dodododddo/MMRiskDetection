import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from utils.AudioAnalyzer import AudioAnalyzer
import uvicorn

class InputData(BaseModel):
    audio_for_analyse_path: str
    
class OutputData(BaseModel):
    fake_or_not: bool

if __name__ == '__main__':
    audioDetection = AudioAnalyzer()
    app = FastAPI()
    @app.post('/audio_detection', response_model=OutputData)
    def reply(data: InputData):
        print(data.audio_for_analyse_path)
        result = audioDetection.analyze_audio(data.audio_for_analyse_path)
        print(result)
        return OutputData(fake_or_not=result)
    
    uvicorn.run(app, host="127.0.0.1", port=9998)
    