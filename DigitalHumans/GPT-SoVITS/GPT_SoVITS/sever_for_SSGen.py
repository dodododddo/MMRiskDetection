import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from convenience import GPTSoVITS
import uvicorn

class InputData(BaseModel):
    ref_wav_path: str
    prompt_text: str
    text: str

class OutputData(BaseModel):
    output_sound_path: str

if __name__ == '__main__':
    app = FastAPI()
    @app.post('/SSGen', response_model=OutputData)
    def reply(data: InputData):
        result = GPTSoVITS.run_tts(
            ref_wav_path=data.ref_wav_path,
            prompt_text=data.prompt_text,
            text=data.text,
            prompt_language="中文",
        )
        return OutputData(output_sound_path=result)
    
    uvicorn.run(app, host="127.0.0.1", port=9995)