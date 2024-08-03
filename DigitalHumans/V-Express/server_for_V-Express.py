import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from utils import VideoGenerator
import uvicorn

class InputData(BaseModel):
    Image_path: str
    Audio_path: str
    function_A: bool
    function_B: bool
    function_C: bool

class OutputData(BaseModel):
    Video_path: str

if __name__ == '__main__':
    app = FastAPI()
    @app.post('/Digital_video', response_model=OutputData)
    def reply(data: InputData):
        if data.function_A == True:
            result = VideoGenerator.run_video_generation(
                retarget_strategy='no_retarget',
                reference_image_path = data.Image_path,
                audio_path = data.Audio_path,
                kps_path='./test_samples/short_case/AOC/kps.pth',
                output_path='../../DataBuffer/DigitalBuffer/Video_gen.mp4'
            )
        elif data.function_B == True:
            result = VideoGenerator.run_video_generation(
                retarget_strategy='fix_face',
                reference_image_path = data.Image_path,
                audio_path = data.Audio_path,
                output_path='../../DataBuffer/DigitalBuffer/Video_gen.mp4'
            )
        else:
            result = VideoGenerator.run_video_generation(
                retarget_strategy='offset_retarget',
                reference_image_path = data.Image_path,
                audio_path = data.Audio_path,
                kps_path='./test_samples/short_case/tys/kps.pth',
                output_path='../../DataBuffer/DigitalBuffer/Video_gen.mp4'
            )
        return OutputData(Video_path=result)
    
    uvicorn.run(app, host="127.0.0.1", port=9996)