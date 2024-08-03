import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from utils import FaceRestorer
import uvicorn

class InputData(BaseModel):
    ImageorVideo_for_restored_path: str
    function_Facial_rejuvenation: bool
    function_image_enhancement: bool
    function_video_enhancement: bool

class OutputData(BaseModel):
    Digital_face1_path: str

if __name__ == '__main__':
    app = FastAPI()
    @app.post('/Digital_face1', response_model=OutputData)
    def reply(data: InputData):
        if data.function_Facial_rejuvenation == True:
            result = FaceRestorer.run_face_restoration(input_path=data.ImageorVideo_for_restored_path, has_aligned=True)
        elif data.function_image_enhancement == True:
            result = FaceRestorer.run_face_restoration(input_path=data.ImageorVideo_for_restored_path, fidelity_weight=0.7)
        else:
            result = FaceRestorer.run_face_restoration(input_path=data.ImageorVideo_for_restored_path, bg_upsampler='realesrgan', face_upsample=True)
        return OutputData(Digital_face1_path=result)
    
    uvicorn.run(app, host="127.0.0.1", port=9997)