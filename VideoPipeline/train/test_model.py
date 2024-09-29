from VideoPipeline.video_pipeline import extract_frames_as_images
from io import BytesIO
import requests
from PIL import Image
from vllm import LLM, SamplingParams
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import os

def image_to_text(image_path, prompt):
    '''用抽帧检测视频内容'''
    image = Image.open(image_path)
    result = []
    llm = LLM(model="./model/llava-v1.6-mistral-7b-hf", max_model_len=4096, gpu_memory_utilization=0.2)
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=256)
    outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        }
    },
    sampling_params=sampling_params)
    generated_text = ""
    for o in outputs:
        generated_text += o.outputs[0].text
    return generated_text

class InputData(BaseModel):
    image_path: str
    prompt:str
    
class OutputData(BaseModel):
    text:str

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # app = FastAPI()
    # @app.post('/image_to_text', response_model=OutputData)
    # def reply(data: InputData):
    #     result = image_to_text(data.image_path, data.prompt)
    #     return OutputData(text=result)

    # uvicorn.run(app, host="127.0.0.1", port=1930)
    prompt = '[INST] <image>Where is the face in this picture? Please use the lower right corner as the coordinate origin, and one pixel width as the unit length.Please output only the 2D coordinates.[/INST]'
    image_path = './data/image/test_frame_104.png'
    print(image_to_text(image_path, prompt))