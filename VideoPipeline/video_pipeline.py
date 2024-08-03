import os
import logging
import pandas as pd
import numpy as np
import torch
import requests
import ffmpeg
from PIL import Image
from datetime import timedelta
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from videofact_main.inference_single import get_videofact_model, load_single_video, process_single_video
from videofact_main.utils import *
from rich.logging import RichHandler
from typing import *

videofact_df_threshold = 0.24

def format_time(seconds):
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

class VideoPipeline():
    def __init__(self) -> None:
        self.llm = LLM(model="./model/llava-v1.6-mistral-7b-hf", max_model_len=4096, gpu_memory_utilization=0.2)
        self.VideoFACT_df = get_videofact_model("my")

    def image_to_text(self, image_path, prompt):
        image = Image.open(image_path)
        sampling_params = SamplingParams(temperature=0.8,
                                        top_p=0.95,
                                        max_tokens=256)
        outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image
                    }
                },
                sampling_params=sampling_params)
        generated_text = ''
        for o in outputs:
            generated_text += o.outputs[0].text
        return generated_text


    def video_to_text(self, video_path):
        '''用抽帧检测视频内容'''

        folder = '../DataBuffer/VideoImageBuffer'
        images = []
        for filename in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                try:
                    img = Image.open(img_path)
                    images.append(img)
                except (IOError, OSError) as e:
                    print(f"Error loading image {img_path}: {e}")

        prompt1 = "[INST] <image>What is shown in this image?Please describe the content of this image in one sentence.[/INST]"
        prompt2 = "[INST] <image>Do these images contain any evidence of pornography, violence, hatred, rumors, or face-changing techniques?Pay more attention to the people.Please reply in one sentence.[/INST]"
        sampling_params = SamplingParams(temperature=0.8,
                                        top_p=0.95,
                                        max_tokens=256)
        describe = []
        risk = []
        task1 = [{
                    "prompt": prompt1,
                    "multi_modal_data": {
                        "image": img
                    }
                } for img in images]
        task2 = [{
                    "prompt": prompt2,
                    "multi_modal_data": {
                        "image": img
                    }
                } for img in images]
        outputs = self.llm.generate(
        task1+task2,
        sampling_params=sampling_params)
        generated_text = ""
        for o in outputs:
            if len(describe) < len(images):
                describe.append(o.outputs[0].text)
            else:
                risk.append(o.outputs[0].text)
        print(describe, risk)
        return describe, risk

    @torch.no_grad()
    def video_deepfake_detection(
        self,
        video_path: str,
        shuffle = True,
        max_num_samples = 100,
        sample_every = 5,
        batch_size = 2,
        num_workers = 5
    ) -> List[str]:
        '''换脸检测'''
        if video_path is None:
            raise ValueError("video_path cannot be None")

        dataloader = load_single_video(
            video_path,
            shuffle,
            int(max_num_samples),
            int(sample_every),
            int(batch_size),
            int(num_workers),
        )
        results = process_single_video(self.VideoFACT_df, dataloader)
        _, _, scores = list(zip(*results))

        return (
            sum(scores) / len(scores),
            True if sum(scores) / len(scores) > videofact_df_threshold else False
        )

    def __call__(self, video_path:str):
        describe, risk = self.video_to_text(video_path)
        deepfake_detection = self.video_deepfake_detection(video_path)
        return VideoData(describe, risk, deepfake_detection[-1]) 
    
class VideoData():
    def __init__(self, describe:list, risk:list, deepfake_detection:bool):
        '''
        describe:视频描述
        risk:视频风险
        deepfake_detection:换脸或真实
        '''
        self.describe = describe
        self.risk = risk
        self.deepfake_detection = deepfake_detection

    def __str__(self):
        return str(self.__dict__)
    
# if __name__ == '__main__':
#     vp = VideoPipeline()
#     vp.video_to_text('./data/p_demo.mp4')
