import os
import logging
import pandas as pd
import numpy as np
import torch
import requests
import ffmpeg
from PIL import Image
from datetime import timedelta
# from vllm import LLM, SamplingParams
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from videofact_main.inference_single import get_videofact_model, load_single_video, process_single_video
from videofact_main.utils import *
from rich.logging import RichHandler
from typing import *
import time
import psutil

def check_port_in_use(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port:
            return conn.pid  # 返回占用该端口的进程 ID
    return None

videofact_df_threshold = 0.24

def format_time(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

class VideoPipeline():
    def __init__(self) -> None:
        self.VideoFACT_df = get_videofact_model("my")

    def video_to_text(self, video_path):
        '''用抽帧检测视频内容'''

        folder = '../DataBuffer/VideoImageBuffer'
        image_paths = []
        for filename in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
        
        length = len(image_paths)
        
        prompt1 = ["[INST] <image>What is shown in this image?Please describe the content of this image in one sentence.[/INST]"] * length
        prompt2 = ["[INST] <image>Do these images contain any evidence of pornography, violence, hatred, rumors, or face-changing techniques?Pay more attention to the people.Please reply in one sentence.[/INST]"] * length
        # prompt1 = ["这张图片展示了什么？请用一句话描述这张图片的内容。"] * length
        # prompt2 = ["这些图片中是否包含色情、暴力、仇恨、谣言或换脸技术的证据？请多注意人们的表现。请用一句话回复。"] * length
       
        outputs = requests.post('http://127.0.0.1:9000/vlm', json={'prompt': prompt1 + prompt2, 'image_path': image_paths + image_paths}).json()
        
        describe = outputs['text'][:length]
        risk = outputs['text'][length:]
        
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
        print('step1')
        print(time.time())
        describe, risk = self.video_to_text(video_path)
        print('step2')
        print(time.time())
        deepfake_detection = self.video_deepfake_detection(video_path)
        print('step3')
        print(time.time())
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
    
if __name__ == '__main__':
    vp = VideoPipeline()
    vp.video_to_text('./data/p_demo.mp4')
