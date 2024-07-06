import os
import logging
import pandas as pd
import numpy as np
import torch
import requests
import ffmpeg
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from videofact_main.inference_single import get_videofact_model, load_single_video, process_single_video
from videofact_main.utils import *
from rich.logging import RichHandler
from typing import *

VideoFACT_xfer = None 
VideoFACT_df = None

videofact_df_threshold = 0.33
videofact_xfer_threshold = 0.4

def extract_frames_as_images(video_path):
    # 用于存储抽取的帧的列表
    images = []

    # 获取视频信息
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    
    # 计算每隔多少帧抽取一帧
    interval = max(num_frames // 10, 1)

    # 运行ffmpeg命令，将视频输出到管道中
    process = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    frame_count = 0
    try:
        while True:
            # 每一帧的大小：宽 * 高 * 每个像素的字节数 (3 bytes for RGB)
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes or len(images) >= 10:
                break

            # 仅保存每间隔interval的帧
            if frame_count % interval == 0:
                # 将字节数据转换为numpy数组
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                # 将numpy数组转换为PIL.Image
                image = Image.fromarray(frame)
                images.append(image)

            frame_count += 1
    finally:
        # 确保进程正确结束
        process.stdout.close()
        if process.stderr:
            process.stderr.close()
        process.wait()
    return images


def video_to_text(video_path):
    '''用抽帧检测视频内容'''
    # Load the model in half-precision
    model = LlavaNextForConditionalGeneration.from_pretrained("./model/llava-v1.6-mistral-7b-hf", 
                                                            torch_dtype=torch.float16, 
                                                            device_map="auto")
                                                            #   use_flash_attention_2=True)
    processor = AutoProcessor.from_pretrained("./model/llava-v1.6-mistral-7b-hf")

    images = extract_frames_as_images(video_path)

    each_prompt = "[INST] <image>\n这张图片中是否包含暴力、色情、犯罪因素或AI合成痕迹？假如有以上因素，请只输出对应因素和相应行为。不要有多于输出。 [/INST]"
    prompt = [each_prompt for _ in images]

    # print(each_prompt)
    for i in range(len(prompt)):
        inputs = processor(text=[prompt[i]], images=[images[i]], padding=True, return_tensors="pt").to(model.device)

        generate_ids = model.generate(**inputs, max_new_tokens=256, pad_token_id=2)
        text_outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        t = text_outputs[0]
        t = t.split('[/INST]')
        print('图片{}:'.format(i+1) + t[1]) 

@torch.no_grad()
def video_forgery_detection(
    video_path: str,
    shuffle = True,
    max_num_samples = 100,
    sample_every = 5,
    batch_size = 1,
    num_workers = 8
) -> List[str]:
    '''阴影检测'''
    if video_path is None:
        raise ValueError("video_path cannot be None")

    global VideoFACT_xfer
    if VideoFACT_xfer is None:
        VideoFACT_xfer = get_videofact_model("xfer")

    dataloader = load_single_video(
        video_path,
        shuffle,
        int(max_num_samples),
        int(sample_every),
        int(batch_size),
        int(num_workers),
    )
    results = process_single_video(VideoFACT_xfer, dataloader)
    result_frame_paths, idxs, scores = list(zip(*results))
    decisions = ["Forged" if score > videofact_xfer_threshold else "Authentic" for score in scores]

    return (
        f"Frame: {idxs[0]}, Score: {sum(scores) / len(scores):.5f}",
        "Forged" if sum(scores) / len(scores) > videofact_xfer_threshold else "Authentic"
    )


@torch.no_grad()
def video_deepfake_detection(
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

    global VideoFACT_df
    if VideoFACT_df is None:
        VideoFACT_df = get_videofact_model("df")

    dataloader = load_single_video(
        video_path,
        shuffle,
        int(max_num_samples),
        int(sample_every),
        int(batch_size),
        int(num_workers),
    )
    results = process_single_video(VideoFACT_df, dataloader)
    result_frame_paths, idxs, scores = list(zip(*results))
    decisions = ["Deepfaked" if score > videofact_df_threshold else "Authentic" for score in scores]

    return (
        sum(scores) / len(scores),
        "Deepfaked" if sum(scores) / len(scores) > videofact_df_threshold else "Authentic"
    )

def data_needed(filePath):
    '''获取文件列表'''
    file_name = list()        #新建列表
    for i in os.listdir(filePath):        #获取filePath路径下所有文件名
        data_collect = ''.join(i)        #文件名字符串格式
        file_name.append(filePath + data_collect)        #将文件名作为列表元素填入
    print("获取filePath中文件名列表成功")        #打印获取成功提示
    return(file_name)        #返回列表


if __name__ == "__main__":
    video_path = './data/p_demo.mp4'
    # print(result)
    # video_path = './videofact_main/examples/df/donald_trump_deepfake.mp4'
    # video_path = './videofact_main/examples/df/sheeran-24kgold_deepfake.mp4'
    # video_path = './data/DFMNIST+/real_dataset/selected_test/id10001#7w0IBEWc9Qw#001298#001705.mp4'
    # video_path = './data/DFMNIST+/fake_dataset/blink/4000.mp4'
    # result = video_deepfake_detection(video_path)
    video_to_text(video_path)
    # print('\n'+result[-1])
    # print('\n'+str(result[-2]))