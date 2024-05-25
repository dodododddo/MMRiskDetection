from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from utils import pipeline

if __name__ == '__main__':
    model = VideoLlavaForConditionalGeneration.from_pretrained("./videoLLaVA",local_files_only=True, device_map='auto')
    processor = VideoLlavaProcessor.from_pretrained("./videoLLaVA",  local_files_only=True, device_map='auto')
    # prompt = "USER: <video>Why is this video funny? ASSISTANT:"
    # prompt = "USER: <video>视频中有任何风险内容吗？比如色情、暴力、仇恨、谣言或是使用了换脸技术的痕迹？如果有请指出具体的风险内容，如果没有，请直接回答无风险。ASSISTANT:"
    # prompt = "USER: <video> Divide the video content into 4 segments for brief descriptions, and include time markers.ASSISTANT:"
    prompt = "USER: <video> How do you understand the characteristics of risky videos? Does this video meet certain characteristics? ASSISTANT:"
    video_path = "./data/p_demo.mp4"
    pipe = pipeline(model, processor)
    answer = pipe(prompt, video_path)
    print('\n' + answer)
