from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from utils import pipeline,image_generate

if __name__ == '__main__':
    model = VideoLlavaForConditionalGeneration.from_pretrained("model/videoLLaVA",local_files_only=True, device_map='auto')
    processor = VideoLlavaProcessor.from_pretrained("model/videoLLaVA",  local_files_only=True, device_map='auto')
    # prompt = "USER: <video>Why is this video funny? ASSISTANT:"
    # prompt = "USER: <video>视频中有任何风险内容吗？比如色情、暴力、仇恨、谣言或是使用了换脸技术的痕迹？如果有请指出具体的风险内容，如果没有，请直接回答无风险。ASSISTANT:"
    # prompt = "USER: <video> Divide the video content into 4 segments for brief descriptions, and include time markers.ASSISTANT:"
    prompt = "USER: <video> How do you understand the characteristics of risky videos? Does this video meet certain characteristics? ASSISTANT:"
    video_list = [1302,1304,1305,1307,1308]
    # data/DFMNIST+/fake_dataset/left_slope/1304.mp4
    video_paths = [f'./data/DFMNIST+/fake_dataset/left_slope/{i}.mp4' for i in video_list]
    for p in video_paths:
        video_path = p
        pipe = pipeline(model, processor)
        answer = pipe(prompt, video_path)
        print('\n' + answer)

    # prompt = "USER: <image> How do you understand the characteristics of risky images? Does this image meet certain characteristics? ASSISTANT:"
    # image_path = './data/frames/output_0001.png'
    # answer = image_generate(model, processor, prompt, image_path)
    # print('\n' + answer)

