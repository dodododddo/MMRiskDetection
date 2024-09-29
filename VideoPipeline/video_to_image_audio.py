import ffmpeg
import numpy as np
from PIL import Image
from datetime import timedelta
import os
import shutil

def format_time(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def extract_frames_as_images(video_path):
    output_folder = '../DataBuffer/VideoImageBuffer'
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 获取视频信息
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    duration = float(video_info['duration'])
    
    output_pattern = os.path.join(output_folder, 'frame_%02d.png')
    fps = 10 / duration 

    # 运行ffmpeg命令，将视频输出到管道中
    (
    ffmpeg
    .input(video_path)
    .output(output_pattern, vf=f'fps={fps}')
    .run(overwrite_output=True)
    )
    return output_folder

def extract_audio_from_video(video_path):
    name = video_path.split('/')[-1].split('.')[0]
    output_folder = '../DataBuffer/VideoAudioBuffer'
    os.makedirs(output_folder, exist_ok=True)
    audio_path = output_folder +  '/' + name + '.wav'
    try:
        # 获取视频信息
        probe = ffmpeg.probe(video_path)
        # 检查是否有音频流
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if audio_streams:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, format='wav') 
                .run(overwrite_output=True)
            )
            print(f"Audio extracted successfully to {audio_path}")
            return audio_path
        else:
            return 'no_audio'
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")
        return 'no_audio'

    
def video_to_image_audio(video_path):
    image = extract_frames_as_images(video_path)
    audio = extract_audio_from_video(video_path)
    return SendData(image, audio)

class SendData():
    def __init__(self, image, audio):
        '''
        image:图片文件夹路径
        audio:分离出的音频路径
        '''
        self.image = image
        self.audio = audio

    def get(self):
        print(self.image, self.audio)

if __name__ == '__main__':
    extract_frames_as_images('./data/p_demo.mp4')

    
