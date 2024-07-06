import os
import ffmpeg
import datetime
# /data1/home/jrchen/MMRiskDetection/VideoPipeline/model
#scp -r D:\cs\python\videofact-wacv-2024-main \ jrchen@10.249.189.249:/data1/home/jrchen/MMRiskDetection/VideoPipeline/model/videofact

# 视频剪辑
def cut_vedio(input_video, start_time, end_time, output_video):
    input_stream = ffmpeg.input(input_video)
    video = (
        input_stream
        .video
        .filter('trim', start=start_time, end=end_time)
        .filter('setpts', 'PTS-STARTPTS')   # 重新设置时间戳
    )
    audio = (
        input_stream
        .audio
        .filter('atrim', start=start_time, end=end_time)
        .filter('asetpts', 'PTS-STARTPTS')
    )
    ffmpeg.output(video, audio, output_video).run(overwrite_output=True)

# 视频抽帧
def extract_frames(input_video, output_pattern, fps):
    (
        ffmpeg
        .input(input_video)
        .output(output_pattern, vf=f'fps={fps}')  # 设置fps=10表示每10秒提取一帧
        .run(overwrite_output=True)
    )

def video_to_image(input_video,output_image,timestamp):
    (
    ffmpeg
    .input(input_video, ss=timestamp)
    .output(output_image, vframes=1)
    .run(overwrite_output=True)
    )

def get_video_time(input_video):
    probe = ffmpeg.probe(input_video)
    # 获取视频的时长（以秒为单位）
    duration = float(probe['format']['duration'])
    return duration


# 获取视频宽高
def get_video_dimensions(input_video):
    probe = ffmpeg.probe(input_video)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError("No video stream found in the input video.")
    
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return width, height

# 视频窗口裁剪
# 左上角（0,0），x向右，y向下
def crop_video(input_video, x, y, width, height, output_video):
    (
        ffmpeg
        .input(input_video)
        .crop(x, y, width, height)
        .output(output_video)
        .run(overwrite_output=True)
    )

#分离音轨
def div_vioce(input_video, output_audio):
    # 使用 ffmpeg 从视频中提取音频并保存为 WAV 文件
    ffmpeg.input(input_video).output(output_audio, format='wav').run()


if __name__ == '__main__':
    # cut_vedio('./data/p_demo.mp4', '00:02:00', '00:03:40', './data/p_cut_demo.mp4')
    # extract_frames('./data/p_demo.mp4', './data/frames/output_%04d.png')
    # width, height = get_video_dimensions('./data/p_cut_demo.mp4')
    # print(width,height)
    # crop_video('./data/p_cut_demo.mp4', 0, height / 4+20, width, width-70, './data/p_crop_demo.mp4')
    # extract_frames('./data/p_crop_demo.mp4','./data/frames_p/output_%04d.png')
    # extract_frames('./data/n_demo.mp4','./data/frames_n/output_%04d.png')
    # video_list = [1302,1304,1305,1307,1308]
    # data/DFMNIST+/fake_dataset/left_slope/1304.mp4
    # input_video = [f'./data/DFMNIST+/fake_dataset/left_slope/{i}.mp4' for i in video_list]
    # output_image = [f'./data/DFMINST+_image/fake/{i}.png' for i in video_list]
    # for i in range(len(video_list)):
    #     if not os.path.exists(input_video[i]):
    #         print(f"文件不存在: {input_video[i]}")
    #     time = get_video_time(input_video[i])
    #     middle_timestamp = time // 2
    #     middle_time_str = str(datetime.timedelta(seconds=middle_timestamp))
    #     print(middle_time_str)
    #     video_to_image(input_video[i],output_image[i],middle_time_str)
    # div_vioce('./data/p_crop_demo.mp4','./data/p_crop_demo.wav')
    extract_frames('./data/DFMNIST+/fake_dataset/blink/4063.mp4','./data/DFMINST+_image/fake/blink/4063/output_%04d.png',10)
