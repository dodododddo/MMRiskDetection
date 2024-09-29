
def web_mark(web_text):
    content = f"来源: 网页, 网页文字内容: {web_text}"
    return content

def image_mark(image_text, image_content, image_synthesis):
    content = f"来源: 图片, 图片上文字内容: {image_text}, 图片内容: {image_content}, 图片人脸是否伪造：{image_synthesis}"
    return content

def audio_mark(audio_text):
    content = f"来源: 音频, 音频内容: {audio_text}"
    return content

def text_mark(text):
    content = f"来源: 文本, 文本内容: {text}"
    return content

def file_mark(file_text, file_image_text):
    content = f"来源: 文件, 文件文字内容: {file_text}, 文件图片中的文字内容: {file_image_text}"
    return content

def video_mark(video_describe, video_risk, audio_text, video_df_detection, audio_df_detection):
    content = f"来源: 视频, 视频内容: {video_describe}, 视频风险内容: {video_risk}, 视频中音频内容: {audio_text}, 视频是否换脸：{video_df_detection}, 声音是否合成：{audio_df_detection}"
    return content


