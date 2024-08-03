import json

def web_structure(web_text):
    content = {'来源': '网页', 
               '网页文字内容': web_text}
    return content

def image_structure(image_text, image_content, risk_content):
    content = {'来源': '图片', 
               '图片上文字内容': image_text, 
               '图片内容': image_content, 
               '图片风险内容': risk_content}
    return content

def audio_structure(audio_text):
    content = {'来源': '音频', 
               '音频内容': audio_text}
    return content

def text_structure(text):
    content = {'来源': '文本', 
               '文本内容': text}
    return content

def file_structure(file_text, file_image_text):
    content = {'来源': '文件', 
               '文件文字内容': file_text, 
               '文件图片中的文字内容': file_image_text}
    return content

def video_structure(video_describe, video_risk, audio_text):
    content = {'来源': '视频', 
               '视频内容': video_describe, 
               '视频风险内容': video_risk, 
               '视频中音频内容': audio_text}
    return content

def output_structure(text):
    data = {}
    lines = text.strip().split('\n')
    print(lines)
    current_key = None
    for line in lines:
        if line.strip():
            if line.startswith('1.'):
                current_key = '是否存在风险内容'
                data[current_key] = line.split(current_key + '：')[1].strip().replace(' ', '').replace('-', '')
            elif line.startswith('2.'):
                current_key = '风险摘要'
                data[current_key] = line.split(current_key + '：')[1].strip().replace(' ', '').replace('-', '')
            elif line.startswith('3.'):
                current_key = '识别的风险类型'
                data[current_key] = line.split(current_key + '：')[1].strip().replace(' ', '').replace('-', '')
            elif line.startswith('4.'):
                current_key = '风险内容分析'
                data[current_key] = line.split(current_key + '：')[1].strip().replace(' ', '').replace('-', '')
            elif line.startswith('5.'):
                current_key = '防范建议'
                data[current_key] = line.split(current_key + '：')[1].strip().replace(' ', '').replace('-', '')
            elif line.startswith('6.'):
                current_key = '细分诈骗类型'
                data[current_key] = line.split(current_key + '：')[1].strip().replace(' ', '').replace('-', '')
            elif line.startswith('7.'):
                current_key = '相似案例'
                data[current_key] = line.split(current_key + '：')[1].strip().replace(' ', '').replace('-', '')
    return json.loads(json.dumps(data, ensure_ascii=False, indent=7))

