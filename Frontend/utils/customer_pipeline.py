from sub_module import *
from mark import *
from structure import *
import re
from pymongo import MongoClient
import time

client = MongoClient('mongodb://jrchen:jrchen@127.0.0.1:27017/jrchen')
db = client['jrchen']
collection = db['text']

def insert(input, output):
    # print(output_structure(output))
    try:
        document = {'input': input, 
                    'output': output_structure(output)}
        collection.insert_one(document)
    except:
        document = {'input': input, 
                    'output': output}
        collection.insert_one(document)

def web_pipeline(web_url):
    webData = web_module(web_url)
    if webData['image_paths'] != []:
        webImage_data = webImage_module(webData['image_paths'])
        webImage_data = '存在色情信息' if webImage_data else '不存在色情信息'
    else:
        webImage_data = '未检测'
    yield webImage_data, ''
    text = text_module(web_mark(webData['text']))
    result_text = ''
    for t in text:
        result_text += t
        yield webImage_data, result_text.replace(' ','')
    insert(web_structure(webData['text']), result_text.replace(' ',''))
    

def image_pipeline(image_path):
    imageData = image_module(image_path)
    # print(imageData)
    image_synthesis = '未检测' if imageData['synthesis'] is None else '伪造' if imageData['synthesis'] else '真实'
    yield image_synthesis, '', ''
    image_sex = '未检测' if imageData['sex'] is None else '存在色情信息' if imageData['sex'] else '不存在色情信息'
    yield image_synthesis, image_sex, ''
    if imageData['have_characters'] is not None:
        if imageData['have_characters'] == True:
            text = image_mark(imageData['ocr_content'], imageData['image_content'], imageData['risk'], image_synthesis)
        else:
            text = image_mark('无', imageData['image_content'], imageData['risk'], image_synthesis)
        text = text_module(text)
        result_text = ''
        for t in text:
            result_text += t
            yield image_synthesis, image_sex, result_text.replace(' ', '')
    else:
        result_text = '未检测'
        yield image_synthesis, image_sex, result_text.replace(' ', '')
    insert(image_structure(imageData['ocr_content'], imageData['image_content'], imageData['risk']), result_text.replace(' ', ''))


def audio_pipeline(audio_path):
    audio_text = audio_to_text_module(audio_path)
    audio_detect = audio_detect_module(audio_path)
    audio_detect = '未检测' if audio_detect['fake_or_not'] is None else '伪造' if audio_detect['fake_or_not'] else '真实'

    yield audio_detect, ''
    text = text_module(audio_mark(audio_text))
    result_text = ''
    for t in text:
        result_text += t
        yield audio_detect, result_text.replace(' ', '')
    insert(audio_structure(audio_text), result_text.replace(' ', ''))
    

def text_pipeline(text):
    ext_text = ext_module(text)
    yield '', '', ext_text['ext_text']
    detect_text = text_module(text_mark(text))
    result_text1 = ''
    result_text2 = ''
    flag = True
    for t in detect_text:
        if flag:
            result_text1 += t
        else:
            result_text2 += t
        if '7.' in result_text1:
            flag = False
            result_text1 = result_text1.split('7.')
            result_text2 = result_text1[1]
            result_text1 = result_text1[0]
        # result_text2 = re.sub(r'(案例 \d)', r'\n\1', result_text2)
        yield result_text1.replace(' ',''), result_text2.replace(' ',''), ext_text['ext_text']
    insert(text_structure(text), (result_text1+result_text2).replace(' ',''))

def file_pipeline(file_path):
    fileData = file_module(file_path)
    if fileData['image_path'] == '':
        fileImageData = fileImage_module(fileData['image_path'])
    else:
        fileImageData = {'text':[], 'sex': None}
    file_sex = '未检测' if fileImageData['sex'] is None else '存在色情因素' if fileImageData['sex'] else '不存在色情因素' 
    yield '', file_sex
    fileImageText = ''.join(fileImageData['text'])
    text = text_module(file_mark(fileData['text'], fileImageText))
    result_text = ''
    for t in text:
        result_text += t
        yield result_text.replace(' ', ''), file_sex
    insert(file_structure(fileData['text'], fileImageText), result_text.replace(' ', ''))
    
    

def video_pipeline(video_path): 
    video_extract = video_extract_module(video_path)
    video_message = video_message_module(video_path)
    video_fake_face = video_fakeface_module(video_path)
    videoImageData = videoImage_module(video_extract['image']) 
    video_sex = '未检测' if videoImageData['sex'] is None else '存在色情信息' if videoImageData['sex'] else '不存在色情信息'
    yield video_sex, '', '', '', ''
    if (video_message['deepfake_detection'] is None and video_fake_face is None):
        video_deepfake_detection = None
    else:
        if video_message['deepfake_detection'] is None:
            video_message['deepfake_detection'] = False
        if video_fake_face is None:
            video_fake_face = False
        video_deepfake_detection = video_message['deepfake_detection'] | video_fake_face
    vid_df_text = '未检测' if video_deepfake_detection is None else '伪造' if video_deepfake_detection else '真实'
    yield video_sex , vid_df_text, '', '', ''
    audio_text, audio_detect = '', ''
    if video_extract['audio'] != 'no_audio':
        audio_text = audio_to_text_module(video_extract['audio'])
        audio_detect = audio_detect_module(video_extract['audio'])
    audio_deepfake_detection = audio_detect['fake_or_not'] if audio_detect != '' else None
    aud_df_text = '未检测' if audio_deepfake_detection is None else '伪造' if audio_deepfake_detection else '真实'
    yield video_sex , vid_df_text, aud_df_text, '', ''
    syn_deepfake_detection = None if (audio_deepfake_detection is None or video_deepfake_detection is None) else video_deepfake_detection or audio_deepfake_detection
    syn_df_text = '未检测' if syn_deepfake_detection is None else '伪造' if syn_deepfake_detection else '真实'
    yield video_sex , vid_df_text, aud_df_text, syn_df_text, ''
    video_deepfake_detection = '未检测' if video_deepfake_detection is None else '伪造' if video_deepfake_detection else '真实'
    audio_deepfake_detection = '未检测' if audio_deepfake_detection is None else '伪造' if audio_deepfake_detection else '真实'
    syn_deepfake_detection = '未检测' if syn_deepfake_detection is None else '伪造' if syn_deepfake_detection else '真实'

    text = text_module(video_mark(video_message['describe'], video_message['risk'] , audio_text, video_deepfake_detection, audio_deepfake_detection))
    result_text = ''
    for t in text:
        result_text += t
        yield video_sex , vid_df_text, aud_df_text, syn_df_text, result_text.replace(' ','')
    insert(video_structure(video_message['describe'], video_message['risk'] , audio_text), result_text)
   
def digital_humans_pipeline(image_path, audio_path, text):
    image_path = face_restoration_module(image_path, facial=False, image=True, video=False)
    audio_text = audio_to_text_module(audio_path)
    audio_path = ssgen_module(audio_path, audio_text, text)
    video_path = video_generator_module(image_path, audio_path, function_A=False, function_B=False, function_C=True)
    video_path = face_restoration_module(video_path, facial=False, image=False, video=True)
    return video_path

def facefusion_pipeline(image_path, video_path):
    video_path = facefusion_module(image_path, video_path)
    return video_path

def text_pipeline_test(text):
    # detect_text = text_module(text_mark(text))
    detect_text = '1. 是否存在风险内容：是2. 风险摘要：该文本通过提供外卖取件信息，可能存在诈骗风险，因为链接可能指向恶意网站或用于诈骗用户的信息。3. 识别的风险类型：诈骗4. 风险内容分析：   - 诈骗：文本中提供了一个链接，要求用户点击以取件，这可能是一个诈骗手段，用于获取用户的个人信息或进行其他形式的欺诈。5. 防范建议：   - 诈骗：不要点击未知来源的链接，尤其是那些要求提供个人信息或进行转账的链接。直接联系美团客服或使用美团官方应用进行取件。'
    detect_text = re.sub(r'(\d+\.)', r'\n\1', detect_text)
    detect_text = detect_text.lstrip()
    return detect_text

if __name__ == '__main__':
    # print(text_pipeline_test('【美团】您的外卖已放在哈工大南门一号柜外卖柜73格口，点击 dpurl.cn/pxfbjRLz 或使用7073取件'))
    # print(video_extract_module('../Frontend/demo/p_demo.mp4'),video_message_module('../Frontend/demo/p_demo.mp4'))
    print(digital_humans_pipeline('../../Frontend/demo/pq.jpg', '../../Frontend/demo/SSB00050005.wav', '要就有要就有'))