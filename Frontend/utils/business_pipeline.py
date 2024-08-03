from sub_module import *
from mark import *
from structure import *
import re
from pymongo import MongoClient
import time

paths=['../Frontend/demo/test.mp4','../Frontend/demo/output0.wav', '../Frontend/demo/F_PGN2_00003.png',
       '您好，欢迎加入百花齐放群，这里有很多赚钱的机会。请点击链接下载STQ软件，按照提示购买春、夏、秋、东即可获得收益。记得先充值哦，这样才能开始赚钱。',
       'https://18mh.org/', '../Frontend/demo/template.pdf']

def risk_count(count):
    return ('有风险视频数：' + str(count[0]) + '\n有风险图片数：' + str(count[1])
        + '\n有风险音频数：' + str(count[2]) + '\n有风险文本数：' + str(count[3])
        + '\n有风险网页数：' + str(count[4]) + '\n有风险文件数：' + str(count[5]))
def divide_line():
    return '\n--------------------------------------------------------------'

def pipeline(paths=paths): 
    # print(video_path)
    result_text = ''
    i=0
    count = [0, 0, 0, 0, 0, 0]
    for path in paths:
        i += 1
        if i != 1:
            result_text += '\n'
        post = path.split('.')[-1]
        pre = path.split('.')[0]


        if post == 'mp4':
            video_path = path
            result_text += '\n[' + str(i) + ']' + '视频路径：'+ video_path
            video_extract = video_extract_module(video_path)
            video_message = video_message_module(video_path)
            video_fake_face = video_fakeface_module(video_path)
            videoImageData = videoImage_module(video_extract['image']) 
            video_sex = '未检测' if videoImageData['sex'] is None else '存在色情信息' if videoImageData['sex'] else '不存在色情信息'
            result_text += '\n视频色情检测：' + video_sex
            yield result_text, risk_count(count)
            if (video_message['deepfake_detection'] is None and video_fake_face is None):
                video_deepfake_detection = None
            else:
                if video_message['deepfake_detection'] is None:
                    video_message['deepfake_detection'] = False
                if video_fake_face is None:
                    video_fake_face = False
                video_deepfake_detection = video_message['deepfake_detection'] | video_fake_face
            vid_df_text = '未检测' if video_deepfake_detection is None else '伪造' if video_deepfake_detection else '真实'
            result_text += '\n视频伪造检测：' + vid_df_text
            yield result_text, risk_count(count)
            audio_text, audio_detect = '', ''
            if video_extract['audio'] != 'no_audio':
                audio_text = audio_to_text_module(video_extract['audio'])
                audio_detect = audio_detect_module(video_extract['audio'])
            audio_deepfake_detection = audio_detect['fake_or_not'] if audio_detect != '' else None
            aud_df_text = '未检测' if audio_deepfake_detection is None else '伪造' if audio_deepfake_detection else '真实'
            result_text += '\n声音伪造检测：' + aud_df_text
            yield result_text, risk_count(count)
            syn_deepfake_detection = None if (audio_deepfake_detection is None or video_deepfake_detection is None) else video_deepfake_detection or audio_deepfake_detection
            syn_df_text = '未检测' if syn_deepfake_detection is None else '伪造' if syn_deepfake_detection else '真实'
            result_text += '\n综合伪造检测：' + syn_df_text
            yield result_text, risk_count(count)
            video_deepfake_detection = '未检测' if video_deepfake_detection is None else '伪造' if video_deepfake_detection else '真实'
            audio_deepfake_detection = '未检测' if audio_deepfake_detection is None else '伪造' if audio_deepfake_detection else '真实'
            syn_deepfake_detection = '未检测' if syn_deepfake_detection is None else '伪造' if syn_deepfake_detection else '真实'

            text = text_module(video_mark(video_message['describe'], video_message['risk'] , audio_text, video_deepfake_detection, audio_deepfake_detection))
            flag = False
            text1 = ''
            for t in text:
                if not flag:
                    result_text += '\n视频风险检测：\n' + t.replace(' ','')
                    flag = True
                else:
                    result_text += t.replace(' ', '')
                text1 += t
                yield result_text, risk_count(count)
            result_text += divide_line()
            text1 = text1.split('2')[0].split('：')[-1]
            if '是' in text1 or video_deepfake_detection == '伪造' or audio_deepfake_detection == '伪造' or video_sex == '存在色情信息':
                count[0] += 1
            yield result_text, risk_count(count)


        elif post == 'png' or post == 'jpg' or post == 'jpeg':
            image_path = path
            result_text += '\n[' + str(i) + ']' + '图片路径：'+ image_path
            imageData = image_module(image_path)
            image_synthesis = '未检测' if imageData['synthesis'] is None else '伪造' if imageData['synthesis'] else '真实'
            result_text += '\n图片合成检测：' + image_synthesis
            yield result_text, risk_count(count)
            image_sex = '未检测' if imageData['sex'] is None else '存在色情信息' if imageData['sex'] else '不存在色情信息'
            result_text += '\n图片色情检测：' + image_sex
            yield result_text, risk_count(count)
            if imageData['have_characters'] is not None:
                if imageData['have_characters'] == True:
                    text = image_mark(imageData['ocr_content'], imageData['image_content'], imageData['risk'], image_synthesis)
                else:
                    text = image_mark('无', imageData['image_content'], imageData['risk'], image_synthesis)
                flag = False
                text = text_module(text)
                text1 = ''
                for t in text:
                    if not flag:
                        result_text += '\n图片风险检测：\n' + t.replace(' ', '')
                        flag = True
                    else:
                        result_text += t.replace(' ', '')
                    text1 += t
                    yield result_text, risk_count(count)
            else:
                result_text += '\n图片风险检测：' + '未检测'
                yield result_text, risk_count(count)
            result_text += divide_line()
            text1 = text1.split('2')[0].split('：')[-1]
            if '是' in text1 or image_sex == '存在色情信息' or image_synthesis == '伪造':
                count[1] += 1
            yield result_text, risk_count(count)


        elif post == 'mp3' or post == 'wav':
            audio_path = path
            result_text += '\n[' + str(i) + ']' + '音频路径：'+ audio_path
            audio_text = audio_to_text_module(audio_path)
            audio_detect = audio_detect_module('../' + audio_path)
            audio_detect = '未检测' if audio_detect['fake_or_not'] is None else '伪造' if audio_detect['fake_or_not'] else '真实'
            result_text += '\n音频伪造检测：' + audio_detect
            yield result_text, risk_count(count)
            text = text_module(audio_mark(audio_text))
            flag = False
            text1 = ''
            for t in text:
                if not flag:
                    result_text += '\n音频风险检测：\n' + t.replace(' ','')
                    flag = True
                else:
                    result_text += t.replace(' ', '')
                text1 += t
                yield result_text, risk_count(count)
            result_text += divide_line()
            text1 = text1.split('2')[0].split('：')[-1]
            if '是' in text1 or audio_detect == '伪造':
                count[2] += 1
            yield result_text, risk_count(count)


        elif post == 'pdf':
            file_path = path
            result_text += '\n[' + str(i) + ']' + '文件路径：'+ file_path
            fileData = file_module(file_path)
            if fileData['image_path'] == '':
                fileImageData = fileImage_module(fileData['image_path'])
            else:
                fileImageData = {'text':[], 'sex': None}
            file_sex = '未检测' if fileImageData['sex'] is None else '存在色情因素' if fileImageData['sex'] else '不存在色情因素' 
            result_text += '\n文件色情检测：' + file_sex
            yield result_text, risk_count(count)
            fileImageText = ''.join(fileImageData['text'])
            text = text_module(file_mark(fileData['text'], fileImageText))
            flag = False
            text1 = ''
            for t in text:
                if not flag:
                    result_text += '\n文件风险检测：\n' + t.replace(' ','')
                    flag = True
                else:
                    result_text += t.replace(' ', '')
                text1 += t
                yield result_text, risk_count(count)
            result_text += divide_line()
            text1 = text1.split('2')[0].split('：')[-1]
            if '是' in text1 or file_sex == '存在色情因素':
                count[5] += 1
            yield result_text, risk_count(count)


        elif 'www' in pre or 'http' in pre:
            web_url = path
            result_text += '\n[' + str(i) + ']' + '网址：'+ web_url
            webData = web_module(web_url)
            if webData['image_paths']:
                webImage_data = webImage_module(webData['image_paths'])
                webImage_data = '未检测' if webImage_data is None else '存在色情信息' if webImage_data else '不存在色情信息'
            else:
                webImage_data = '未检测'
            result_text += '\n网页色情检测：' + webImage_data
            yield result_text, risk_count(count)
            text = text_module(web_mark(webData['text']))
            flag = False
            text1 = ''
            for t in text:
                if not flag:
                    result_text += '\n网页风险检测：\n' + t.replace(' ','')
                    flag = True
                else:
                    result_text += t.replace(' ', '')
                text1 += t
                yield result_text, risk_count(count)
            result_text += divide_line()
            text1 = text1.split('2')[0].split('：')[-1]
            if '是' in text1 or webImage_data == '存在色情信息':
                count[4] += 1
            yield result_text, risk_count(count)


        else:
            text = path
            result_text += '\n[' + str(i) + ']' + '文本：'+ text
            text = text_module(text_mark(text))
            flag = False
            text1 = ''
            for t in text:
                if not flag:
                    result_text += '\n文本风险检测：\n' + t.replace(' ','')
                    flag = True
                else:
                    result_text += t.replace(' ', '')
                text1 += t
                yield result_text, risk_count(count)
            result_text += divide_line()
            text1 = text1.split('2')[0].split('：')[-1]
            if '是' in text1:
                count[3] += 1
            yield result_text, risk_count(count)
    # insert(text_structure(text), (result_text1+result_text2).replace(' ',''))
    # insert(web_structure(webData['text']), result_text.replace(' ',''))
    # insert(file_structure(fileData['text'], fileImageText), result_text.replace(' ', ''))
    # insert(audio_structure(audio_text), result_text.replace(' ', ''))
    # insert(image_structure(imageData['ocr_content'], imageData['image_content'], imageData['risk']), result_text.replace(' ', ''))
    # insert(video_structure(video_message['describe'], video_message['risk'] , audio_text), result_text)
   