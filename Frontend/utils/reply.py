import requests

def check(status_code):
    if status_code != 200:
        return False
    else:
        return True

def text_reply(message, history=None):
    try:
        resp = requests.post(url = "http://127.0.0.1:1111/generate", json={'message': message}, stream=True)

        if check(resp.status_code):
            # for t in resp.iter_lines():
            #     print(t.decode('utf-8'))
            return resp.iter_content(decode_unicode=True)
        else:
            return ''
    except:
        return ''

def image_reply(image_path):
    try:
        print(image_path)
        resp = requests.post(url = "http://127.0.0.1:6666/image", json={'image_path': image_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'synthesis':None, 'fakeface': None, 'have_characters':None, 'ocr_content':'', 'image_content':'', 'risk':'', 'sex': None}
    except:
        return {'synthesis':None, 'fakeface': None, 'have_characters':None, 'ocr_content':'', 'image_content':'', 'risk':'', 'sex': None}

def webImage_reply(image_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:6666/webImage", json={'image_path': image_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'sex':None}
    except:
        return {'sex':None}

def web_reply(web_url):
    try:
        resp = requests.post(url = "http://127.0.0.1:6667/web", json={'web_url': web_url})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'text':'','image_paths':[]}
    except:
        return {'text':'','image_paths':[]}

def audio_to_text_reply(audio_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:9999/audio", json={'audio_path': audio_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'text':''}
    except:
        return {'text':''}

def audio_detect_reply(audio_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:9998/audio_detection", json={'audio_for_analyse_path': audio_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'fake_or_not':None}
    except:
        return {'fake_or_not':None}

def file_reply(file_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:6670/file", json={'file_path': file_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'text':'','image_path':''}
    except:
        return {'text':'','image_path':''}

def fileImage_reply(image_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:6666/fileImage", json={'image_path': image_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'text':[], 'sex': None}
    except:
        return {'text':[], 'sex': None} 

def video_message_reply(video_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:1927/video", json={'video_path': video_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'describe':'','risk':'','deepfake_detection':None}
    except:
        return {'describe':'','risk':'','deepfake_detection':None}

def video_extract_reply(video_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:1928/video_extract", json={'video_path': video_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'image':'', 'audio':'no_audio'}
    except:
        return {'image':'', 'audio':'no_audio'}

def videoImage_reply(image_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:6666/videoImage", json={'image_path': image_path})
        if check(resp.status_code):
                return resp.json()
        else:
                return {'fake_face':None, 'sex':None}
    except:
        return {'fake_face':None, 'sex':None}

def face_restoration_reply(path, facial, image, video):
    resp = requests.post(url = "http://127.0.0.1:9997/Digital_face1", json={'ImageorVideo_for_restored_path': path, 
                                                                            'function_Facial_rejuvenation': facial, 
                                                                            'function_image_enhancement': image, 
                                                                            'function_video_enhancement': video}).json()
    return resp

def facefusion_reply(image_path, video_path):
    resp = requests.post(url = "http://127.0.0.1:1929/facefusion", json={'source_path': image_path, 
                                                                         'target_path': video_path}).json()
    return resp

def ssgen_reply(ref_wav_path, prompt_text, text):
    resp = requests.post(url = "http://127.0.0.1:9995/SSGen", json={'ref_wav_path': ref_wav_path,
                                                                    'prompt_text': prompt_text, 
                                                                    'text': text}).json()
    return resp

def video_generator_reply(Image_path, Audio_path, function_A, function_B, function_C):
    resp = requests.post(url = "http://127.0.0.1:9996/Digital_video", json={'Image_path': Image_path, 
                                                                            'Audio_path': Audio_path, 
                                                                            'function_A': function_A, 
                                                                            'function_B': function_B, 
                                                                            'function_C': function_C}).json()
    return resp

def ext_reply(text):
    try:
        resp = requests.post(url="http://127.0.0.1:1930/ext", json={'text':text})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'ext_text':'未识别'}
    except:
        return {'ext_text':'未识别'}

def video_fakeface_reply(video_path):
    try:
        resp = requests.post(url="http://127.0.0.1:1932/videofakeface", json={'video_path': video_path})
        if check(resp.status_code):
            return resp.json()
        else:
            return {'fake':None}
    except:
        return {'fake':None}
 


      



