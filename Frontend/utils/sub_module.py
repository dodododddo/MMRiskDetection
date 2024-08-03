from reply import *
import re

def web_module(web_url):
    webData = web_reply(web_url)
    return webData

def webImage_module(image_paths):
    webImageData = webImage_reply(image_paths)
    return webImageData

def text_module(text):
    # print(text)
    textData = text_reply(text)
    result = ''
    for data in textData:
        yield data
        # yield re.sub(r'(\d+\.)', r'\n\1', data)
    #     result += data.decode('utf-8')
    # result = re.sub(r'(\d+\.)', r'\n\1', result)
    # result = result.lstrip()
    # result = '未检测' if result == '' else result 
    # return result

def image_module(image_paths):
    imageData = image_reply(image_paths)
    return imageData

def audio_to_text_module(audio_paths):
    audioData = audio_to_text_reply(audio_paths)
    return audioData['text']

def audio_detect_module(audio_paths):
    audioData = audio_detect_reply(audio_paths)
    return audioData

def file_module(file_path):
    fileData = file_reply(file_path)
    return fileData

def fileImage_module(image_paths):
    fileImageData = fileImage_reply(image_paths)
    return fileImageData

def video_message_module(flie_path):
    video_message = video_message_reply(flie_path)
    return video_message

def video_extract_module(flie_path):
    video_extract = video_extract_reply(flie_path)
    return video_extract

def videoImage_module(image_path):
    videoImageData = videoImage_reply(image_path)
    return videoImageData

def face_restoration_module(path:str, facial, image, video):
    output_path  = face_restoration_reply(path, facial, image, video)
    print(path, image)
    if image == True:
        output_path = output_path['Digital_face1_path']  + '/restored_faces/' + path.split('/')[-1] + '.jpg'
    elif video == True:
        output_path = './Frontend/demo/Video_gen.mp4'
    print(output_path)
    return output_path

def facefusion_module(image_path, video_path):
    output_path = facefusion_reply(image_path, video_path)
    return output_path['output_file_path']

def ssgen_module(ref_wav_path, prompt_text, text):
    output_path = ssgen_reply(ref_wav_path, prompt_text, text)
    return output_path['output_sound_path']

def video_generator_module(Image_path, Audio_path, function_A, function_B, function_C):
    output_path = video_generator_reply(Image_path, Audio_path, function_A, function_B, function_C)
    return output_path['Video_path']

def ext_module(text):
    ext_text = ext_reply(text)
    return ext_text

def video_fakeface_module(video_path):
    video_fakeface_data = video_fakeface_reply(video_path)
    return video_fakeface_data['fake']

def sms_text_module(text):
    textData = text_reply(text)
    result = ''
    for data in textData:
        result += data
    result = '未检测' if result == '' else result 
    return result.replace(' ', '')

