from utils import *
from transformers import AutoProcessor, AutoModelForPreTraining
import torch
from cnocr import CnOcr
import requests
from io import BytesIO
import opennsfw2 as n2
import concurrent.futures
import os

class ImageData():
    def __init__(self, synthesis:bool, fakeface:bool, have_characters:bool, ocr_content:str, image_content:str, risk:str, sex:bool):
        '''
        synthesis: true 表示合成图片
        fakeface: true 表示换脸图片
        have_characters: true 表示图片中有文字
        sex: true 表示图中包含色情
        '''
        self.synthesis = synthesis
        self.fakeface = fakeface
        self.have_characters = have_characters
        self.ocr_content = ocr_content
        self.image_content = image_content
        self.risk = risk
        self.sex = sex
    def __str__(self):
        return str(self.__dict__)

class ImagePipeline:
    def __init__(self, arch='CLIP:ViT-L/14', risk_url="http://127.0.0.1:1927/image", 
                 face_detect_model_weight="model/UniversalFakeDetect/checkpoints/clip_vitl14/ffhq_dffd_best.pth", 
                 synthesis_detect_model_weight="model/UniversalFakeDetect/pretrained_weights/fc_weights.pth"):
        self.arch = arch
        self.face_detect_model = detect_model(arch, face_detect_model_weight)
        self.synthesis_detect_model = detect_model(arch, synthesis_detect_model_weight)
        self.ocr_model = CnOcr()
        self.risk_url = risk_url 

    def _dirprocessing(self, img_path:str):
        if os.path.isdir(img_path):
            img_path = recursively_read(img_path)
        return img_path

    def _preprocessing(self, img_path:str|list):
        img_path = self._dirprocessing(img_path)
        if isinstance(img_path, str):
            if img_path.startswith(('http://', 'https://')):
                img_path = BytesIO(requests.get(img_path).content)
        elif isinstance(img_path, list):
            img_path = [BytesIO(requests.get(x).content) for x in img_path if x.startswith(('http://', 'https://'))]
        else:
            raise SyntaxError('不支持该类型')
        return img_path
    
    def multi_preprocessing(self, img_path:str|list):
        img_path = self._dirprocessing(img_path)
        max_workers = 128
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._preprocessing, x) for x in img_path]
            img_path = [future.result() for future in concurrent.futures.as_completed(futures)]
        print('预处理')
        return img_path

    def _detect_synthesis(self, img_path:str|list):
        '''检测图片是否是合成图片'''
        return predict(img_path, self.synthesis_detect_model, self.arch)
    
    def _detect_fake_face(self, img_path:str|list):
        '''检测是否人脸是否是生成的'''
        image_paths = have_face(img_path)
        if image_paths == []:
            return False
        return predict(image_paths, self.face_detect_model, self.arch)
    
    def detect_fake_face(self, img_path:list):
        return self._detect_fake_face(img_path)
    
    def _detect_characters_ocr(self, img_path, save_image=False, result_path=None):
        '''输出 是否有文字 以及 ocr内容'''
        return ocr(img_path, self.ocr_model, save_image, result_path)
    
    def _batch_ocr(self, img_path):
        max_workers = 128
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._detect_characters_ocr, x, self.ocr_model) for x in img_path]
            ocr_contents = [future.result()[1] for future in concurrent.futures.as_completed(futures)]
        return ocr_contents
    
    def detect_characters_ocr(self, img_path):
        return self._batch_ocr(img_path)
    
    def _llava_requests(self, img_path, prompt):
        try:
            resp = requests.post(url=self.risk_url, json={'image_path': img_path,
                                                      'prompt': prompt})
            if resp.status_code == 200:
                return resp.json()['text']
            else:
                return ''
        except:
            return ''
    
    def _describe_image_content(self, img_path):
        if isinstance(img_path, BytesIO):
            return ''
        prompt = "[INST] <image>\nPlease describe the content of the image.[/INST]"
        return self._llava_requests(img_path, prompt)
    
    def _detect_risk(self, img_path):
        if isinstance(img_path, BytesIO):
            return ''
        prompt = "[INST] <image>\nDoes this image contain elements of violence, pornography, or crime? Please pay extra attention to individuals exhibiting unusual behavior. If any of the above elements are present, please only output the corresponding element and describe the associated behavior. Do not provide any additional information.[/INST]"
        return self._llava_requests(img_path, prompt)
    
    def _detect_sex(self, img_path:str|list):
        if isinstance(img_path, list):
            batch_size = 128
            nsfw_probability = n2.predict_images(img_path, batch_size=batch_size)
            print(nsfw_probability)
            return [x > 0.5 for x in nsfw_probability]
        elif isinstance(img_path, BytesIO):
            nsfw_probability = n2.predict_image(img_path)
            return nsfw_probability > 0.5
        elif isinstance(img_path, str):
            nsfw_probability = n2.predict_image(img_path)
            return nsfw_probability > 0.5
        else:
           raise SyntaxError('不支持该类型')

    def detect_sex(self, img_path:str|list):
        sex_list = self._detect_sex(img_path)
        if True in sex_list:
            return True
        else:
            return False
        
    def __call__(self, img_path:str):
        img_path = self._preprocessing(img_path)
        synthesis = self._detect_synthesis(img_path)
        fakeface = self._detect_fake_face(img_path)
        have_characters, ocr_content = self._detect_characters_ocr(img_path)
        image_content = self._describe_image_content(img_path)
        risk = self._detect_risk(img_path)
        sex = self._detect_sex(img_path)
        return ImageData(synthesis, fakeface, have_characters, ocr_content, image_content, risk, sex)




        


    