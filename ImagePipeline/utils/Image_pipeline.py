from ImagePipeline.utils import *
from transformers import AutoProcessor, AutoModelForPreTraining
import torch
from cnocr import CnOcr

class ImageData:
    def __init__(self, synthesis:bool, haveface:bool, fakeface:bool, have_characters:bool, ocr_content:str, image_content:str, risk:str):
        '''
        synthesis: true 表示合成图片
        haveface: true 表示图片中含有人脸
        fakeface: true 表示换脸图片
        have_characters: true 表示图片中有文字
        '''
        self.synthesis = synthesis
        self.haveface = haveface
        self.fakeface = fakeface
        self.have_characters = have_characters
        self.ocr_content = ocr_content
        self.image_content = image_content
        self.risk = risk

    def __str__(self):
        return str(self.__dict__)

class ImagePipeline:
    def __init__(self, arch='CLIP:ViT-L/14', risk_model_path = "VideoPipeline/model/llava-v1.6-mistral-7b-hf", 
                 face_detect_model_weight="ImagePipeline/model/UniversalFakeDetect/checkpoints/clip_vitl14/ffhq_dffd_best.pth", 
                 synthesis_detect_model_weight="ImagePipeline/model/UniversalFakeDetect/pretrained_weights/fc_weights.pth"):
        self.arch = arch
        self.face_detect_model = detect_model(arch, face_detect_model_weight)
        self.synthesis_detect_model = detect_model(arch, synthesis_detect_model_weight)
        self.ocr_model = CnOcr()
        processor = AutoProcessor.from_pretrained(risk_model_path, device_map="auto")
        risk_model = AutoModelForPreTraining.from_pretrained(risk_model_path,   
                                                     torch_dtype=torch.float16, 
                                                     device_map="auto")
        self.risk_pipe = risk_pineline(risk_model, processor)

    def detect_synthesis(self, img_path):
        '''检测图片是否是合成图片'''
        return predict(img_path, self.synthesis_detect_model, self.arch)
    
    def detect_face(self, img_path):
        '''检测图片是否有人脸'''
        return have_face(img_path)
    
    def detect_fake_face(self, img_path):
        '''检测是否人脸是否是生成的'''
        return predict(img_path, self.face_detect_model, self.arch)
    
    def detect_characters_ocr(self, img_path, save_image=False, result_path=None):
        '''输出 是否有文字 以及 ocr内容'''
        return ocr(img_path, self.ocr_model, save_image, result_path)
    
    def describe_image_content(self, img_path):
        prompt = "[INST] <image>\nPlease describe the content of the image.[/INST]"
        return self.risk_pipe(img_path, prompt)
    
    def detect_risk(self, img_path):
        prompt = "[INST] <image>\nDoes this image contain elements of violence, pornography, or crime? Please pay extra attention to individuals exhibiting unusual behavior. If any of the above elements are present, please only output the corresponding element and describe the associated behavior. Do not provide any additional information.[/INST]"
        return self.risk_pipe(img_path, prompt)
    
    def __call__(self, img_path):
        synthesis = self.detect_synthesis(img_path)
        haveface = self.detect_face(img_path)
        fakeface = self.detect_fake_face(img_path)
        have_characters, ocr_content = self.detect_characters_ocr(img_path)
        print(ocr_content)
        image_content = self.describe_image_content(img_path)
        risk = self.detect_risk(img_path)
        return ImageData(synthesis, haveface, fakeface, have_characters, ocr_content, image_content, risk)
    
    


        


    