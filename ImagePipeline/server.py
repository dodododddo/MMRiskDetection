from fastapi import FastAPI
from pydantic import BaseModel
from utils import ImagePipeline
import uvicorn

class InputData(BaseModel):
    image_path: str

class InputListData(BaseModel):
    image_path: list
    
class OutputImageData(BaseModel):
    synthesis: bool
    fakeface: bool
    have_characters: bool
    ocr_content: str
    image_content: str
    risk: str
    sex: bool

class OutputVideoData(BaseModel):
    fake_face: bool
    sex: bool

class OutputWebData(BaseModel):
    sex: bool

class OutputFileData(BaseModel):
    text: list
    sex: bool

if __name__ == '__main__':
    imagePipeline = ImagePipeline()
    app = FastAPI()
    @app.post('/image', response_model=OutputImageData)
    def reply(data: InputData):
        imageData = imagePipeline(data.image_path)
        return OutputImageData(**imageData.__dict__)
    
    @app.post('/webImage', response_model=OutputWebData)
    def reply(data: InputListData):
        img_path = imagePipeline.multi_preprocessing(data.image_path)
        imageData = imagePipeline.detect_sex(img_path)
        return OutputWebData(sex=imageData)
    
    @app.post('/videoImage', response_model=OutputVideoData)
    def reply(data: InputData):
        img_path = imagePipeline.multi_preprocessing(data.image_path)
        fake_face = imagePipeline.detect_fake_face(img_path)
        sex = imagePipeline.detect_sex(img_path)
        return OutputVideoData(fake_face=fake_face, sex=sex)
    
    @app.post('/fileImage', response_model=OutputFileData)
    def reply(data: InputData):
        img_path = imagePipeline.multi_preprocessing(data.image_path)
        text = imagePipeline.detect_characters_ocr(img_path)
        sex = imagePipeline.detect_sex(img_path)
        return OutputFileData(text=text, sex=sex)

    uvicorn.run(app, host="127.0.0.1", port=6666)


