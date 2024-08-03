from fastapi import FastAPI
from pydantic import BaseModel
from utils import *
import uvicorn

class InputData(BaseModel):
    file_path: str
    
class OutputData(BaseModel):
    text: str
    image_path:str
    
if __name__ == '__main__':
    app = FastAPI()
    @app.post('/file', response_model=OutputData)
    def reply(data: InputData):
        print(data.file_path)
        file_path = data.file_path
        if file_path.lower().endswith('.pdf'):
            text = get_pdf_text(file_path)
            image_path = get_pdf_image(file_path)
        elif file_path.lower().endswith('.docx'):
            text = get_word_text(file_path)
            image_path = get_word_image(file_path)
        else:
            raise SyntaxError('暂不支持该格式')
        return OutputData(text=text, image_path=image_path)

    uvicorn.run(app, host="127.0.0.1", port=6670)





