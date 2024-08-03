from fastapi import FastAPI
from pydantic import BaseModel
from utils import WebPipeline
import uvicorn

class InputData(BaseModel):
    web_url: str
    
class OutputData(BaseModel):
    text: str
    image_paths: list

if __name__ == '__main__':
    webPipeline = WebPipeline()
    app = FastAPI()
    @app.post('/web', response_model=OutputData)
    def reply(data: InputData):
        print(data.web_url)
        webData = webPipeline(data.web_url)
        return OutputData(**webData.__dict__)

    uvicorn.run(app, host="127.0.0.1", port=6667)


