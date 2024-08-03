from fastapi import FastAPI
from pydantic import BaseModel
from facefusion_pipeline import facefusion_pipeline
import uvicorn
import os

class InputData(BaseModel):
    source_path: str
    target_path: str
    
class OutputData(BaseModel):
    output_file_path: str

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    app = FastAPI()
    @app.post('/facefusion', response_model=OutputData)
    def reply(data: InputData):
        print(data.source_path)
        print(data.target_path)
        result = facefusion_pipeline(data.source_path, data.target_path)
        return OutputData(output_file_path=result)

    uvicorn.run(app, host="127.0.0.1", port=1929)