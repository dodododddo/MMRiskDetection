from transformers import AutoProcessor, AutoModelForPreTraining
from utils import risk_pineline, recursively_read
import torch

risk_model_path = "../VideoPipeline/model/llava-v1.6-mistral-7b-hf"
processor = AutoProcessor.from_pretrained(risk_model_path)
risk_model = AutoModelForPreTraining.from_pretrained(risk_model_path, 
                                                     torch_dtype=torch.float16, 
                                                     device_map="auto")

img_paths = recursively_read('data/risk')

prompt = "[INST] <image>\n请用中文描述一下图片的内容[/INST]"
# prompt = "[INST] <image>\n这张图片中是否包含暴力、色情、犯罪因素？请多关注有异常行为的人。假如有以上因素，请只输出对应因素和对应因素的行为是什么。不要有多于输出。 [/INST]"
risk_pipe = risk_pineline(img_paths, risk_model, processor)
risk_pipe(prompt)
