import torch
import time
import os
from faster_whisper import WhisperModel
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "model/Whisper_large-v3"

model = WhisperModel('./model/Whisper_large-v3', device='cuda')

def process(input_path):
    output_text = ""
    for root, dirs, files in os.walk(input_path):
        for file in files:
            print(file)
            segments, info = model.transcribe(f'./{input_path}/{file}')
            for segment in segments:
                output_text += segment.text + '\n'
                print(segment.text)
    return output_text



