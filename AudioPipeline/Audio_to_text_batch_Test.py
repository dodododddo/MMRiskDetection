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

file_path = "./Test.txt"

for root, dirs, files in os.walk('/data1/home/jrchen/MMRiskDetection/AudioPipeline/Chinese_Corpus_Test_Real'):
    for file_name in files:
        print(file_name)
        segments, info = model.transcribe(f'/data1/home/jrchen/MMRiskDetection/AudioPipeline/Chinese_Corpus_Test_Real/{file_name}')
        transcript = ' '.join(segment.text for segment in segments)
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(f"{file_name}:{transcript}\n")


