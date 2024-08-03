# -*- coding: utf-8 -*-
import torch
import time
from faster_whisper import WhisperModel
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "model/Whipser_large-v3"
model = WhisperModel('./model/Whisper_large-v3', device='cuda')

def result(path):
    segments, info = model.transcribe(path)
    for segment in segments:
        print(segment.text)


if __name__ == "__main__":
    path = input("Please input the path:")
    result(path)

