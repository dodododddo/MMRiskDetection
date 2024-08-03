import torch
import time
import os
from faster_whisper import WhisperModel
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class To_Text:
    def __init__(self):
        self.device = 'cuda'
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_path = os.path.abspath("./model/Whisper_large-v3")
        print(f"Loading model from {self.model_path}")
        self.model = WhisperModel(self.model_path, self.device)

    def process(self, input_path):
        output_text = ""
        segments, info = self.model.transcribe(input_path)
        for segment in segments:
            print(segment)
            output_text += segment.text
        return output_text
