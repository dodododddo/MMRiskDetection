import torch
import time
from faster_whisper import WhisperModel
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def extract_text(result):
    text = ''
    for segment in result['segments']:
        text = text + 'start:' + str(segment['start']) + '  '+ 'end:' + str(segment['end']) +'  '+ 'text:' + segment['text']+ '\n'
    return text

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "model/Whisper_large-v3"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model = whisperx.load_model("large", device='cuda',download_root='model')#model = WhisperModel("large-v3", device= 'cuda',download_root='model')
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]``

# result = pipe(sample)
# print(result["text"])

paths = ['./data/Audio/Audio/SSB00050353.wav', './data/Audio/Audio/SSB00090138.wav', './data/Audio/Audio/SSB00090176.wav',
         './data/Audio/Audio/SSB00090221.wav', './data/Audio/Audio/SSB00090369.wav', './data/Audio/Audio/SSB00090030.wav',
         './data/Audio/Audio/SSB00090164.wav', './data/Audio/Audio/SSB00090179.wav', './data/Audio/Audio/SSB00090229.wav',
         './data/Audio/Audio/SSB00090400.wav', './data/乐团-李荣浩.128.mp3']
# path = './data/Audio/Audio/SSB00050353.wav'

model = WhisperModel('./model/Whisper_large-v3', device='cuda')

for i in range(11):
        #model = WhisperModel("large-v3", device= 'cuda',download_root='model')
    path = paths[i]
    start_time = time.time()  # 开始计时
    segments, info = model.transcribe(path)
    end_time = time.time()  # 结束计时
    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    #     print(f"Time taken for {path}: {end_time - start_time:.2f} seconds\n")  # 打印处理每个音频文件所花的时间
    for segment in segments:
        print(segment.text)


