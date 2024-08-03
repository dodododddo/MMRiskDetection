# -*- coding: gbk -*-
import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False)  

texts = ["刘沛权要就有"]

wavs = chat.infer(texts)

for i in range(len(texts)):
    wav_tensor = torch.from_numpy(wavs[i])
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0) 

    torchaudio.save(f"output{i}.wav", wav_tensor, 24000)
