import ChatTTS
import torch
import torchaudio
import os

chat = ChatTTS.Chat()
chat.load(compile=False)


file_path = "../result.txt"
output_dir = "output_wavs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


with open(file_path, "r", encoding="utf-8") as file:
    line_number = 0
    while True:
        line = file.readline()
        if not line:
            break  
        text = line.strip()
        if text:
            wavs = chat.infer([text])
            wav_tensor = torch.from_numpy(wavs[0])
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            
            output_path = os.path.join(output_dir, f"output{line_number}.wav")
            torchaudio.save(output_path, wav_tensor, 24000)
            print(f"Haved save to:{output_path}")
            line_number += 1

