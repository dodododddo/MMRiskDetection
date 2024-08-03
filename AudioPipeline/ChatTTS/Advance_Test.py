# -*- coding: gbk -*-
import ChatTTS
import torch
import torchaudio


chat = ChatTTS.Chat()
chat.load(compile=False)  


texts = ["�������ȨҪ���У��¼����޵��Ƕ�ô��į"]


wavs = chat.infer(texts)


wav_tensor = torch.from_numpy(wavs[0])


if wav_tensor.ndim == 1:
    wav_tensor = wav_tensor.unsqueeze(0) 


torchaudio.save("output1.wav", wav_tensor, 24000)


rand_spk = chat.sample_random_speaker()
print("���˵����:", rand_spk)  # �����Ա�����


params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,
    temperature=0.3,
    top_P=0.7,
    top_K=20,
)


params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)


wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)


torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
