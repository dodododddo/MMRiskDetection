import os
import re
import logging
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto
import LangSegment

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

i18n = I18nAuto()

if os.path.exists("./gweight.txt"):
    with open("./gweight.txt", 'r', encoding="utf-8") as file:
        gweight_data = file.read()
        gpt_path = os.environ.get("gpt_path", gweight_data)
else:
    gpt_path = os.environ.get("gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

if os.path.exists("./sweight.txt"):
    with open("./sweight.txt", 'r', encoding="utf-8") as file:
        sweight_data = file.read()
        sovits_path = os.environ.get("sovits_path", sweight_data)
else:
    sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")

cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
punctuation = set(['!', '?', '…', ',', '.', '-'," "])

cnhubert.cnhubert_base_path = cnhubert_base_path

ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)

change_sovits_weights(sovits_path)

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: 
        f.write(gpt_path)

change_gpt_weights(gpt_path)

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)

    return bert

def get_phones_and_bert(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "auto"}:
        textlist = []
        langlist = []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones, bert.to(dtype), norm_text

def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        prompt_phones, prompt_bert, prompt_norm_text = get_phones_and_bert(prompt_text, prompt_language)
        t1 = ttime()
        print("文本处理完成", t1 - t0)
        prompt_phones = torch.LongTensor(prompt_phones).to(device)
        prompt_phones_lengths = torch.LongTensor([prompt_phones.size(0)]).to(device)
        prompt_bert = prompt_bert.unsqueeze(0)
        ssl_unit = cnhubert.get_units(ref_wav_path, ssl_model)
        ssl_unit = torch.FloatTensor(ssl_unit).to(device)
        if ssl_unit.size(0) >= max_sec * hz:
            ssl_unit = ssl_unit[: int(max_sec * hz), :]
        elif ssl_unit.size(0) < max_sec * hz:
            ssl_unit = torch.cat((ssl_unit, torch.zeros(int(max_sec * hz) - ssl_unit.size(0), ssl_unit.size(1)).to(device)))
        ssl_unit = ssl_unit.to(device)
        print("声音处理完成")
    t1 = ttime()
    text = text.strip("\n")
    if (text[-1] not in splits): text += "。" if text_language != "en" else "."
    text = replace_consecutive_punctuation(text)
    text_phones, text_bert, text_norm_text = get_phones_and_bert(text, text_language)
    text_phones = torch.LongTensor(text_phones).to(device)
    text_phones_lengths = torch.LongTensor([text_phones.size(0)]).to(device)
    text_bert = text_bert.unsqueeze(0)
    t2 = ttime()
    print("文本处理完成", t2 - t1)
    if not ref_free:
        with torch.no_grad():
            try:
                pred_y, attn = t2s_model.net_g.infer(ref_mel=None, ref_bert=prompt_bert, ref_phones=prompt_phones, ref_phones_lengths=prompt_phones_lengths, tgt_mel=None, tgt_bert=text_bert, tgt_phones=text_phones, tgt_phones_lengths=text_phones_lengths, ssl_unit=ssl_unit.unsqueeze(0), vq_model=vq_model, max_len=None, top_k=top_k, top_p=top_p, temperature=temperature, ref_free=ref_free)
            except:
                return None, i18n("生成失败")
    else:
        with torch.no_grad():
            try:
                pred_y, attn = t2s_model.net_g.infer(ref_mel=None, ref_bert=None, ref_phones=None, ref_phones_lengths=None, tgt_mel=None, tgt_bert=text_bert, tgt_phones=text_phones, tgt_phones_lengths=text_phones_lengths, ssl_unit=None, vq_model=vq_model, max_len=None, top_k=top_k, top_p=top_p, temperature=temperature, ref_free=ref_free)
            except:
                return None, i18n("生成失败")
    t3 = ttime()
    print("生成完成", t3 - t2)
    pred_y = pred_y.squeeze().float().cpu().numpy()
    return pred_y, None

if __name__ == "__main__":
    from tools.wavfile import write as write_wav
    import argparse

    parser = argparse.ArgumentParser(description="Generate TTS wav file")
    parser.add_argument("ref_wav_path", type=str, help="Path to reference wav file")
    parser.add_argument("prompt_text", type=str, help="Prompt text")
    parser.add_argument("prompt_language", type=str, help="Prompt language")
    parser.add_argument("text", type=str, help="Text to generate speech for")
    parser.add_argument("text_language", type=str, help="Text language")
    parser.add_argument("--output_path", type=str, default="output.wav", help="Output wav file path")
    args = parser.parse_args()

    ref_wav_path = args.ref_wav_path
    prompt_text = args.prompt_text
    prompt_language = args.prompt_language
    text = args.text
    text_language = args.text_language
    output_path = args.output_path

    wav, err = get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language)
    if err is not None:
        print(err)
    else:
        write_wav(output_path, 22050, wav)
        print(f"Wav file saved to {output_path}")
