# import os
# import re
# import logging
# import LangSement
# import torch
# import librosa
# import numpy as np
# import soundfile as sf
# from transformers import AutoModelForMaskedLM, AutoTokenizer
# from feature_extractor import cnhubert
# from module.models import SynthesizerTrn
# from AR.models.t2s_lightning_module import Text2SemanticLightningModule
# from text import cleaned_text_to_sequence
# from text.cleaner import clean_text
# from tools.my_utils import load_audio
# from tools.i18n.i18n import I18nAuto
# from module.mel_processing import spectrogram_torch
# import argparse

# logging.getLogger("markdown_it").setLevel(logging.ERROR)
# logging.getLogger("urllib3").setLevel(logging.ERROR)
# logging.getLogger("httpcore").setLevel(logging.ERROR)
# logging.getLogger("httpx").setLevel(logging.ERROR)
# logging.getLogger("asyncio").setLevel(logging.ERROR)
# logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
# logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

# class GPTSoVITS:
#     def __init__(self):
#         self.i18n = I18nAuto()
#         self._setup_environment_variables()
#         self._initialize_models()
#         self._set_language_dict()
#         self._setup_punctuation()

#     def _setup_environment_variables(self):
#         self.gpt_path = os.environ.get("gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
#         self.sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
#         self.cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
#         self.bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
#         self.infer_ttswebui = int(os.environ.get("infer_ttswebui", 9872))
#         self.is_share = eval(os.environ.get("is_share", "False"))
#         self.is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
        
#     def _initialize_models(self):
#         cnhubert.cnhubert_base_path = self.cnhubert_base_path
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
#         self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
#         if self.is_half:
#             self.bert_model = self.bert_model.half().to(self.device)
#         else:
#             self.bert_model = self.bert_model.to(self.device)

#         self.ssl_model = cnhubert.get_model()
#         if self.is_half:
#             self.ssl_model = self.ssl_model.half().to(self.device)
#         else:
#             self.ssl_model = self.ssl_model.to(self.device)

#         self._change_sovits_weights(self.sovits_path)
#         self._change_gpt_weights(self.gpt_path)

#     def _change_sovits_weights(self, sovits_path):
#         dict_s2 = torch.load(sovits_path, map_location="cpu")
#         self.hps = DictToAttrRecursive(dict_s2["config"])
#         self.hps.model.semantic_frame_rate = "25hz"
#         self.vq_model = SynthesizerTrn(
#             self.hps.data.filter_length // 2 + 1,
#             self.hps.train.segment_size // self.hps.data.hop_length,
#             n_speakers=self.hps.data.n_speakers,
#             **self.hps.model
#         )
#         if "pretrained" not in sovits_path:
#             del self.vq_model.enc_q
#         if self.is_half:
#             self.vq_model = self.vq_model.half().to(self.device)
#         else:
#             self.vq_model = self.vq_model.to(self.device)
#         self.vq_model.eval()
#         self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

#     def _change_gpt_weights(self, gpt_path):
#         self.hz = 50
#         dict_s1 = torch.load(gpt_path, map_location="cpu")
#         self.config = dict_s1["config"]
#         self.max_sec = self.config["data"]["max_sec"]
#         self.t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
#         self.t2s_model.load_state_dict(dict_s1["weight"])
#         if self.is_half:
#             self.t2s_model = self.t2s_model.half()
#         self.t2s_model = self.t2s_model.to(self.device)
#         self.t2s_model.eval()

#     def _set_language_dict(self):
#         self.dict_language = {
#             self.i18n("����"): "all_zh",  # ȫ��������ʶ��
#             self.i18n("Ӣ��"): "en",  # ȫ����Ӣ��ʶ��
#             self.i18n("����"): "all_ja",  # ȫ��������ʶ��
#             self.i18n("��Ӣ���"): "zh",  # ����Ӣ���ʶ��
#             self.i18n("��Ӣ���"): "ja",  # ����Ӣ���ʶ��
#             self.i18n("�����ֻ��"): "auto",  # �����������з�ʶ������
#         }

#     def _setup_punctuation(self):
#         self.punctuation = set(['!', '?', '��', ',', '.', '-'," "])
#         self.splits = {"��", "��", "��", "��", ",", ".", "?", "!", "~", ":", "��", "��", "��"}

#     def get_bert_feature(self, text, word2ph):
#         with torch.no_grad():
#             inputs = self.tokenizer(text, return_tensors="pt")
#             for i in inputs:
#                 inputs[i] = inputs[i].to(self.device)
#             res = self.bert_model(**inputs, output_hidden_states=True)
#             res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
#         assert len(word2ph) == len(text)
#         phone_level_feature = []
#         for i in range(len(word2ph)):
#             repeat_feature = res[i].repeat(word2ph[i], 1)
#             phone_level_feature.append(repeat_feature)
#         phone_level_feature = torch.cat(phone_level_feature, dim=0)
#         return phone_level_feature.T

#     def get_spepc(self, filename):
#         audio = load_audio(filename, int(self.hps.data.sampling_rate))
#         audio = torch.FloatTensor(audio)
#         audio_norm = audio.unsqueeze(0)
#         spec = spectrogram_torch(
#             audio_norm,
#             self.hps.data.filter_length,
#             self.hps.data.sampling_rate,
#             self.hps.data.hop_length,
#             self.hps.data.win_length,
#             center=False,
#         )
#         return spec

#     def clean_text_inf(self, text, language):
#         phones, word2ph, norm_text = clean_text(text, language)
#         phones = cleaned_text_to_sequence(phones)
#         return phones, word2ph, norm_text

#     def get_bert_inf(self, phones, word2ph, norm_text, language):
#         language = language.replace("all_", "")
#         if language == "zh":
#             bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
#         else:
#             bert = torch.zeros(
#                 (1024, len(phones)),
#                 dtype=torch.float16 if self.is_half else torch.float32,
#             ).to(self.device)
#         return bert

#     def get_phones_and_bert(self, text, language):
#         if language in {"en", "all_zh", "all_ja"}:
#             language = language.replace("all_", "")
#             if language == "en":
#                 LangSegment.setfilters(["en"])
#                 formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
#             else:
#                 formattext = text
#             while "  " in formattext:
#                 formattext = formattext.replace("  ", " ")
#             phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
#             if language == "zh":
#                 bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
#             else:
#                 bert = torch.zeros(
#                     (1024, len(phones)),
#                     dtype=torch.float16 if self.is_half else torch.float32,
#                 ).to(self.device)
#         elif language in {"zh", "ja", "auto"}:
#             textlist = []
#             langlist = []
#             LangSegment.setfilters(["zh", "ja", "en", "ko"])
#             if language == "auto":
#                 for tmp in LangSegment.getTexts(text):
#                     if tmp["lang"] == "ko":
#                         langlist.append("zh")
#                         textlist.append(tmp["text"])
#                     else:
#                         langlist.append(tmp["lang"])
#                         textlist.append(tmp["text"])
#             else:
#                 for tmp in LangSegment.getTexts(text):
#                     if tmp["lang"] == "en":
#                         langlist.append(tmp["lang"])
#                     else:
#                         langlist.append(language)
#                     textlist.append(tmp["text"])
#             phones_list = []
#             bert_list = []
#             norm_text_list = []
#             for i in range(len(textlist)):
#                 lang = langlist[i]
#                 phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
#                 bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
#                 phones_list.append(phones)
#                 norm_text_list.append(norm_text)
#                 bert_list.append(bert)
#             bert = torch.cat(bert_list, dim=1)
#             phones = sum(phones_list, [])
#             norm_text = ''.join(norm_text_list)
#         return phones, bert.to(torch.float16 if self.is_half else torch.float32), norm_text

#     def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="����", top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
#         if prompt_text is None or len(prompt_text) == 0:
#             ref_free = True
#         prompt_language = self.dict_language[prompt_language]
#         text_language = self.dict_language[text_language]
#         if not ref_free:
#             prompt_text = prompt_text.strip("\n")
#             if prompt_text[-1] not in self.splits:
#                 prompt_text += "��" if prompt_language != "en" else "."
#         text = text.strip("\n")
#         text = self.replace_consecutive_punctuation(text)
#         if text[0] not in self.splits and len(self.get_first(text)) < 4:
#             text = "��" + text if text_language != "en" else "."
#         zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3), dtype=np.float16 if self.is_half else np.float32)
#         if not ref_free:
#             with torch.no_grad():
#                 wav16k, sr = librosa.load(ref_wav_path, sr=16000)
#                 if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
#                     raise OSError("�ο���Ƶ��3~10�뷶Χ�⣬�������")
#                 wav16k = torch.from_numpy(wav16k)
#                 zero_wav_torch = torch.from_numpy(zero_wav)
#                 if self.is_half:
#                     wav16k = wav16k.half().to(self.device)
#                     zero_wav_torch = zero_wav_torch.half().to(self.device)
#                 else:
#                     wav16k = wav16k.to(self.device)
#                     zero_wav_torch = zero_wav_torch.to(self.device)
#                 wav16k = torch.cat([wav16k, zero_wav_torch])
#                 ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
#                 codes = self.vq_model.extract_latent(ssl_content)
#                 prompt_semantic = codes[0, 0]
#                 prompt = prompt_semantic.unsqueeze(0).to(self.device)

#         if how_to_cut == "���ľ�һ��":
#             text = self.cut1(text)
#         elif how_to_cut == "��50��һ��":
#             text = self.cut2(text)
#         elif how_to_cut == "�����ľ�š���":
#             text = self.cut3(text)
#         elif how_to_cut == "��Ӣ�ľ��.��":
#             text = self.cut4(text)
#         elif how_to_cut == "����������":
#             text = self.cut5(text)
#         while "\n\n" in text:
#             text = text.replace("\n\n", "\n")
#         texts = text.split("\n")
#         texts = self.process_text(texts)
#         texts = self.merge_short_text_in_array(texts, 5)
#         audio_opt = []
#         if not ref_free:
#             phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language)
#         for text in texts:
#             if len(text.strip()) == 0:
#                 continue
#             if text[-1] not in self.splits:
#                 text += "��" if text_language != "en" else "."
#             phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language)
#             if not ref_free:
#                 bert = torch.cat([bert1, bert2], 1)
#                 all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
#             else:
#                 bert = bert2
#                 all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
#             bert = bert.to(self.device).unsqueeze(0)
#             all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
#             with torch.no_grad():
#                 pred_semantic, idx = self.t2s_model.model.infer_panel(
#                     all_phoneme_ids,
#                     all_phoneme_len,
#                     None if ref_free else prompt,
#                     bert,
#                     top_k=top_k,
#                     top_p=top_p,
#                     temperature=temperature,
#                     early_stop_num=self.hz * self.max_sec,
#                 )
#             pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
#             refer = self.get_spepc(ref_wav_path)
#             if self.is_half:
#                 refer = refer.half().to(self.device)
#             else:
#                 refer = refer.to(self.device)
#             audio = self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refer).detach().cpu().numpy()[0, 0]
#             max_audio = np.abs(audio).max()
#             if max_audio > 1:
#                 audio /= max_audio
#             audio_opt.append(audio)
#             audio_opt.append(zero_wav)
#         return self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

#     def merge_short_text_in_array(self, texts, threshold):
#         if len(texts) < 2:
#             return texts
#         result = []
#         text = ""
#         for ele in texts:
#             text += ele
#             if len(text) >= threshold:
#                 result.append(text)
#                 text = ""
#         if len(text) > 0:
#             if len(result) == 0:
#                 result.append(text)
#             else:
#                 result[len(result) - 1] += text
#         return result

#     def split(self, todo_text):
#         todo_text = todo_text.replace("����", "��").replace("����", "��")
#         if todo_text[-1] not in self.splits:
#             todo_text += "��"
#         i_split_head = i_split_tail = 0
#         len_text = len(todo_text)
#         todo_texts = []
#         while True:
#             if i_split_head >= len_text:
#                 break
#             if todo_text[i_split_head] in self.splits:
#                 i_split_head += 1
#                 todo_texts.append(todo_text[i_split_tail:i_split_head])
#                 i_split_tail = i_split_head
#             else:
#                 i_split_head += 1
#         return todo_texts

#     def cut1(self, inp):
#         inp = inp.strip("\n")
#         inps = self.split(inp)
#         split_idx = list(range(0, len(inps), 4))
#         split_idx[-1] = None
#         if len(split_idx) > 1:
#             opts = []
#             for idx in range(len(split_idx) - 1):
#                 opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
#         else:
#             opts = [inp]
#         opts = [item for item in opts if not set(item).issubset(self.punctuation)]
#         return "\n".join(opts)

#     def cut2(self, inp):
#         inp = inp.strip("\n")
#         inps = self.split(inp)
#         if len(inps) < 2:
#             return inp
#         opts = []
#         summ = 0
#         tmp_str = ""
#         for i in range(len(inps)):
#             summ += len(inps[i])
#             tmp_str += inps[i]
#             if summ > 50:
#                 summ = 0
#                 opts.append(tmp_str)
#                 tmp_str = ""
#         if tmp_str != "":
#             opts.append(tmp_str)
#         if len(opts) > 1 and len(opts[-1]) < 50:
#             opts[-2] = opts[-2] + opts[-1]
#             opts = opts[:-1]
#         opts = [item for item in opts if not set(item).issubset(self.punctuation)]
#         return "\n".join(opts)

#     def cut3(self, inp):
#         inp = inp.strip("\n")
#         opts = ["%s" % item for item in inp.strip("��").split("��")]
#         opts = [item for item in opts if not set(item).issubset(self.punctuation)]
#         return "\n".join(opts)

#     def cut4(self, inp):
#         inp = inp.strip("\n")
#         opts = ["%s" % item for item in inp.strip(".").split(".")]
#         opts = [item for item in opts if not set(item).issubset(self.punctuation)]
#         return "\n".join(opts)

#     def cut5(self, inp):
#         inp = inp.strip("\n")
#         punds = {',', '.', ';', '?', '!', '��', '��', '��', '��', '��', ';', '��', '��'}
#         mergeitems = []
#         items = []
#         for i, char in enumerate(inp):
#             if char in punds:
#                 if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
#                     items.append(char)
#                 else:
#                     items.append(char)
#                     mergeitems.append("".join(items))
#                     items = []
#             else:
#                 items.append(char)
#         if items:
#             mergeitems.append("".join(items))
#         opt = [item for item in mergeitems if not set(item).issubset(punds)]
#         return "\n".join(opt)

#     def get_first(self, text):
#         pattern = "[" + "".join(re.escape(sep) for sep in self.splits) + "]"
#         text = re.split(pattern, text)[0].strip()
#         return text

#     def replace_consecutive_punctuation(self, text):
#         punctuations = ''.join(re.escape(p) for p in self.punctuation)
#         pattern = f'([{punctuations}])([{punctuations}])+'
#         result = re.sub(pattern, r'\1', text)
#         return result

#     def process_text(self, texts):
#         _text = []
#         if all(text in [None, " ", "\n", ""] for text in texts):
#             raise ValueError("��������Ч�ı�")
#         for text in texts:
#             if text in [None, " ", ""]:
#                 pass
#             else:
#                 _text.append(text)
#         return _text

#     def save_audio(self, audio, sampling_rate, output_path="output.wav"):
#         sf.write(output_path, audio.astype(np.float32) / 32768, sampling_rate)
#         print(f"Output saved to {output_path}")

#     @staticmethod
#     def parse_args():
#         parser = argparse.ArgumentParser(description="GPT-SoVITS CLI")
#         parser.add_argument("--ref_wav_path", type=str, required=True, help="Path to the reference audio file (3-10 seconds).")
#         parser.add_argument("--prompt_text", type=str, default="", help="Text corresponding to the reference audio.")
#         parser.add_argument("--prompt_language", type=str, default="����", choices=["����", "Ӣ��", "����", "��Ӣ���", "��Ӣ���", "�����ֻ��"], help="Language of the prompt.")
#         parser.add_argument("--text", type=str, required=True, help="Target text to be synthesized.")
#         parser.add_argument("--text_language", type=str, default="����", choices=["����", "Ӣ��", "����", "��Ӣ���", "��Ӣ���", "�����ֻ��"], help="Language of the target text.")
#         parser.add_argument("--how_to_cut", type=str, default="����", choices=["����", "���ľ�һ��", "��50��һ��", "�����ľ�š���", "��Ӣ�ľ��.��", "����������"], help="How to split the text.")
#         parser.add_argument("--top_k", type=int, default=5, help="Top-k sampling parameter for GPT.")
#         parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter for GPT.")
#         parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for GPT.")
#         parser.add_argument("--ref_free", action="store_true", help="Enable reference-free mode.")
#         return parser.parse_args()

#     @classmethod
#     def run_tts(cls, **kwargs):
#         args = cls.parse_args(**kwargs)
#         SSGen = cls(args)
#         sampling_rate, audio = SSGen.get_tts_wav()
#         output_path = "output.wav"
#         SSGen.save_audio(audio, sampling_rate, output_path)
#         return output_path

# class DictToAttrRecursive(dict):
#     def __init__(self, input_dict):
#         super().__init__(input_dict)
#         for key, value in input_dict.items():
#             if isinstance(value, dict):
#                 value = DictToAttrRecursive(value)
#             self[key] = value
#             setattr(self, key, value)

#     def __getattr__(self, item):
#         try:
#             return self[item]
#         except KeyError:
#             raise AttributeError(f"Attribute {item} not found")

#     def __setattr__(self, key, value):
#         if isinstance(value, dict):
#             value = DictToAttrRecursive(value)
#         super(DictToAttrRecursive, self).__setitem__(key, value)
#         super().__setattr__(key, value)

#     def __delattr__(self, item):
#         try:
#             del self[item]
#         except KeyError:
#             raise AttributeError(f"Attribute {item} not found")

import os
import re
import logging
import LangSegment
import torch
import librosa
import numpy as np
import soundfile as sf
from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto
from module.mel_processing import spectrogram_torch
import argparse

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

class GPTSoVITS:
    def __init__(self, args):
        self.args = args
        self.i18n = I18nAuto()
        self._setup_environment_variables()
        self._initialize_models()
        self._set_language_dict()
        self._setup_punctuation()

    def _setup_environment_variables(self):
        self.gpt_path = os.environ.get("gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
        self.sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
        self.cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
        self.bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        self.infer_ttswebui = int(os.environ.get("infer_ttswebui", 9872))
        self.is_share = eval(os.environ.get("is_share", "False"))
        self.is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
        
    def _initialize_models(self):
        cnhubert.cnhubert_base_path = self.cnhubert_base_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)

        self.ssl_model = cnhubert.get_model()
        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)

        self._change_sovits_weights(self.sovits_path)
        self._change_gpt_weights(self.gpt_path)

    def _change_sovits_weights(self, sovits_path):
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        self.hps = DictToAttrRecursive(dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        if "pretrained" not in sovits_path:
            del self.vq_model.enc_q
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

    def _change_gpt_weights(self, gpt_path):
        self.hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()

    def _set_language_dict(self):
        self.dict_language = {
            self.i18n("中文"): "all_zh",  # 全部按中文识别
            self.i18n("英文"): "en",  # 全部按英文识别
            self.i18n("日文"): "all_ja",  # 全部按日文识别
            self.i18n("中英混合"): "zh",  # 按中英混合识别
            self.i18n("日英混合"): "ja",  # 按日英混合识别
            self.i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
        }

    def _setup_punctuation(self):
        self.punctuation = set(['!', '?', '…', ',', '.', '-'," "])
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_spepc(self, filename):
        audio = load_audio(filename, int(self.hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return spec

    def clean_text_inf(self, text, language):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)
        return bert

    def get_phones_and_bert(self, text, language):
        if language in {"en", "all_zh", "all_ja"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
            if language == "zh":
                bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half else torch.float32,
                ).to(self.device)
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
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)
        return phones, bert.to(torch.float16 if self.is_half else torch.float32), norm_text

    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="不切", top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        prompt_language = self.dict_language[prompt_language]
        text_language = self.dict_language[text_language]
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in self.splits:
                prompt_text += "。" if prompt_language != "en" else "."
        text = text.strip("\n")
        text = self.replace_consecutive_punctuation(text)
        if text[0] not in self.splits and len(self.get_first(text)) < 4:
            text = "。" + text if text_language != "en" else "."
        zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3), dtype=np.float16 if self.is_half else np.float32)
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                    raise OSError("参考音频在3~10秒范围外，请更换！")
                wav16k = torch.from_numpy(wav16k)
                zero_wav_torch = torch.from_numpy(zero_wav)
                if self.is_half:
                    wav16k = wav16k.half().to(self.device)
                    zero_wav_torch = zero_wav_torch.half().to(self.device)
                else:
                    wav16k = wav16k.to(self.device)
                    zero_wav_torch = zero_wav_torch.to(self.device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(self.device)

        if how_to_cut == "凑四句一切":
            text = self.cut1(text)
        elif how_to_cut == "凑50字一切":
            text = self.cut2(text)
        elif how_to_cut == "按中文句号。切":
            text = self.cut3(text)
        elif how_to_cut == "按英文句号.切":
            text = self.cut4(text)
        elif how_to_cut == "按标点符号切":
            text = self.cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = self.process_text(texts)
        texts = self.merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language)
        for text in texts:
            if len(text.strip()) == 0:
                continue
            if text[-1] not in self.splits:
                text += "。" if text_language != "en" else "."
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language)
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = self.get_spepc(ref_wav_path)
            if self.is_half:
                refer = refer.half().to(self.device)
            else:
                refer = refer.to(self.device)
            audio = self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refer).detach().cpu().numpy()[0, 0]
            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
        return self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

    def merge_short_text_in_array(self, texts, threshold):
        if len(texts) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    def split(self, todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while True:
            if i_split_head >= len_text:
                break
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def cut1(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut2(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        if len(opts) > 1 and len(opts[-1]) < 50:
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut3(self, inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut4(self, inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip(".").split(".")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut5(self, inp):
        inp = inp.strip("\n")
        punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
        mergeitems = []
        items = []
        for i, char in enumerate(inp):
            if char in punds:
                if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)
        if items:
            mergeitems.append("".join(items))
        opt = [item for item in mergeitems if not set(item).issubset(punds)]
        return "\n".join(opt)

    def get_first(self, text):
        pattern = "[" + "".join(re.escape(sep) for sep in self.splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    def replace_consecutive_punctuation(self, text):
        punctuations = ''.join(re.escape(p) for p in self.punctuation)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result

    def process_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("请输入有效文本")
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def save_audio(self, audio, sampling_rate, output_path="output.wav"):
        sf.write(output_path, audio.astype(np.float32) / 32768, sampling_rate)
        print(f"Output saved to {output_path}")

    @staticmethod
    def parse_args(ref_wav_path, text, prompt_text="", prompt_language="中文", text_language="中文", 
                how_to_cut="不切", top_k=5, top_p=1.0, temperature=1.0, ref_free=False):
        parser = argparse.ArgumentParser(description="GPT-SoVITS CLI")
        parser.add_argument("--ref_wav_path", type=str, default=ref_wav_path, required=True, 
                            help="Path to the reference audio file (3-10 seconds).")
        parser.add_argument("--prompt_text", type=str, default=prompt_text, 
                            help="Text corresponding to the reference audio.")
        parser.add_argument("--prompt_language", type=str, default=prompt_language, 
                            choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], 
                            help="Language of the prompt.")
        parser.add_argument("--text", type=str, default=text, required=True, 
                            help="Target text to be synthesized.")
        parser.add_argument("--text_language", type=str, default=text_language, 
                            choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], 
                            help="Language of the target text.")
        parser.add_argument("--how_to_cut", type=str, default=how_to_cut, 
                            choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"], 
                            help="How to split the text.")
        parser.add_argument("--top_k", type=int, default=top_k, 
                            help="Top-k sampling parameter for GPT.")
        parser.add_argument("--top_p", type=float, default=top_p, 
                            help="Top-p sampling parameter for GPT.")
        parser.add_argument("--temperature", type=float, default=temperature, 
                            help="Temperature parameter for GPT.")
        parser.add_argument("--ref_free", action="store_true", default=ref_free, 
                            help="Enable reference-free mode.")
        return parser.parse_args()


    @classmethod
    def run_tts(cls, **kwargs):
        defaults = {
            "prompt_text": "",
            "prompt_language": "中文",
            "text_language": "中文",
            "how_to_cut": "按标点符号切",
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "ref_free": False,
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        SSGen = cls(args)
        sampling_rate, audio = SSGen.get_tts_wav(
            ref_wav_path=args.ref_wav_path,
            prompt_text=args.prompt_text,
            prompt_language=args.prompt_language,
            text=args.text,
            text_language=args.text_language,
        )
        output_path = "./DataBuffer/DigitalBuffer/output.mp3"
        SSGen.save_audio(audio, sampling_rate, output_path)
        return output_path

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")
