# from transformers import AutoModel, AutoTokenizer
# from utils import ocr_pipeline, recursively_read

# ocr_model_path = "./model/MiniCPM-V-2"
# tokenizer = AutoTokenizer.from_pretrained(ocr_model_path, trust_remote_code=True)
# ocr_model = AutoModel.from_pretrained(ocr_model_path, trust_remote_code=True)

# img_paths = recursively_read('data/ocr')

# question = '请输出图片中的所有文字'
# ocr_pipe = ocr_pipeline(img_paths, ocr_model, tokenizer)
# ocr_pipe(question)

from utils import ocr

img_path = 'data/ocr/test/test1.png'
result_path = 'data/ocr/result/test1.png'
out = ocr(img_path, save_image=True)
print(out)