from model.LLaVA_NeXT_inference.llava.model.builder import load_pretrained_model
from model.LLaVA_NeXT_inference.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from model.LLaVA_NeXT_inference.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from model.LLaVA_NeXT_inference.llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

pretrained = "./model/LLaVA_NeXT"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

image_paths = [f'./data/frames_p/output_{100*i:04}.png' for i in range(3,10)]
# image_path = './data/frames/output_0001.png'
images = [Image.open(p) for p in image_paths]
image_tensor = process_images(images, image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
# question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in these images?Do these images contain any evidence of pornography, violence, hatred, rumors, or face-changing techniques?Pay more attention to the people."
question = DEFAULT_IMAGE_TOKEN + "\n这张图片展示了什么？其中是否包含暴力、色情、犯罪因素和AI合成痕迹？请多关注有异常行为的人。"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print(prompt_question)

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size for image in images]

print('\n model generate')
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=True,
    temperature=0.5,
    max_new_tokens=256,
    pad_token_id=128001,
    use_cache=True
)
print('\n' + question)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
print(text_outputs[0])
