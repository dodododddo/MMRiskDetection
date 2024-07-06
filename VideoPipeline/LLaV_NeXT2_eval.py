import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# Load the model in half-precision
model = LlavaNextForConditionalGeneration.from_pretrained("./model/llava-v1.6-mistral-7b-hf", 
                                                          torch_dtype=torch.float16, 
                                                          device_map="auto",
                                                          use_flash_attention_2=True)
processor = AutoProcessor.from_pretrained("./model/llava-v1.6-mistral-7b-hf")

# Get three different images
image_paths = [f'./data/frames_p/output_{100*i:04}.png' for i in range(1,9)]
image_paths.append('./data/frames_n/output_0052.png')
image_paths.append('./data/test5.png')
# list = [1302,1304,1305,1307,1308]
# image_paths = [f'./data/DFMINST+_image/fake/{i}.png' for i in list]
# image_path = './data/frames/output_0001.png'
images = [Image.open(p) for p in image_paths]

# Prepare a batched prompt, where the first one is a multi-turn conversation and the second is not
# prompt = [
#     "[INST] <image>\nWhat is shown in these images?Do these images contain any evidence of pornography, violence, hatred, rumors, or face-changing techniques?Pay more attention to the people. [INST] <image>\nWhat is shown in these images?Do these images contain any evidence of pornography, violence, hatred, rumors, or face-changing techniques?Pay more attention to the people. [/INST]",
#     "[INST] <image>\nWhat is shown in these images?Do these images contain any evidence of pornography, violence, hatred, rumors, or face-changing techniques?Pay more attention to the people. [/INST]"
# ]
each_prompt = "[INST] <image>\n这张图片中是否包含暴力、色情、犯罪因素或AI合成痕迹？请多关注有异常行为的人。假如有以上因素，请只输出对应因素和对应因素的行为是什么。不要有多于输出。 [/INST]"
prompt = [each_prompt for _ in image_paths]

# We can simply feed images in the order they have to be used in the text prompt
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
print(each_prompt)
for i in range(len(prompt)):
    inputs = processor(text=[prompt[i]], images=[images[i]], padding=True, return_tensors="pt").to(model.device)

    # Generate
    # generate_ids = model.generate(**inputs, max_new_tokens=256, pad_token_id=2, use_cache=True)
    generate_ids = model.generate(**inputs, max_new_tokens=256, pad_token_id=2)
    text_outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    t = text_outputs[0]
    t = t.split('[/INST]')
    print('图片{}:'.format(i+1) + t[1]) 