# # from model.LLaVA_NeXT_inference.llava.constants import DEFAULT_IMAGE_TOKEN

# # i = 0
# # t = '[INST]这张图片展示了什么？其中是否包含暴力、色情、犯罪因素和AI合成痕迹？请多关注有异常行为的人。 [/INST] 这张图片显示了一组人在一个外部场景中，其中一些人的行为异常。这些人在一个红色的地面上聚集，有些人的行为看起来像是在抢劫或者扰凶，因为他们的手势和姿势表现出来的是暴力行为。'
# # t = t.split('[/INST]')
# # # print(t)
# # print('图片{}:'.format(i+1) + t[1]) 

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
# model.eval()

# class KVCache:
#     def __init__(self):
#         self.key_cache = None
#         self.value_cache = None

#     def update_cache(self, key, value):
#         if self.key_cache is None:
#             self.key_cache = key
#             self.value_cache = value
#         else:
#             self.key_cache = torch.cat([self.key_cache, key], dim=-2)
#             self.value_cache = torch.cat([self.value_cache, value], dim=-2)

#     def get_cache(self):
#         return self.key_cache, self.value_cache

#     def reset_cache(self):
#         self.key_cache = None
#         self.value_cache = None

# def generate_with_kvcache(model, tokenizer, prompt, max_length=50, cache=None):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs.input_ids

#     # 初始化缓存
#     if cache is None:
#         cache = KVCache()

#     # 生成初始输入的输出和注意力缓存
#     with torch.no_grad():
#         outputs = model(input_ids, use_cache=True)
#         logits = outputs.logits
#         past_key_values = outputs.past_key_values

#         # 更新缓存
#         for layer_idx, (key, value) in enumerate(past_key_values):
#             cache.update_cache(key, value)

#     # 生成后续的 token
#     generated_tokens = []
#     for _ in range(max_length):
#         with torch.no_grad():
#             key_cache, value_cache = cache.get_cache()
#             outputs = model(input_ids, past_key_values=(key_cache, value_cache), use_cache=True)
#             logits = outputs.logits
#             next_token = torch.argmax(logits[:, -1, :], dim=-1)
#             generated_tokens.append(next_token.item())
#             input_ids = next_token.unsqueeze(0)

#             # 更新缓存
#             past_key_values = outputs.past_key_values
#             for layer_idx, (key, value) in enumerate(past_key_values):
#                 cache.update_cache(key, value)

#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#     return generated_text

# # 示例使用
# prompt = "Once upon a time"
# generated_text = generate_with_kvcache(model, tokenizer, prompt)
# print(generated_text)

### 定义文件名获取函数
import os 
def data_needed(filePath):
    file_name = list()        #新建列表
    for i in os.listdir(filePath):        #获取filePath路径下所有文件名
        data_collect = ''.join(i)        #文件名字符串格式
        file_name.append(filePath + data_collect)        #将文件名作为列表元素填入
    print("获取filePath中文件名列表成功")        #打印获取成功提示
    return(file_name)        #返回列表
### 主函数
if __name__ == '__main__':
    filePath = "./data/DFMNIST+/fake_dataset/blink/"        #想要获取文件名的路径
    print(data_needed(filePath))        #调用文件名获取函数，并打印结果
