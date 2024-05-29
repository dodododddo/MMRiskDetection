import os
import json
import ast

from tqdm import tqdm
from openai import OpenAI

from utils import pipeline

api_key = os.getenv('DEEPSEEK_API_KEY')
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
src_path = "./Dataset/test.json"
dst_path = "./LLM_API/data/generate.json"

with open(src_path, 'r') as f:
    source = json.load(f)

# prompt = '''任务：你是一个诈骗场景重现的专家，你将接受进行了脱敏处理的诈骗案情描述，请你仔细阅读案情描述，以受害者视角还原当时对方可能使用的语句，
#           涉及到具体银行名、账号名混淆的可自行选择一个补全，不允许出现X元、某个这样的脱敏词汇。你应该以场景还原的角度思考，但你只需要输出对方使用的语句即可，并标注其中的风险点，即指出句子中
#           可能存在风险的内容，尽可能用短语归类。
#             输出格式：[{"文本": "...", "风险点": "..."}, {...}](不要出现任何多余内容，包括```json)'''

prompt = '''任务：你是一个诈骗场景重现的专家，你将接受进行了脱敏处理的诈骗案情描述，请你仔细阅读案情描述，以受害者视角还原当时对方可能使用的语句，
          涉及到具体银行名、账号名混淆的可自行选择一个补全，不允许出现X元、某个这样的脱敏词汇。你应该以场景还原的角度思考，但你只需要输出对方使用的语句即可。
          并标注其中的风险点，即指出句子中可能存在风险的词句，并尽可能用短语归类。
          同时，请你考虑一个类似的场景，输出正常的、无风险情况下对面可能使用的语句，并将风险点标注为无。
            输出格式：[{"文本": "...", "风险点": "..."}, {...}](不要出现任何多余内容，包括```json, 请将有无风险的内容写在同个列表中，并确保无风险的内容不低于2条)'''


dst = []
parse_error = []
pipe = pipeline(client)
for idx, data in enumerate(tqdm(source[:100])):
    example = "案情描述: " + data['案情描述']

    input_text = [
        {
            'role':'user',
            'content': prompt + example
        }
    ]
    answer = pipe(input_text)
    try:
        item = ast.literal_eval(answer)
        dst.append(item)
    except(SyntaxError):
        parse_error.append(idx)
        print(answer)
        print(f'第{idx + 1}条案情转写内容解析失败')
        

with open(dst_path, 'w') as g:
    json.dump(dst, g, indent=4, ensure_ascii=False)
    
print(f'转写完成, {len(dst)}条转写成功，{len(parse_error)}条转写失败')
print(f'需重新转写的案件id: {parse_error}')
   
