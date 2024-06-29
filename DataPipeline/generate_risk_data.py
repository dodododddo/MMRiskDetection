import os
import json
import ast
import random

from tqdm import tqdm
from openai import OpenAI

from utils import pipeline, replace_x_with_random

def generate_risk_data(client, src_path, example_path, dst_path, sample_num=600, max_regenerate_time = 5):

    with open(src_path, 'r') as f:
        source = json.load(f)
        
    with open(example_path, 'r') as h:
        examples = json.load(h)

    prompt = '''你是一个诈骗场景重现的专家，你将接受进行了脱敏处理的诈骗案情描述，请你仔细阅读案情描述，以受害者视角还原当时对方可能使用的语句，
            涉及到具体号码或数字的可自行选择一个补全，不允许出现xxxxx，不允许直接置空。
            你应该以场景还原的角度思考，但你只需要输出对方使用的语句即可，并标注其中的风险点，即指出句子中可能存在风险的内容，尽可能用短语归类，若可以用原文中含有的词汇概括则用原文词汇。
            我会为你提供一条示例短信，这条短信是无风险的，请你不要关注其内容，但在还原案件时模仿示例短信的格式、模式、语气等。
                输出格式：{"文本": "...", "风险点": "..."}(不要出现任何多余内容，包括```json)'''


    dst = []
    parse_error = []
    pipe = pipeline(client, temperature=1.25)
    for idx, data in enumerate(tqdm(source[:sample_num])):
        risk_example = "案情描述： " + data['案情描述']
        no_risk_example = "示例短信（无需关注内容，仅关注语气、格式）： " + random.choice(examples)['文本']
        input_text = [
            {
            'role':'system',
            'content': prompt
            },
            {   
                'role':'user',
                'content': risk_example + no_risk_example
            }
        ]
        
        for _ in range(max_regenerate_time):
            answer = pipe(input_text)
            try:
                item = ast.literal_eval(answer)
                
                if item['风险点'] == '无':
                    item['风险类别'] = '无风险'
                else:
                    item['风险类别'] = data['案件类别']
                    
                item['文本'] = replace_x_with_random(item['文本'])
                dst.append(item)
                break
            
            except(SyntaxError, KeyError, TypeError):
                print('parse failed, regenerating...')
        else:
            parse_error.append(idx)
            print('以下文本无法解析:' + answer)
            print(f'第{idx + 1}条案情转写内容解析失败')
            

    with open(dst_path, 'w') as g:
        json.dump(dst, g, indent=4, ensure_ascii=False)
        
    print(f'转写完成, {len(dst)}条转写成功，{len(parse_error)}条转写失败')
    print(f'需重新转写的案件id: {parse_error}')
    

if __name__ == '__main__':
    client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    src_path = "./dataset/ccl_2023_eval_6_train.json"
    example_path = "./dataset/no_risk_origin.json"
    # dst_path = "./DataPipeline/output/risk_diff1.json"
    dst_path = "./output/test1.json"
    
    generate_risk_data(client, src_path, example_path, dst_path, sample_num=10)
    
   
