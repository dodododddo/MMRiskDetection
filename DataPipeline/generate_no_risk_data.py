import os
import json
import ast
import asyncio
import random
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI

from utils import pipeline, apipeline, replace_x_with_random
from utils.pipeline import async_pipeline


def generate_no_risk_data(client, example_path, dst_path, sample_num=600, max_regenerate_time = 5):
        
    with open(example_path, 'r') as h:
        examples = json.load(h)[:sample_num]
        

    prompt = '''你是一个短信多样性生成专家，你将接受一些短信，请你模仿这些短信进行生成，你可以自由变换其内容，形式，语气等，多样性是你的核心目标。
                输出格式：[{"文本": "..."}, {"文本": "..."}, ...](不要出现任何多余内容，包括```json)'''
    

    dst = []
    parse_error = []
    pipe = pipeline(client, temperature=1.25)
    for idx, data in enumerate(tqdm(examples)):
        example = "示例短信：" + data['文本']
        input_text = [
            {
            'role':'system',
            'content': prompt
            },
            {   
                'role':'user',
                'content': example
            }
        ]
        
        for _ in range(max_regenerate_time):
            answer = pipe(input_text)
            try:
                items = ast.literal_eval(answer)
                for item in items:
                    item['风险点'] = '无'
                    item['风险类别'] = '无风险'
                    item['文本'] = replace_x_with_random(item['文本'])
                    dst.append(item)
                break
            
            except(SyntaxError, KeyError):
                print('parse failed, regenerating...')
        else:
            parse_error.append(idx)
            print('以下文本无法解析:' + answer)
            print(f'第{idx + 1}条短信重写失败')
            

    with open(dst_path, 'w') as g:
        json.dump(dst, g, indent=4, ensure_ascii=False)
        
    print(f'转写完成, {len(dst)}条转写成功，{len(parse_error)}条转写失败')
    print(f'需重新转写的短信id: {parse_error}')


async def batch_generate_no_risk_data(aclient, model, base_url, example_path, dst_path, sample_num=600):
        
    with open(example_path, 'r') as h:
        examples = json.load(h)[:sample_num]
        

    prompt = '''你是一个短信多样性生成专家，你将接受一些短信，请你模仿这些短信进行生成，你可以自由变换其内容，形式，语气等，多样性是你的核心目标。
                输出格式：[{"文本": "..."}, {"文本": "..."}, ...](不要出现任何多余内容，包括```json)'''


    dst = []
    parse_error = 0
    # batch_pipe = apipeline(aclient, temperature=1.25)
    batch_pipe = async_pipeline(model, base_url)
    input_texts = [[
            {
            'role':'system',
            'content': prompt
            },
            {   
                'role':'user',
                'content': example
            }
        ] for example in examples]
    # print(input_texts)
    
    answers = await batch_pipe(input_texts)
    
    for answer in tqdm(answers):
        try:
            items = ast.literal_eval(answer)
            for item in items:
                assert(isinstance(item, dict))
                item['风险点'] = '无'
                item['风险类别'] = '无风险'
                item['文本'] = replace_x_with_random(item['文本'])
                dst.append(item)
            break
        
        except(SyntaxError, KeyError, AssertionError, TypeError):
            print('以下文本无法解析:' + answer)
            
    
    with open(dst_path, 'w') as g:
        json.dump(dst, g, indent=4, ensure_ascii=False)
        
    print(f'转写完成, {len(dst)}条转写成功，{parse_error}条转写失败')
    
    

if __name__ == '__main__':
    api_key = os.getenv('DEEPSEEK_API_KEY')
    base_url = "https://api.deepseek.com/v1"
    model = "deepseek-chat"
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    example_path = "./dataset/no_risk_origin.json"
    dst_path = "./output/no_risk_batch.json"
    # generate_no_risk_data(client, example_path, dst_path, sample_num=200)
    
    asyncio.run(batch_generate_no_risk_data(client, model, base_url, example_path, dst_path, sample_num=1))
    
   
