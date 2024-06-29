import os
import json
import ast
import asyncio
import time
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI

from utils import pipeline, apipeline, replace_x_with_random, stream_pipeline
from utils.pipeline import async_pipeline


def generate_one_step(pipe:callable, input_text, max_retry_time=10):
    retry_interval = 1
    for _ in range(max_retry_time):
        try:
            answer = pipe(input_text)
        except Exception as e:
            print("任务执行出错：", e)
            print('重新请求....')
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)
            continue
        
        try:
            items = ast.literal_eval(answer)
            for item in items:
                item['风险点'] = '无'
                item['风险类别'] = '无风险'
                item['文本'] = replace_x_with_random(item['文本'])
            return items
            
        except(SyntaxError, KeyError, TypeError):
            print('parse failed, regenerating...')

        

            
        
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


# async def batch_generate_no_risk_data(client, example_path, dst_path, sample_num=5):
        
#     with open(example_path, 'r') as h:
#         examples = json.load(h)[:sample_num]
        

#     prompt = '''你是一个短信多样性生成专家，你将接受一些短信，请你模仿这些短信进行生成，你可以自由变换其内容，形式，语气等，多样性是你的核心目标。
#                 输出格式：[{"文本": "..."}, {"文本": "..."}, ...](不要出现任何多余内容，包括```json)'''


#     dst = []
#     parse_error = 0
#     batch_pipe = apipeline(client, temperature=1.25)
#     input_texts = [[
#             {
#             'role':'system',
#             'content': prompt
#             },
#             {   
#                 'role':'user',
#                 'content': example
#             }
#         ] for example in examples]
#     # print(input_texts)
    
#     answers = await batch_pipe(input_texts)
    
#     for answer in tqdm(answers):
#         try:
#             items = ast.literal_eval(answer)
#             for item in items:
#                 assert(isinstance(item, dict))
#                 item['风险点'] = '无'
#                 item['风险类别'] = '无风险'
#                 item['文本'] = replace_x_with_random(item['文本'])
#                 dst.append(item)
#             break
        
#         except(SyntaxError, KeyError, AssertionError, TypeError):
#             print('以下文本无法解析:' + answer)
            
    
#     with open(dst_path, 'w') as g:
#         json.dump(dst, g, indent=4, ensure_ascii=False)
        
#     print(f'转写完成, {len(dst)}条转写成功，{parse_error}条转写失败')

# def batch_generate_no_risk_data(client, example_path, dst_path, sample_num=5):
        
#     with open(example_path, 'r') as h:
#         examples = json.load(h)[:sample_num]

#     prompt = '''你是一个短信多样性生成专家，你将接受一些短信，请你模仿这些短信进行生成，你可以自由变换其内容，形式，语气等，多样性是你的核心目标。
#                 输出格式：[{"文本": "..."}, {"文本": "..."}, ...](不要出现任何多余内容，包括```json)'''
   
#     input_texts = [[
#         {
#         'role':'system',
#         'content': prompt
#         },
#         {   
#             'role':'user',
#             'content': example
#         }
#     ] for example in examples]
    
#     dst = []
#     pipe = pipeline(client, temperature=1.25)
    
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = [executor.submit(generate_one_step, pipe, input_text) for input_text in input_texts]
        
#         for job in as_completed(futures):
#             item = job.result(timeout=None)  # 默认timeout=None，不限时间等待结果
#             if item != None:
#                 dst += item
 
#             time.sleep(1)  # 为了避免超过OpenAI API的速率限制，每次预测之间间隔1秒


#     with open(dst_path, 'w') as g:
#         json.dump(dst, g, ind=4, ensure_ascii=False)

def batch_generate_no_risk_data(client, example_path, dst_path, sample_num=5):
    with open(example_path, 'r') as h:
        examples = json.load(h)[:sample_num]
    
    prompt = '''你是一个短信多样性生成专家，你将接受一些短信，请你模仿这些短信进行生成，你可以自由变换其内容，形式，语气等，多样性是你的核心目标。
                输出格式：[{"文本": "..."}, {"文本": "..."}, ...](不要出现任何多余内容，包括```json)'''
   
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
    
    dst = []
    resps = []
    pipe = stream_pipeline(client, temperature=1.25)
    
    for input_text in input_texts:
        print(pipe(input_text))
        dst.append(pipe(input_text))
        
    for resp in resps:
        reply = ""
        for x in resp:
            reply += x.choices[0].delta.content
        dst.append(reply)
        
    with open(dst_path, 'w') as g:
        json.dump(dst, g, ind=4, ensure_ascii=False)
        


    

if __name__ == '__main__':
    api_key = os.getenv('DEEPSEEK_API_KEY')
    base_url = "https://api.deepseek.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    example_path = "./dataset/no_risk_origin.json"
    dst_path = "./output/no_risk_batch.json"
    
    batch_generate_no_risk_data(client, example_path, dst_path, sample_num=20)
    
   