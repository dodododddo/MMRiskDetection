import os
import json
import ast
import random
import concurrent.futures

from tqdm import tqdm
from openai import OpenAI

from utils import replace_x_with_random, GenPipeline

def generate_risk_data(pipe, src_path, example_path, dst_path, sample_num=600, max_regenerate_time = 5, max_workers=500):

    with open(src_path, 'r') as f:
        source = json.load(f)
        
    with open(example_path, 'r') as h:
        examples = json.load(h)

    prompt = '''你是一个诈骗场景重现的专家，你将接受进行了脱敏处理的诈骗案情描述，请你仔细阅读案情描述，以受害者视角还原当时对方可能使用的语句，
            涉及到具体号码或数字或用户名称的如果未给出，请自行补全，不允许出现xxxxx，不允许直接置空；涉及到字符串或者网址的可任意编写一个字符串或者网址，不允许出现xxxxx，不允许直接置空。
            你应该以场景还原的角度思考，还原对方受骗时收到的短信消息，并标注其中的风险点，即指出句子中可能存在风险的内容，尽可能用短语归类，并控制在三个短语以内，可以将多个短语进行归类整合，若可以用原文中含有的词汇概括则用原文词汇，或者对原文词汇进行简写，尽量使用常见词汇描述，对于风险点的描述尽量控制在五个字以内，且风险点中一定要体现出风险，例如“QQ联系”应该换为“诱导添加陌生账号”。
            我会为你提供一条示例短信，这条短信是无风险的，请你不要关注其内容，但在还原案件时模仿示例短信的格式、模式、语气等。
            考虑到诈骗信息的多样性，在生成语句时，你应当注意不要一直使用一个模板进行描述，在句子中可以不使用您好问候。在称呼受害者时可以使用原文中出现的软件名称加上一个随机的用户名称呼该软件的用户，请保证用户名的多样性。
            一些涉及风险点的一些词汇可以使用谐音字代替，来规避诈骗短信的筛查。
            请保证生成的文本是在与受害者对话，而不是描述案情。
            并再根据输出的文本进行改写，思考如果不是诈骗短信应该如何编辑，并输出一条无风险的文本。请保证该文本的风险点为“无”，并保证该无风险文本与先前输出的有风险文本有一定的相似性，但不具有诈骗性质，不能携带有风险的内容，例如不能出现请联系xx或请添加xx联系方式或向某个银行卡转账。
            对于无风险文本请不要出现谐音代替，例如使用“支付宝”而不是其它谐音代替。如果在生成有风险文本时使用了谐音转换，请在生成无风险文本时将其还原成原本描述。
            请保证生成的无风险文本的多样性，可以是对用户的活动邀请，也可以是对用户的风险提示，不要拘泥于一种表达形式。请根据给出的无风险示例短信，保证输出的文本更贴近于人类的语言。
            一共需要输出两条文本，一条为有风险文本，一条为无风险文本
                输出格式：[{"文本": "...", "风险点": "..."}，{"文本": "...", "风险点": "..."}](不要出现任何多余内容，包括```json)'''


    dst = []
    parse_error = []
    texts = ["案情描述： " + data['案情描述'] + "示例短信（无需关注内容，仅关注语气、格式）： " + random.choice(examples)['文本'] for data in source[:sample_num]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(pipe.generate, text, prompt) for text in texts]
        answers = [future.result() for future in concurrent.futures.as_completed(futures)]
        
    for idx, answer in enumerate(tqdm(answers)): 
        try:
            items = ast.literal_eval(answer)
            for item in items:
                if item['风险点'] == '无':
                    item['风险类别'] = '无风险'
                else:
                    item['风险类别'] = source[idx]['案件类别']
                item['文本'] = replace_x_with_random(item['文本'])
                dst.append(item)
        
        except:
            parse_error.append(idx)
            print('以下文本无法解析:' + answer)
            print(f'第{idx + 1}条案情转写内容解析失败')
            

    with open(dst_path, 'w') as g:
        json.dump(dst, g, indent=4, ensure_ascii=False)
        
    print(f'转写完成, {len(dst)}条转写成功，{len(parse_error)}条转写失败')
    print(f'需重新转写的案件id: {parse_error}')
    

if __name__ == '__main__':
    pipe = GenPipeline(api_key=os.getenv('DEEPSEEK_API_KEY'))
    src_path = "./dataset/ccl_2023_eval_6_train_trans_and_eval_split.json"
    example_path = "./dataset/no_risk_origin.json"
    dst_path = "./nothing.json"
    
    generate_risk_data(pipe, src_path, example_path, dst_path, sample_num=100)
    
   
