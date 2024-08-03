import os
import json
import ast
import csv
import math
import random
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from utils import replace_x_with_random, GenPipeline

def extract_between(s, x, y):
    if s is None:
        return None 
    start_index = s.find(x)
    end_index = s.rfind(y)

    if start_index == -1:
        return s
    if end_index == -1:
        return None
    extracted_substring = s[start_index:end_index + 1]
    return extracted_substring

def generate_risk_data(pipe, src_path, example_path, dst_path, start_num=0, sample_num=600, max_regenerate_time = 5, max_workers=500):

    with open(src_path, 'r') as f:
        source = json.load(f)
        
    example_texts = []
    with open(example_path, 'r') as h:
        examples = json.load(h)
    for sublist in examples:
        for item in sublist:
            example_texts.append(item["文本"])

    prompt1 = '''你是一个诈骗场景重现的专家，你将接受进行了脱敏处理的诈骗案情描述，
            首先，请你仔细阅读案情描述中受害者与对方客服的交流内容，以受害者视角还原当时双方可能进行的多轮对话，
            涉及到具体号码或数字的可自行选择一个补全，不允许出现xxxxx，不允许直接置空；
            涉及到字符串或者网址的可任意编写一个字符串或者网址，不允许出现xxxxx，不允许直接置空。
            你应该以场景还原的角度思考，但你只需要输出对话双方使用的语句即可，并按照案件描述的风险类别来识别风险点并进行标注，
            即指出句子中可能存在风险的内容，尽可能用短语归类，若可以用原文中含有的词汇概括则用原文词汇。
            注意有风险对话内容必须是【坐席】对【客户】进行诈骗的诱导，有风险的对话内容不应该出现【坐席】的安全提醒和劝说【客户】报警等内容。
            注意有风险对话内容必须是基于案情描述和相应的案件类别产生的，目的在于对案情描述中的受害人进行诈骗，涉及具体信息不能由【客户】口述，也不应出现点击链接、打开邮箱等操作或括号（）。
            
            其次，根据输出的对话进行改写，思考如果不是在进行诈骗应该如何对话，并输出一段无风险的对话。请保证该文本的风险点为“无”，并保证该无风险文本与先前输出的有风险文本有一定的相似性，但不能完全一样或具有高度的相似性。
            请保证生成的无风险对话内容的多样性，可以是对用户的活动邀请，也可以是对用户的风险提示，不要拘泥于一种表达形式。请根据给出的无风险示例对话，保证输出的文本更贴近于人类对话的语言。
            我会为你提供一段示例对话，不要关注对话内容，只需在还原对话输出时模仿示例对话中双方说话的格式、模式、语气等。
             一共需要输出两条文本，一条为有风险文本，一条为无风险文本，无风险文本的“风险点”为“无”，“风险类别”为“无风险”。
             无论冒充的身份和受害者是谁，冒充者的发言统一用【坐席】表示，受害者的发言统一用【客户】表示。
             请严格按照输出格式输出，不要出现任何多余内容，包括```json。
             注意两条文本的内容不能一样！
              风险点需要用简短的短语概括，若可用原文中词句可直接使用，尽可能用短语归类，并控制在三个短语以内，可以将多个短语进行归类整合，若可以用原文中含有的词汇概括则用原文词汇，或者对原文词汇进行简写，尽量使用常见词汇描述，对于风险点的描述尽量控制在五个字以内，且风险点中一定要体现出风险，例如“QQ联系”应该换为“诱导添加陌生账号”。
              风险类型和序号已给定。
              正确的输出格式：[{"文本": "【坐席】...，【客户】...，【坐席】...，【客户】...", "风险点": "...", "风险类别": "...", "序号":...}, {"文本": "【坐席】...，【客户】...，【坐席】...，【客户】...", "风险点": "...", "风险类别": "...", "序号":...}]（不要输出无关内容，例如```json） 
    '''


    prompt2 = '''你是一个诈骗专家，可以根据给出的场景和诈骗手法对不同用户群体进行诈骗。现在给你一个诈骗场景，要求你使用指定的诈骗手法，对相应的受骗群体进行诈骗。
             同时根据对方的回答，继续进行诱导或者指导。
             涉及到具体号码或数字或用户名称的请自行补全，不允许出现xxxxx，不允许直接置空；涉及到字符串的可任意编写一个字符串，不允许出现xxxxx，不允许直接置空，不允许用括号代替，必须用有意义的数字和字符。
             对于你所生成的诈骗信息，你需要标注其中的风险点，即指出句子中可能存在风险的内容。
             双方在说话时，都会习惯性地添加一些类似于“嗯”“哦”“噢”“诶”“啊”“吧”之类的语气词。
             考虑到诈骗信息的多样性，在生成语句时，你应当注意不要一直使用一个模板进行描述，在句子中可以不使用您好问候。
             在称呼受害者时可以使用原文中出现的软件名称加上一个随机的用户名称呼该软件的用户，请保证用户名的多样性。
             请保证生成的对话是在与受害者交流，而不是描述案情。
             并再根据输出的对话进行改写，思考如果不是在进行对话应该如何编辑，并输出一段无风险的对话。请保证该文本的风险点为“无”，并保证该无风险文本与先前输出的有风险文本有一定的相似性，但不具有诈骗性质，不能携带有风险的内容，例如不能出现请联系xx或请添加xx联系方式或向某个银行卡转账。
             请保证生成的无风险文本的多样性，可以是对用户的活动邀请，也可以是对用户的风险提示，不要拘泥于一种表达形式。请根据给出的无风险示例短信，保证输出的文本更贴近于人类对话的语言。
             现在给你一个诈骗场景为：‘<risk_scene>’，要求你使用‘<risk_way>’的方法，冒充<risk_criminal>对<risk_victim>群体进行诈骗。
             注意对话中不应该出现“点击链接”、“打开邮箱”或者出现括号“（）”等内容。
             我会为你提供一段示例对话，不要关注对话内容，只需在还原对话输出时模仿示例对话中双方说话的格式、模式、语气等。
             一共需要输出两条文本，一条为有风险文本，一条为无风险文本。
             无论冒充的身份和受害者是谁，冒充者的发言统一用【坐席】表示，受害者的发言统一用【客户】表示。
             请严格按照输出格式输出，不要出现任何多余内容，包括```json。
             1. 风险类型包括{'冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类', '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险'}，若认为短信是无风险文本则风险类型为无风险。
             2. 风险点需要用简短的短语概括，若可用原文中词句可直接使用，尽可能用短语归类，并控制在三个短语以内，可以将多个短语进行归类整合，若可以用原文中含有的词汇概括则用原文词汇，或者对原文词汇进行简写，尽量使用常见词汇描述，对于风险点的描述尽量控制在五个字以内，且风险点中一定要体现出风险，例如“QQ联系”应该换为“诱导添加陌生账号”。不同风险点之间用"、"连接。
             3. 对话内容即文本按照格式输出一个字符串，其中不含换行符。
             正确的输出格式：[{"文本": "【坐席】...，【客户】...，【坐席】...，【客户】...", "风险点": "...", "风险类别": "..."}, {"文本": "【坐席】...，【客户】...，【坐席】...，【客户】...", "风险点": "...", "风险类别": "..."}]（不要输出无关内容，例如```json） 
        '''
    
    # df = pd.DataFrame({'index': [], 'f_index':[], 'riskType': [], 'riskPoint': [], 'text': []})
    df = pd.DataFrame({'index': [], 'riskType': [], 'riskPoint': [], 'text': []})
    # texts = ["不要输出无关内容，例如```json \n据此案情描述进行对话生成\n"+"案情描述： " + data['案情描述'] + "风险类别：" + data['案件类别'] + "序号：" + str(data['序号']) + "示例短信（无需关注内容，仅关注语气、格式、双方语言风格）： "
    # texts = ["不要输出无关内容，例如```json \n 示例短信（无需关注内容，仅关注语气、格式、双方语言风格）： "
    #          + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] 
    #          + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] + random.choice(examples)['文本'] 
    #          for i, data in enumerate(source[start_num:start_num+sample_num])]
    dst = []
    rewrite_index = []
    parse_error = []
    input_text = "现在给你一个诈骗场景为：‘银行’，要求你使用‘提供个人信息’的方法，冒充‘银行员工’对‘消费者群体’进行诈骗。"
    question = "现在给你一个诈骗场景为：‘<risk_scene>’，要求你使用‘<risk_way>’的方法，冒充<risk_criminal>对<risk_victim>群体进行诈骗。"
    
    # input_text = "据此案情描述进行对话生成\n" + "案情描述：2022年12月23日，在7-1-202家中想在网上贷款，于是在网上搜索并下载名为“小度”的软件，其认为是百度公司的“度小满”借款软件。后在网站注册并贷款三万元。其查询软件显示银行信息有误，银行卡冻结。遂联系该软件业务员，业务员称其银行卡填写错误，并给其发送一个二维码，通过该软件与客服沟通。照做，并下载“诚信通”软件。后该客服给其看“银监会”公告，称其涉嫌骗贷，需要交纳百分之五十的认证金15000元，并给发来一张银行卡号，并称操作成功后给其发放45000元。给该账号转款15000元。后去提现发现二次冻结。于是又问业务员是怎么回事，业务员称后台没有修改成功，就点提现导致贷款被冻结，需要交纳45000元解冻，意识被骗，遂报警，共计损失15000元。嫌疑账户：，账号：，浙江农村信用社。涉案APP：小度、诚信通。被害人通过嫌疑账号与对方联系，故无其他联系方式。国家反诈中心APP已采集被害人手机。经查，此案受害人自己从网上下载，不涉及嫌疑电话。" + "风险类别：虚假信用服务类" + "序号：0"
    
    i = 40743
    # leak_index = [18963]
    
    # leak_source = [source[i] for i in leak_index]
    for k in range(math.ceil(sample_num / max_workers)):
        # question = ["据此案情描述进行对话生成\n案情描述：" + data['案情描述'] + "风险类别：" + data['案件类别'] + "序号：" + 
                    # str(data['序号']) for data in source[start_num+k*max_workers: start_num+(k+1)*max_workers]]
                    # str(data['序号']) for data in leak_source[start_num+k*max_workers: start_num+(k+1)*max_workers]]
        output_texts = [random.choice(examples) for _ in range(max_workers)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(pipe.generate, input_text, str(output_text), prompt2, question) for output_text in output_texts]
            answers = [future.result() for future in concurrent.futures.as_completed(futures)]

        # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        #     futures = [executor.submit(pipe.generate, input_text, str(output_texts[i]), prompt1, question[i]) for i in range(max_workers)]
        #     answers = [future.result() for future in concurrent.futures.as_completed(futures)]
        if "failed" in answers:
            return k * max_workers
        # print(answers)
        all_path = "DataPipeline/output/dialog/all_generate.csv"
        with open(all_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([answers])

        for idx, answer in enumerate(tqdm(answers)): 
            i = i + 1
            if i == start_num + sample_num:
                return "success"
            try:
                answer = extract_between(answer,'[',']')
                if answer is None:
                    continue
                items = ast.literal_eval(answer)
                
                for item in items:
                    if item['文本'] in example_texts:
                        rewrite_index.append(item['序号'])
                        continue
                    df = pd.DataFrame({'index': [], 'f_index':[], 'riskType': [], 'riskPoint': [], 'text': []})
                    # df = pd.DataFrame({'index': [], 'riskType': [], 'riskPoint': [], 'text': []})
                    item['文本'] = replace_x_with_random(item['文本'])
                    dst.append(item)
                    # data = {'index': i, 'f_index': item['序号'], 'riskType': item['风险类别'], 'riskPoint': item['风险点'], 'text':item['文本']}
                    data = {'index': i, 'riskType': item['风险类别'], 'riskPoint': item['风险点'], 'text':item['文本']}
                    df.loc[len(df)] = data
                    df.to_csv("DataPipeline/output/dialog/prompt2_generate_8w.csv",encoding="utf-8",index=False,mode='a',header=False)

            except:
                wrong_path = "DataPipeline/output/dialog/error_interrupt.csv"
                with open(wrong_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([answer])
                parse_error.append(idx)
                print('以下文本无法解析:' + answer)
                print(f'第{idx + 1}条案情转写内容解析失败')
                with open(wrong_path, 'a') as g:
                    json.dump(answer + "\n", g, indent=4, ensure_ascii=False)
                        
        # print(dst)
        with open(dst_path, 'w') as g:
            json.dump(dst, g, indent=4, ensure_ascii=False)
        # df.to_csv("DataPipeline/output/dialog/prompt2_test.csv",encoding="utf-8",index=False,mode='a',header=False)
            
        print(f'转写完成, {len(dst)}条转写成功，{len(parse_error)}条转写失败')
        print(f'需重新转写的案件id: {parse_error}')
        print(f'重复示例的案件id: {rewrite_index}')
                                                            
    return "success"
    
if __name__ == '__main__':
    # api_key = 'sk-3a7a48e5c9db4968b8283422e7fe1624'     # 已用完的
    # api_key = 'sk-18cbfd122fb249e8b626a9003209eb1d'
    # api_keys = ['sk-18cbfd122fb249e8b626a9003209eb1d', 'sk-b6147a3f06db47c38690f0a106493f2e']
    # pipe = GenPipeline(api_key=api_key)
    src_path = "DataPipeline/dataset/train_rag_or_finetuning_split_index.json"
    example_path = "DataPipeline/output/dialog/test/prompt1_output_examples.json"
    dst_path = "DataPipeline/output/dialog/test.json"
    keys_path = "DataPipeline/output/dialog/key.txt"
    with open(keys_path, 'r', encoding='utf-8') as file:
        api_keys = file.readlines()
    pipe = GenPipeline(api_key=api_keys[0].strip())

    start_num=40700
    sample_num=1
    max_workers=1
    result = generate_risk_data(pipe, src_path, example_path, dst_path, start_num=start_num, sample_num=sample_num, max_workers=max_workers)
    print(result)
    while result != "success":
        print(api_keys[0].strip()+"   tokens已耗尽")
        del api_keys[0]
        pipe = GenPipeline(api_key=api_keys[0].strip())
        start_num = start_num + result
        sample_num = sample_num - result
        result = generate_risk_data(pipe, src_path, example_path, dst_path, start_num=start_num, sample_num=sample_num, max_workers=max_workers)
    print(api_keys)
    
   
