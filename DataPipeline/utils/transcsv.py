import json
import csv
import ast
import pandas as pd
from collections import Counter

def extract(original_string, start_string):
    start_index = original_string.find(start_string)
    if start_index != -1:
        return original_string[start_index:]
    else:
        return original_string

def trans_test_to_dataset():
    file = "DataPipeline/dataset/customer_service_call_text_test_dataset_no_risk.csv"
    dst_path = "DataPipeline/output/dialog/dialog_norisk_test.json"
    df = pd.read_csv(file,usecols=[0])
    df.columns = ['input']
    df['output'] = "{'风险类别':'无风险','风险点':'无'}"
    df['input'] = df['input'].apply(lambda x: x[6:])
    data = []
    json_data = df.to_json(orient='records', force_ascii=False)
    items = ast.literal_eval(json_data)
    for item in items:
        data.append(item)
    with open(dst_path, 'w') as g:
        json.dump(data, g, indent=4, ensure_ascii=False)

def trans_train_to_dataset():
    file = "DataPipeline/dataset/customer_service_call_text_train_dataset_no_risk.csv"
    dst_path = "DataPipeline/output/dialog/dialog_norisk_train.json"
    df = pd.read_csv(file,usecols=[0])
    df.columns = ['input']
    df['output'] = "{'风险类别':'无风险','风险点':'无'}"
    df['input'] = df['input'].apply(lambda x: extract(x, "【"))
    data = []
    json_data = df.to_json(orient='records', force_ascii=False)
    items = ast.literal_eval(json_data)
    for item in items:
        data.append(item)
    with open(dst_path, 'w') as g:
        json.dump(data, g, indent=4, ensure_ascii=False)

def trans_test_to_example():
    file = "DataPipeline/dataset/customer_service_call_text_test_dataset_no_risk.csv"
    dst_path = "DataPipeline/output/test/dialog_norisk_example.json"
    df = pd.read_csv(file,usecols=[0])
    df.columns = ['input']
    df['output'] = "{'风险类别'：无风险,'风险点'：无}"
    df['input'] = df['input'].apply(lambda x: x[6:])
    df.columns = ['文本']
    df['风险点'] = "无"
    df['风险类别'] = "无风险"
    data = []
    json_data = df.to_json(orient='records', force_ascii=False)
    items = ast.literal_eval(json_data)
    for item in items:
        data.append(item)
    with open(dst_path, 'w') as g:
        json.dump(data, g, indent=4, ensure_ascii=False)

def trans_test_to_diversity():
    file1 = "DataPipeline/output/dialog/table_new/p2_selected.json"
    file2 = "DataPipeline/output/message/useful/finetuning_initial_small3.json"
    file3 = "DataPipeline/output/message/message_finetuning11.json"
    dst_file = "DataPipeline/output/dialog/table_new/p2_selected.csv"
    files = [file1]
    for file in files:
        with open(file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        df.columns = ['text','riskType','index']
        df = df.drop(columns=['index'])
        df.to_csv(dst_file,mode='a',encoding='utf-8',header=False,index=False)
        # print(df[:10])

def leak_filling():
    df = pd.read_csv("DataPipeline/output/dialog/prompt1_generate_8w.csv")
    df = df[df['riskType']!='无风险']
    print(len(df))
    df_index = df['index'].values
    df_riskType = df['riskType'].values
    index = range(40700)
    # result = [element for element in df_index if element not in index]
    counter = Counter(df_index)
    counter_rt = Counter(df_riskType)
    result = [element for element, count in counter.items() if count == 2]
    print(result)
    # print(counter_rt)

def read_csv(file_path):
    data = []
    with open(file_path, newline='',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def csv_to_json(csv_file, json_file):
    data = read_csv(csv_file)
    with open(json_file, 'w',encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4,ensure_ascii=False)
    return data
    # with open(json_file, 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    # new_data = []
    # for item in data:
    #     item['文本'] = item.pop('text')
    #     item['风险类别'] = item.pop('riskType')
    #     # item.pop('f_index')
    #     # item.pop('index')
    #     new_data.append(item)
    # with open(json_file, 'w', encoding='utf-8') as file:
    #     json.dump(new_data, file, indent=4, ensure_ascii=False)

def trans_reward_json(data_path, to_path):
    # 读取JSON文件
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 创建一个新的数据列表
    new_data = []

    instruction = '''你是一个风险判断的专家，你将接受一段对话文本，请给出该对话中存在的风险类型：
                    注意点：1. 风险类型包括；'冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类','网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险'，若认为短信是无风险文本则风险类型为无风险。
                    2.只需要输出风险类型即可，不需要添加其它多余符号。输出格式：...'''
    # 遍历原数据，修改标签并加入新内容
    for item in data:
        message = {}
        message['messages'] = []
        message['messages'].append({'role': 'system', 'content': instruction})
        message['messages'].append({'role': 'user', 'content': item['文本']})
        message['messages'].append({'role': 'assistant', 'content': item['风险类别']})
        new_data.append(message)

    # 将修改后的数据写入新的JSON文件
    with open(to_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

def trans_json(source_file, dst_file):
    with open(source_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []

    # 遍历原数据，修改标签并加入新内容
    for item in data:
        item['文本'] = item.pop('text')
        item['风险类别'] = item.pop('riskType')
        item.pop('type_num')
        item.pop('index')
        new_data.append(item)

    # 将修改后的数据写入新的JSON文件
    with open(dst_file, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

def check():
    input_file = 'DataPipeline/output/dialog/table_new/p2_corr_select_new.csv'  # 替换为你的 CSV 文件路径

    # 打开 CSV 文件进行读取
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # 遍历每一行并检查项数
        for row_number, row in enumerate(reader, start=1):
            num_items = len(row)
            if num_items != 4:
                print(f"第 {row_number} 行的项数为 {num_items}")

def trans_to_json(csv_file, json_file):
    data = []
    with open(csv_file, 'r', encoding='utf-8') as csvf:
        csv_reader = csv.DictReader(csvf)
        for row in csv_reader:
            data.append(row)

    # 写入JSON文件
    with open(json_file, 'w', encoding='utf-8') as jsonf:
        json.dump(data, jsonf, indent=4,ensure_ascii=False)

    print(f'转换完成，JSON文件保存在 {json_file}')

if __name__ == '__main__':
    csv_file = "DataPipeline/output/dialog/selected_dataset final/p2_selected copy.csv"
    json_file = "DataPipeline/output/dialog/prompt2_8w_selected.json"
    reward_file = "DataPipeline/output/dialog/table_new/p2_selected_reward.json"
    # csv_to_json(csv_file, json_file)
    # trans_reward_json(json_file, reward_file)
    # print("完成")
    # trans_test_to_diversity()
    trans_to_json(csv_file, json_file)


