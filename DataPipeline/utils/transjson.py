import csv
import json

# 函数：读取CSV文件并返回字典列表
def read_csv(file_path):
    data = []
    with open(file_path, newline='',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# 函数：将CSV文件转换为JSON格式并保存到文件
def csv_to_json(csv_file, json_file):
    data = read_csv(csv_file)
    with open(json_file, 'w',encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4,ensure_ascii=False)


def trans_json1():
    # 读取JSON文件
    with open('DataPipeline/output/message/useful/keyword_message_small2.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 创建一个新的数据列表
    new_data = []

    instruction = '''你是一个风险判断的专家，你将接受一段对话文本，请给出该对话中存在的风险类型：
                    注意点：
                    1. 风险类型包括；'冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
                    '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类','网络婚恋、交友类', 
                    '冒充军警购物类诈骗', '网黑案件','无风险'，若认为短信是无风险文本则风险类型为无风险。
                    2.只需要输出风险类型即可，不需要添加其它多余符号。
                    输出格式：...'''
    # 遍历原数据，修改标签并加入新内容
    for item in data:
        message = {}
        message['messages'] = []
        message['messages'].append({'role': 'system', 'content': instruction})
        message['messages'].append({'role': 'user', 'content': item['文本']})
        message['messages'].append({'role': 'assistant', 'content': item['风险类别']})
    # #     # item['instruction'] = '''你是一个风险判断的专家，你将接受一个短信文本，请给出短信文本的风险类别和风险点：
    # #     #             注意点：
    # #     #             1. 风险类型包括{['冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
    # #     #             '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类','网络婚恋、交友类', 
    # #     #             '冒充军警购物类诈骗', '网黑案件','无风险']}，若认为短信是无风险文本则风险类型为无风险。
    # #     #             2.请严格按照输出格式来输出，不要增加任何多余内容
    # #     #             输出格式：{...}'''
    # #     # if '文本' in item:
    # #     #     item['input'] = item.pop('文本')  
    # #     # if '风险点' and '风险类别' in item:
    # #     #     item['output'] = "'风险类别':" + item['风险类别']
    # #     #     if item['风险类别'] == '冒充电商物流客服类':
    # #     #         item['label'] = 0
    # #     #     if item['风险类别'] == '虚假网络投资理财类':
    # #     #         item['label'] = 1
    # #     #     if item['风险类别'] == '虚假信用服务类':
    # #     #         item['label'] = 2
    # #     #     if item['风险类别'] == '虚假购物、服务类':
    # #     #         item['label'] = 3
    # #     #     if item['风险类别'] == '冒充公检法及政府机关类':
    # #     #         item['label'] = 4
    # #     #     if item['风险类别'] == '冒充领导、熟人类':
    # #     #         item['label'] = 5
    # #     #     if item['风险类别'] == '网络婚恋、交友类':
    # #     #         item['label'] = 6
    # #     #     if item['风险类别'] == '冒充军警购物类诈骗':
    # #     #         item['label'] = 7
    # #     #     if item['风险类别'] == '网黑案件':
    # #     #         item['label'] = 8
    # #     #     if item['风险类别'] == '无风险':
    # #     #         item['label'] = 9
    # #     #     item.pop('案件编号')
    # #     #     item.pop('风险类别')    
        new_data.append(message)

            

    # 将修改后的数据写入新的JSON文件
    with open('DataPipeline/output/message/useful/keyword_message_small3.json', 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

    
def trans_json2(source_file, dst_file):
    # 读取JSON文件
    with open(source_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 创建一个新的数据列表
    new_data = []

    instruction = '''你是一个风险判断的专家，你将接受一段对话文本，请给出该对话中存在的风险类型：
                    注意点：
                    1. 风险类型包括；'冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
                    '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类','网络婚恋、交友类', 
                    '冒充军警购物类诈骗', '网黑案件','无风险'，若认为短信是无风险文本则风险类型为无风险。
                    2.只需要输出风险类型即可，不需要添加其它多余符号。
                    输出格式：...'''
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

# 主程序入口
if __name__ == "__main__":
    # csv_file = 'DataPipeline/output/message/keyword_message_small.csv'   # CSV文件路径
    # json_file = 'DataPipeline/output/message/useful/keyword_message_small1.json' # JSON文件路径
    # csv_to_json(csv_file, json_file)
    # print(f'CSV文件 {csv_file} 已成功转换为JSON文件 {json_file}')
    # source_file = 'DataPipeline/output/message/useful/keyword_message_small1.json'
    # dst_file = 'DataPipeline/output/message/useful/keyword_message_small1.json'
    # trans_json2(source_file,dst_file)
    trans_json1()
    
    