import json

# 读取JSON文件
with open('DataPipeline/output/dialog/test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建一个新的数据列表
new_data = []

# 遍历原数据，修改标签并加入新内容
for item in data:
    item['instruction'] = '''你是一个风险判断的专家，你将接受一些诈骗案件示例作为参考，然后接受一个对话文本，请按输出格式完成以下两个任务：
                    输出格式：{{'风险类别':'...', '风险点':'...'}}
                    1. 风险类型包括{'刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类','虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类','网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗','网黑案件', '无风险'}，若认为短信是无风险文本则风险类型为无风险。
                    2. 风险点可用短语概括，若可用原文中词句可直接使用
                    注意：除该字典外不要输出任何其他内容，确保你仅仅输出一个字典'''
    if '文本' in item:
        item['input'] = item.pop('文本')  
    if '风险点' and '风险类别' in item:
        item['output'] = "{'风险类别':" + item['风险类别'] + ",'风险点':" + item['风险点'] + '}'
        item.pop('风险点')
        item.pop('风险类别')    
    new_data.append(item)

# 将修改后的数据写入新的JSON文件
with open('DataPipeline/output/dialog/dialog_risk.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, indent=4, ensure_ascii=False)