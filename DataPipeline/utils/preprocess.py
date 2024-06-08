import csv
import json
import re
import random

def preprocess(file_path, dst_path):
    dst = []
    # 读取CSV文件
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # 输出短信内容
            d = {}
            text = row['短信内容']
            if '【' in text:
                text = text[text.find('】') + 1:]
            d['文本'] = text
            d['风险点'] = "无"
            d['风险类别'] = "无风险"
        
            dst.append(d)
        
    with open(dst_path, 'w') as g:
        json.dump(dst, g, indent=4, ensure_ascii=False)
        
def replace_x_with_random(text):
    
    x_indices = [match.start() for match in re.finditer('x', text, re.IGNORECASE)]
    
    
    replaced_text = list(text)
    for index in x_indices:
        replaced_text[index] = str(random.randint(0, 9))

    return ''.join(replaced_text)
        
if __name__ == '__main__':
    # 文件路径
    file_path = 'test.csv'
    dst_path = 'no_risk.json'
    preprocess(file_path, dst_path)
