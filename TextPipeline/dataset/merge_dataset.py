import json
import random

def merge(risk_path, no_risk_path, dst_path):
    with open(risk_path, 'r') as f, open(no_risk_path, 'r') as g, open(dst_path, 'w') as h:
        risk_data = json.load(f)
        no_risk_data = json.load(g)
        data = risk_data + no_risk_data
        random.shuffle(data)
        json.dump(data, h, indent=4, ensure_ascii=False)
        
if __name__ == '__main__':
    risk_path = './dataset/risk.json'
    no_risk_path = './dataset/no_risk.json'
    dst_path = './dataset/test.json'
    merge(risk_path, no_risk_path, dst_path)