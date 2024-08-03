from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm import tqdm


label_cat = ['冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
                '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
                '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险']

def batch_iterator(dataset, batch_size):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset[i:i + batch_size]
        
def test(test_path='./dataset/dialog/sft_eval.json', num_sample=4000):
    model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model/ernie-3.0-xbase-zh-finetune-dialog", num_labels=10)
    tokenizer = AutoTokenizer.from_pretrained("model/ernie-3.0-xbase-zh")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_file = json.load(f)[:num_sample]

    correct = 0
    print(f'use sample: {len(test_file)}')
    for items in batch_iterator(test_file, 128):
        texts = [item['messages'][1]['content'] for item in items]
        data = tokenizer(texts, padding=True, return_tensors="pt")
        outputs = model(**data)
        logits = outputs.logits

        # 获取预测结果
        predicted_classes = [label_cat[logit.argmax().item()] for logit in logits]# 获取概率最大的类别索引
        labels = [item['messages'][-1]['content'] for item in items]
        for label, predicted_class in zip(labels, predicted_classes):
            correct += (label == predicted_class)

    print(f'acc: {correct / len(test_file)}')
    
def test_bin(test_path='./dataset/message/sft_eval.json', num_sample=4000):
    model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model/ernie-3.0-xbase-zh-finetune-message", num_labels=10)
    tokenizer = AutoTokenizer.from_pretrained("model/ernie-3.0-xbase-zh")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_file = json.load(f)[:num_sample]

    correct = 0
    print(f'use sample: {len(test_file)}')
    for items in batch_iterator(test_file, 16):
        texts = [item['messages'][1]['content'] for item in items]
        data = tokenizer(texts, padding=True, return_tensors="pt")
        outputs = model(**data)
        logits = outputs.logits

        # 获取预测结果
        predicted_classes = [label_cat[logit.argmax().item()] for logit in logits]# 获取概率最大的类别索引
        labels = [item['messages'][-1]['content'] for item in items]
        for label, predicted_class in zip(labels, predicted_classes):
            correct += (label == predicted_class) if predicted_class == '无风险' else label != '无风险' 

    print(f'acc: {correct / len(test_file)}')
    
if __name__ == '__main__':
    test_bin()

