import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class ClassifyDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.label_cat = [ '冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
                '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
                '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['文本']
        
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length = 100)
        labels = self.label_cat.index(item['风险类别'])
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(labels)
        }
        

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            self.data = [x['messages'] for x in self.data]
        self.tokenizer = tokenizer
        self.label_cat = [ '冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
                '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
                '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[1]['content']
        
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length = 100)
        labels = self.label_cat.index(item[-1]['content'])
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(labels)
        }

def train(train_dataset_path):

    # Step 1: Load and preprocess data

    # Step 2: Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('./model/ernie-3.0-xbase-zh')
    model = AutoModelForSequenceClassification.from_pretrained('model/ernie-3.0-xbase-zh', num_labels=10)  # Adjust num_labels according to your categories

    # Step 3: Prepare dataset and dataloader
    # train_dataset = ClassifyDataset('../DataPipeline/output/message/message_finetuning11.json', tokenizer)
    train_dataset = SFTDataset(train_dataset_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Step 4: Training loop
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'gpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.005)
    loss_np = np.array([])
    right_np = np.array([])
    right = 0
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            for i in range(len(labels)):
                predicted_probabilities = outputs.logits[i].softmax(dim=-1)  # 对 logits 进行 softmax 获取概率分布
                predicted_class = predicted_probabilities.argmax().item()  # 获取概率最大的类别索引
                if predicted_class == labels[i].item():
                    right += 1
                right_np = np.append(right_np, right)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_np = np.append(loss_np, loss.cpu().detach().numpy())
            
        # Validation (optional): Evaluate your model's performance on a validation set here

    # Step 5: Save the trained model
    model.save_pretrained('finetuned_model/ernie-3.0-xbase-zh-finetune-dialog')

if __name__ == '__main__':
    train('dataset/dialog/sft_train.json')