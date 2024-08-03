import torch
from tqdm import tqdm
from FlagEmbedding import FlagModel
import random
import json
import os

label_cat = ['冒充电商物流客服类', '虚假网络投资理财类', '虚假信用服务类', 
                '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
                '网络婚恋、交友类', '冒充军警购物类诈骗', '网黑案件','无风险']

label_map = {x: i for i, x in enumerate(label_cat)}

class Block(torch.nn.Module):
    def __init__(self, idim, odim):
        super().__init__()
        self.fc1 = torch.nn.Linear(idim, 4 * idim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(4 * idim, odim)
        # self.fc = torch.nn.Linear(idim, odim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
        # return self.fc(x)
        
class Classifier(torch.nn.Module):
    def __init__(self, dim, num_cls, n_layer=12):
        super().__init__()
        blocks = [Block(dim // (2 ** i), dim // (2 ** (i + 1))) for i in range(n_layer - 1)] 
        blocks.append(Block(dim // (2 ** (n_layer - 1)), num_cls))
        self.blocks = torch.nn.Sequential(*blocks)
        self.n_layer = n_layer
    
    def forward(self, x):
        x = self.blocks(x)
        return x
    
def train(batch_size=64, num_epochs=100, save_path='embedding/classifier.pt'):
    
    classifier = Classifier(1024, 10, n_layer=1).cuda()
    # classifier.load_state_dict(torch.load(save_path))
    train_data = torch.load('./embedding/data_train_classifier_dialog.pt')
    test_data = torch.load('./embedding/data_eval_classifier_dialog.pt')
    embeddings = train_data['embeddings']
    # embeddings = torch.nn.functional.normalize(train_data['embeddings'], dim=-1)
    labels = train_data['labels']
    
    test_embeddings = test_data['embeddings']
    test_labels = test_data['labels']
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.00001, weight_decay=0.0005)
    for epoch in tqdm(range(num_epochs)):
        shuffle_idx = random.sample(list(range(len(labels))), len(labels))
        shuffle_embeddings = embeddings[shuffle_idx]
        shuffle_labels = labels[shuffle_idx]
        correct = 0
        for i in tqdm(range(0, len(embeddings), batch_size)):
            optimizer.zero_grad()
            batch_embeddings = shuffle_embeddings[i:i+batch_size].cuda()
            batch_labels = shuffle_labels[i:i+batch_size].cuda()
            outputs = classifier(batch_embeddings)
            loss = criterion(outputs,batch_labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Acc: {correct / len(embeddings):.4f}')  
        
        if (epoch + 1) % 10 == 0: 
            with torch.no_grad():
                correct = 0
                for i in tqdm(range(0, len(embeddings), batch_size)):
                    batch_embeddings = test_embeddings[i:i+batch_size].cuda()
                    batch_labels = test_labels[i:i+batch_size].cuda()
                    outputs = classifier(batch_embeddings)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == batch_labels).sum().item()
                print(f'Epoch [{epoch+1}/{num_epochs}], Test Acc: {correct / len(embeddings):.4f}')

        
    torch.save(classifier.state_dict(), save_path)
        
if __name__ == '__main__':
    if not os.path.exists('./embedding/data_train_classifier_dialog.pt'):
        with open('./dataset/dialog/sft_train.json') as f:
            data = json.load(f)
            texts = [x['messages'][1]['content'] for x in data]
            labels = [label_map[x['messages'][-1]['content']] for x in data]   
        
        embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",use_fp16=True)
        embeddings = embedding_model.encode(texts)
        embeddings = torch.tensor(embeddings).to(torch.float32)
        
        labels = torch.tensor(labels)
        data_dict = {}
        data_dict['embeddings'] = embeddings
        data_dict['labels'] = labels 
        torch.save(data_dict, './embedding/data_train_classifier_dialog.pt')
        
    if not os.path.exists('./embedding/data_eval_classifier_dialog.pt'):
        with open('./dataset/dialog/sft_eval.json') as g:
            data = json.load(g)
            texts = [x['messages'][1]['content'] for x in data]
            labels = [label_map[x['messages'][-1]['content']] for x in data]   
            
        embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",use_fp16=True)
        test_embeddings = embedding_model.encode(texts)
        test_embeddings = torch.tensor(test_embeddings).to(torch.float32)
        
        test_labels = torch.tensor(labels)
        data_dict = {}
        data_dict['embeddings'] = test_embeddings
        data_dict['labels'] = test_labels 
        torch.save(data_dict, './embedding/data_eval_classifier_dialog.pt')
        
    train()