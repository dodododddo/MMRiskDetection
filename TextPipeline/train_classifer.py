import torch
from tqdm import tqdm
import random

class Block(torch.nn.Module):
    def __init__(self, idim, odim):
        super().__init__()
        self.fc1 = torch.nn.Linear(idim, 4 * idim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(4 * idim, odim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
        
class Classifier(torch.nn.Module):
    def __init__(self, dim, num_cls, n_layer=1):
        super().__init__()
        blocks = [Block(dim // (2 ** i), dim // (2 ** (i + 1))) for i in range(n_layer - 1)] 
        blocks.append(Block(dim // (2 ** (n_layer - 1)), num_cls))
        self.blocks = torch.nn.Sequential(*blocks)
        self.n_layer = n_layer
    
    def forward(self, x):
        x = self.blocks(x)
        return x
    
def train(batch_size=64, num_epochs=30, save_path='embedding/classifier.pt'):
   
    classifier = Classifier(1024, 13, n_layer=1).cuda()
    train_data = torch.load('./embedding/data_train_classifier.pt')
    embeddings = train_data['embeddings']
    labels = train_data['labels']
    
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {correct / len(embeddings):.4f}')        
        # acc = 65.7
    torch.save(classifier.state_dict(), save_path)
        
if __name__ == '__main__':
    train()
    
        