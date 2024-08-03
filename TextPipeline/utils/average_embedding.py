import torch
import numpy as np
from FlagEmbedding import FlagModel

# # 加载数据
# train_data = torch.load('embedding/data_train_classifier.pt')

# embeddings = train_data['embeddings']
# labels = train_data['labels']

# # 标签和对应的类别
# risk_label = [
#     '刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类', '贷款、代办信用卡类',
#     '虚假征信类', '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
#     '网络游戏产品虚假交易类', '网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
#     '网黑案件', '无风险'
# ]

# # 初始化用于存储平均值的张量
# mean_tensors = []

# # 对每个标签计算平均值
# for label in risk_label:
#     # 找到当前标签在列表中的索引
#     lab = torch.tensor(risk_label.index(label))
#     labels_np = labels.numpy()
#     indices = np.where(labels_np == lab.item())[0]
#     label_embeddings = []
#     for index in indices:
#         label_embeddings.append(embeddings[index])

#     # 使用 torch.stack() 将列表中的张量堆叠成一个张量
#     stacked_embeddings = torch.stack(label_embeddings)

#     # 计算平均张量
#     average_tensor = torch.mean(stacked_embeddings, dim=0)
    
#     # 存储平均值张量
#     mean_tensors.append(average_tensor)

# print("平均值张量数量:")
# print(len(mean_tensors))

# # 将结果转换为张量格式
# mean_tensors = torch.stack(mean_tensors)

# # 输出结果
# print("平均值张量:")
# print(mean_tensors)
# torch.save(mean_tensors, 'embedding/prototypes.pt')

prototypes = torch.load('embedding/prototypes.pt').cuda()
embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)
train_data = torch.load('./embedding/data_train_classifier.pt')
embeddings = train_data['embeddings'].cuda()
labels = train_data['labels'].cuda()

sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), prototypes, dim=2)
sim = embeddings @ prototypes.T
predict = torch.argmax(sim, dim=-1)
print(predict)
print(labels)
acc = (predict == labels).sum() / len(predict)
print(acc)

