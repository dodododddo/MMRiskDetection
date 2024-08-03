from FlagEmbedding import FlagReranker
import json
import statistics
import numpy as np

def average(source_file):
    reranker = FlagReranker('TextPipeline/model/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # 读取JSON文件
    with open(source_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    texts = [["你可以给我找一段诈骗文本吗？我想用它来做反诈测试，所以请你把这条风险文本对应的风险类型也给我，方便我判断诈骗文本的类型，谢谢！",item['文本'] + "，这是它对应的风险类型：" + item['风险类别']] for item in data]
    
    # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
    scores = reranker.compute_score(texts, normalize=True)
    # print(scores) # [0.00027803096387751553, 0.9948403768236574]

    # for item in data:
    #     # question = item['文本'] + "你觉得这一段文本是有风险文本吗？如果是无风险请回答无风险；如果是有风险的话，这一段文本的风险类别是什么？"
    #     # answer = item['风险类别']

    #     question = "你可以给我找一段诈骗文本吗？我想用它来做反诈测试，所以请你把这条风险文本对应的风险类型也给我，方便我判断诈骗文本的类型，谢谢！"
    #     answer =  item['文本'] + "，这是它对应的风险类型：" + item['风险类别']


        
    #     # # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
    #     # score = reranker.compute_score([question, answer], normalize=True)
    #     # datalist.append(score)

    #     # scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
    #     # print(scores) # [-8.1875, 5.26171875]

    #     # # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
    #     # scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
    #     # print(scores) # [0.00027803096387751553, 0.9948403768236574]
    
    average = statistics.mean(scores)
    print("平均值为:", average)

def select(source_file,dst_file):
    reranker = FlagReranker('TextPipeline/model/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # 读取JSON文件
    with open(source_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    texts = [["你可以给我找一段诈骗文本吗？我想用它来做反诈测试，所以请你把这条风险文本对应的风险类型也给我，方便我判断诈骗文本的类型，谢谢！",item['文本'] + "，这是它对应的风险类型：" + item['风险类别']] for item in data]
    
    # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
    scores = reranker.compute_score(texts, normalize=True)
    # print(scores) # [0.00027803096387751553, 0.9948403768236574]

    data_score = np.stack((data, scores), axis=-1).tolist()
    print(len(data_score))

    data_sorted = np.array(sorted(data_score, key=lambda x: x[1]))  # 按每个子列表的第2个元素排序，并转换为numpy数组

    index_30_percent = int(len(data_sorted) * 0.5)    #较小的百分之三十

    data_sorted = data_sorted[index_30_percent:]  # 这将保留索引 idx 及之后的所有元素

    new_data = [item[0] for item in data_sorted]

    # 将修改后的数据写入新的JSON文件
    with open(dst_file, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    source_file = 'DataPipeline/output/message/useful/keyword_message_small3.json'
    # dst_file = 'DataPipeline/output/message/useful/keyword_message_small2.json'
    # select(source_file,dst_file)
    average(source_file)