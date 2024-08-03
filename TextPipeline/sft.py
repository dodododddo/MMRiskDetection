from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils import batch_iterator, process_string
from rag import get_nearest_indices
import json
import argparse
import torch
from FlagEmbedding import FlagModel

def train(model_path, train_dataset_path, eval_dataset_path, save_dir, rank):
    # 本地模型的路径

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')

    # 配置PEFT
    peft_config = LoraConfig(
        r=rank,  # LoRA的秩
        lora_alpha=32,  # LoRA的alpha参数
        lora_dropout=0.1,  # LoRA的dropout率
    )

    model = get_peft_model(model, peft_config)

    # 定义训练参数
    training_args = TrainingArguments(
        learning_rate=1e-5,
        output_dir='checkpoints/',
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.001,
    )
    

    with open(train_dataset_path, 'r') as f:
        raw_data = json.load(f)
        train_dataset = Dataset.from_list(raw_data)
        
        
    with open(eval_dataset_path, 'r') as g:
        raw_data = json.load(g)
        eval_dataset = Dataset.from_list(raw_data)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    # 保存微调后的模型
    model.save_pretrained(save_dir)
    
def preprocess(train_dataset_path, eval_dataset_path):
    with open(train_dataset_path, 'r') as f:
        train_dataset = json.load(f)
    
    with open(eval_dataset_path, 'r') as f:
        eval_dataset = json.load(f)
    
    
    train_data = []
    for x in tqdm(train_dataset):
        message = {}
        message['messages'] = []
        message['messages'].append({'role': 'system', 'content': x['instruction']})
        message['messages'].append({'role': 'user', 'content': x['input']})
        message['messages'].append({'role': 'assistant', 'content': x['output']})
        train_data.append(message)
        
    eval_data = []
    for x in tqdm(eval_dataset):
        message = {}
        message['messages'] = []
        message['messages'].append({'role': 'system', 'content': x['instruction']})
        message['messages'].append({'role': 'user', 'content': x['input']})
        message['messages'].append({'role': 'assistant', 'content': x['output']})
        eval_data.append(message)
        
    
    with open('dataset/sft_train.json', 'w') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False) 
        
    with open('dataset/sft_eval.json', 'w') as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False) 
    
def eval_lora(model_path, eval_dataset_path,adapter_path='checkpoints/checkpoint-10000'):
    with open(eval_dataset_path, 'r') as g:
        raw_data = json.load(g)[:4000]
        eval_dataset = Dataset.from_list(raw_data)
        
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    model = LLM(model_path, gpu_memory_utilization=0.4, enable_prefix_caching=True, enable_lora=True, max_lora_rank=64)
    sample_params = SamplingParams(temperature=1, top_p=0.9, max_tokens=128)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    
    tp = 0
    for batch in batch_iterator(eval_dataset, 64):
        texts = [tokenizer.apply_chat_template(
                x[:-1], add_generation_prompt=True, tokenize=False
            ) for x in batch['messages']]
        outputs = model.generate(texts, sample_params, lora_request=LoRARequest('classifier', 1, adapter_path))
        answers = [x.outputs[0].text for x in outputs]
        targets = [x[-1]['content'] for x in batch['messages']]
        for i in range(len(answers)):
            tp += answers[i] == targets[i]
        
    print(f'acc: {tp / len(eval_dataset):.4f}')


def eval_zero_shot(model_path, eval_dataset_path):
    with open(eval_dataset_path, 'r') as g:
        raw_data = json.load(g)[:4000]
        eval_dataset = Dataset.from_list(raw_data)
        
    model = LLM(model_path, gpu_memory_utilization=0.4, enable_prefix_caching=True)
    sample_params = SamplingParams(temperature=1, top_p=0.9, max_tokens=128)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    
    tp = 0
    for batch in batch_iterator(eval_dataset, 64):
        texts = [tokenizer.apply_chat_template(
                x[:-1], add_generation_prompt=True, tokenize=False
            ) for x in batch['messages']]
        outputs = model.generate(texts, sample_params)
        answers = [x.outputs[0].text for x in outputs]
        targets = [x[-1]['content'] for x in batch['messages']]
        for i in range(len(answers)):
            tp += process_string(answers[i]) == targets[i]
        
    print(f'acc: {tp / len(eval_dataset):.4f}')


def eval_rag(model_path, eval_dataset_path):
    with open(eval_dataset_path, 'r') as g:
        raw_data = json.load(g)[:4000]
        eval_dataset = Dataset.from_list(raw_data)
    
    embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)
    model = LLM(model_path, gpu_memory_utilization=0.4, enable_prefix_caching=True)
    sample_params = SamplingParams(temperature=1, top_p=0.9, max_tokens=128)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    ref_vectors = torch.load('embedding/represent_mean_top100_1200.pt')
    with open('dataset/represent_mean_top100_1200.json') as f:
        ref_docs = json.load(f)
        
    tp = 0
    for batch in batch_iterator(eval_dataset, 64):
        messages = [x[1]['content'] for x in batch['messages']]
        print(messages)
        embeddings = embedding_model.encode(messages)
        embeddings = torch.tensor(embeddings).to(torch.float32)
        indices = get_nearest_indices(embeddings, ref_vectors, 3)
        raw_examples = [[ref_docs[x] for x in list(indices[i])] for i in range(indices.shape[0])]
        examples = [" ".join([f'案例{i+1}: {example}' for i, example in enumerate(raw_example)]) for raw_example in raw_examples]
        for i, x in enumerate(batch['messages']):
            x[1]['content'] = ('\n你可参考以下相关案例:' + examples[i]) + x[1]['content']
            
        texts = [tokenizer.apply_chat_template(
                x[:-1], add_generation_prompt=True, tokenize=False
            ) for x in batch['messages']]
        outputs = model.generate(texts, sample_params)
        answers = [x.outputs[0].text for x in outputs]
        targets = [x[-1]['content'] for x in batch['messages']]
        for i in range(len(answers)):
            tp += process_string(answers[i]) == targets[i]
        
    print(f'acc: {tp / len(eval_dataset):.4f}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and use a LoRA fine-tuned LLAMA3 model")
    parser.add_argument("--model_path", type=str, default='model/llama3-chat-chinese', help="Path to the pre-trained model")
    parser.add_argument("--train_dataset_path", type=str, default='dataset/dialog/sft_train.json', help="Path to the training dataset")
    parser.add_argument("--eval_dataset_path", type=str, default='dataset/dialog/sft_eval.json', help="Path to the evaluation dataset")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank parameter")
    parser.add_argument("--save_dir", type=str, default='finetuned_model/llama3-finetune-dialog')
    args = parser.parse_args()
    # train(args.model_path, args.train_dataset_path, args.eval_dataset_path, args.save_dir, args.rank)
    eval_lora(args.model_path, args.eval_dataset_path)