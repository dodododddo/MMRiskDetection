from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json
import argparse

def encoder_gen(tokenizer):
    def encode(examples):
        inputs = [x + y for x, y in zip(examples["instruction"], examples["input"])]
        targets = examples["output"]
        input_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        target_encodings = tokenizer(targets, truncation=True, padding="max_length", max_length=512)
        return {"input_ids": input_encodings.input_ids, "attention_mask": input_encodings.attention_mask, "labels": target_encodings.input_ids}
    return encode


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
        target_modules=["q_proj", "v_proj"]  # 要应用LoRA的目标模块
    )

    model = get_peft_model(model, peft_config)

    # 准备数据加载器
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 定义训练参数
    training_args = TrainingArguments(
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.001,
    )

    encode = encoder_gen(tokenizer)

    with open(train_dataset_path, 'r') as f:
        raw_data = json.load(f)
        data = {}
        data['instruction'] = [x['instruction'] for x in raw_data][:1000]
        data['input'] = [x['input'] for x in raw_data][:1000]
        data['output'] = [x['output'] for x in raw_data][:1000]
        train_dataset = Dataset.from_dict(data).map(encode, batched=True, batch_size=1000)
        
        
    with open(eval_dataset_path, 'r') as g:
        raw_data = json.load(g)
        data = {}
        data['instruction'] = [x['instruction'] for x in raw_data]
        data['input'] = [x['input'] for x in raw_data]
        data['output'] = [x['output'] for x in raw_data]
        eval_dataset = Dataset.from_dict(data).map(encode, batched=True)


    # 训练模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    # 保存微调后的模型
    model.save_pretrained(save_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and use a LoRA fine-tuned LLAMA3 model")
    parser.add_argument("--model_path", type=str, default='model/llama3-chat-chinese', help="Path to the pre-trained model")
    parser.add_argument("--train_dataset_path", type=str, default='dataset/finetune.json', help="Path to the training dataset")
    parser.add_argument("--eval_dataset_path", type=str, default='dataset/eval.json', help="Path to the evaluation dataset")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank parameter")
    parser.add_argument("--save_dir", type=str, default='finetuned_model/llama3-finetune')
    args = parser.parse_args()
    train(args.model_path, args.train_dataset_path, args.eval_dataset_path, args.save_dir, args.rank)