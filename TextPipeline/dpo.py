import json
from trl import  DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from datasets import Dataset

max_seq_length = 512 # Supports automatic RoPE Scaling, so choose any number.

with open('dataset/dialog/dpo_new.json') as f:
    train_data = json.load(f)
    

    
    
train_dataset = Dataset.from_list(train_data)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "model/llama3-chat-chinese",
    max_seq_length = max_seq_length,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, 
    bias = "none", 
    use_gradient_checkpointing = True,
    random_state = 3407
)

training_args = DPOConfig(
    output_dir="./output",
    beta=0.1
)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    beta=0.1,
    args=training_args,
    max_prompt_length=max_seq_length,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()