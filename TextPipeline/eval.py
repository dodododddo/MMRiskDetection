from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import pipeline

if __name__ == '__main__':
    
    model_path = './llama3-chat-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
    
    src_text = "<短信文本>尊敬的用户，您购买的商品已发货，但物流显示异常，需点击链接查看详情。<短信文本>"
    question = "简述以上短信中可能存在的风险。"
    
    pipe = pipeline(model, tokenizer)
    answer = pipe(src_text + question)
    print('\n' + answer)