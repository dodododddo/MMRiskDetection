import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import pipeline
from FlagEmbedding import FlagModel
from dataclasses import dataclass
from vllm import LLM, SamplingParams


@dataclass
class VllmConfig:
    gpu_num: int = 4
    gpu_utilization:float = 0.2
    prefix_cache: bool = True
    

class LLMWrapper:
    def __init__(self, model: LLM | AutoModelForCausalLM | str, tokenizer=None, system_prompt=None, use_vllm=False, vllm_config:VllmConfig=None):
        if isinstance(model, str):
            if use_vllm:
                self.model = LLM(model, tensor_parallel_size=vllm_config.gpu_num, 
                             gpu_memory_utilization=vllm_config.gpu_utilization, enable_prefix_caching=vllm_config.prefix_cache) 
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model, device_map='auto')
        else:
            self.model = model
            if tokenizer == None:
                raise ValueError('No valid tokenizer')
            
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model, padding_side='left')
        self.system_prompt = system_prompt
        self.use_vllm = use_vllm or isinstance(model, LLM)
        
    def preprocess(self, input_text, instruct):
        if isinstance(input_text, list):
            messages = input_text
        
        elif instruct == None:
            if self.system_prompt:
                messages = [{'role': 'system', 'content': self.system_prompt},
                            {'role': 'user', 'content': input_text}]
            else:
                messages = [{'role': 'user', 'content': input_text}]
        
        else:
            messages = [{'role': 'system', 'content': instruct},
               {'role': 'user', 'content': input_text}]
        
        return messages
        
    def generate(self, input_text, instruct, temperature, top_p, max_new_token):
        messages = self.preprocess(input_text, instruct)
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_token,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0][input_ids.shape[-1]:]
        # response = [outputs[i][input_ids.shape[i, -1]:] for i in range(len(outputs))]
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def batch_generate(self, input_texts, instructs, temperature, top_p, max_new_token):
        if isinstance(instructs, list):
            messages = [self.preprocess(input_text, instruct) for input_text, instruct in zip(input_texts, instructs)]
        else:
            messages = [self.preprocess(input_text, instructs) for input_text in input_texts]
        
        texts = [self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        ) for message in messages]
        
        # max_len = max([len(x) for x in input_ids])
        # pad_token_id = self.tokenizer.pad_token_id
        # padding_input_ids = [[pad_token_id] * (max_len - len(x)) + x for x in input_ids]
        # padding_input_ids = torch.tensor(padding_input_ids).to(self.model.device)
        
        input_ids = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_token,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        responses= [outputs[i][input_ids['input_ids'][i].shape[-1]:] for i in range(len(outputs))]
        return [self.tokenizer.decode(response, skip_special_tokens=True) for response in responses]
    
    def generate_vllm(self, input_text, instruct, temperature, top_p, max_new_token):
        sample_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_token)
        messages = self.preprocess(input_text, instruct)
        messages = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        output = self.model.generate(messages, sample_params)
        return output[0].outputs[0].text
    
    def batch_generate_vllm(self, input_texts, instructs, temperature, top_p, max_new_token):
        sample_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_token)
        if isinstance(instructs, list):
            messages = [self.preprocess(input_text, instruct) for input_text, instruct in zip(input_texts, instructs)]
        else:
            messages = [self.preprocess(input_text, instructs) for input_text in input_texts]
        messages = [self.tokenizer.apply_chat_template(message, add_generation_prompt=True,tokenize=False) for message in messages]
        output = self.model.generate(messages, sample_params)
        return [x.outputs[0].text for x in output]
    
    def __call__(self, input_text: str | list, instruct: str | list | None = None, max_new_token=1024, temperature=0.6, top_p=0.9)->str | list:
        if isinstance(input_text, str):
            return self.generate_vllm(input_text, instruct, temperature, top_p, max_new_token) if self.use_vllm else self.generate(input_text, instruct, temperature, top_p, max_new_token)
        elif isinstance(input_text, list):
            return self.batch_generate_vllm(input_text, instruct, temperature, top_p, max_new_token) if self.use_vllm else self.batch_generate(input_text, instruct, temperature, top_p, max_new_token)
        else:
            raise ValueError('input_text should be string or list.')
        
@dataclass
class RAGResp:
    answer: str
    ref_examples: str | list
    
    
class RAGPipeline:
    def __init__(self, llm, embedding_model=None, reranker=None, ref_vectors=None, ref_docs=None):
        self.llm = llm
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.ref_vectors = ref_vectors
        self.ref_docs = ref_docs
    
        
    def generate(self, input_text, instruct, max_new_token, temperature, top_p, use_rag, use_reranker, ref_vectors, ref_docs, k):
        if not use_rag or self.embedding_model == None:
            return RAGResp(self.llm(input_text, instruct, max_new_token, temperature, top_p),'无')
               
        embedding = self.embedding_model.encode(input_text)
        embedding = torch.tensor(embedding).to(torch.float32)
        if not use_reranker:
            indices = get_nearest_indices(embedding, ref_vectors if ref_vectors else self.ref_vectors, k)
            raw_examples = [ref_docs[x] for x in list(indices)] if ref_docs else [self.ref_docs[x] for x in list(indices)]
            examples = " ".join([f'案例{i}: {example}' for i, example in enumerate(raw_examples)])
            user_text = examples + '短信文本: ' + input_text
            return RAGResp(self.llm(user_text, instruct, max_new_token, temperature, top_p),examples)         
        else:
            raise NotImplemented('数据量较小，暂时不需要重排器')
        
    def batch_generate(self, input_texts, instructs, max_new_token, temperature, top_p, use_rag, use_reranker, ref_vectors, ref_docs, k):
        if not use_rag or self.embedding_model == None:
            return [RAGResp(x,None) for x in self.llm(input_texts, instructs, max_new_token, temperature, top_p)]
            
        embeddings = self.embedding_model.encode(input_texts)
        embeddings = torch.tensor(embeddings).to(torch.float32)
        if not use_reranker:
            indices = get_nearest_indices(embeddings, ref_vectors if ref_vectors else self.ref_vectors, k)
            raw_examples = [[ref_docs[x] for x in list(indices[i])] if ref_docs else [self.ref_docs[x] for x in list(indices[i])] for i in range(indices.shape[0])]
            examples = [" ".join([f'案例{i+1}: {example}' for i, example in enumerate(raw_example)]) for raw_example in raw_examples]
            user_texts = [example + '短信文本: ' + input_text for example, input_text in zip(examples, input_texts)]
            answers = self.llm(user_texts, instructs, max_new_token, temperature, top_p)
            return [RAGResp(answer, example) for answer, example in zip(answers, examples)]    
        else:
            raise NotImplemented('数据量较小，暂时不需要重排器')
        
    def __call__(self, input_text:str | list, instruct: str | list | None = None, max_new_token=8192, temperature=0.6, top_p=0.9, use_rag=True, use_reranker=False, ref_vectors=None, ref_docs=None, k=3)-> RAGResp | list[RAGResp]:
        if isinstance(input_text, str):
            return self.generate(input_text, instruct, max_new_token, temperature, top_p, use_rag, use_reranker, ref_vectors, ref_docs, k)
        elif isinstance(input_text, list):
            return self.batch_generate(input_text, instruct, max_new_token, temperature, top_p, use_rag, use_reranker, ref_vectors, ref_docs, k)
        else:
            raise ValueError('input_text should be string or list.')

def get_nearest_indices(embeddings, ref_vectors, k)->torch.Tensor:
    if len(embeddings.shape) == 2:
        cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), ref_vectors, dim=2)
    elif len(embeddings.shape) == 1:
        cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(0), ref_vectors, dim=1)
    _, top_k_indices = torch.topk(cos_sim, k)
    return top_k_indices

if __name__ == '__main__':
    test_path = "./dataset/test.json"
    with open(test_path, 'r') as f:
        all_data = json.load(f)
        

    risk_label = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类','贷款、代办信用卡类',
                    '虚假征信类','虚假购物、服务类','冒充公检法及政府机关类', '冒充领导、熟人类',
                    '网络游戏产品虚假交易类','网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类诈骗',
                    '网黑案件', '无风险']
        
    prompt = f'''你是一个风险判断的专家，你将接受一些诈骗案件示例作为参考，然后接受一个短信文本，请按输出格式完成以下两个任务：
                    输出格式：{{'风险类别':'...', '风险点':'...'}}
                    1. 风险类型包括{str(risk_label)}，若认为短信是无风险文本则风险类型为无风险。
                    2. 风险点可用短语概括，若可用原文中词句可直接使用
                    注意：除该字典外不要输出任何其他内容，确保你仅仅输出一个字典'''
                    
    ref_vectors = torch.load('embedding/represent_mean_top100_1200.pth')
    with open('dataset/represent_mean_top100_1200.json') as f:
        ref_docs = json.load(f)

    datas = [str(x['文本']) for x in all_data[:5]]
    model_path = './model/llama3-chat-chinese'
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #         model_path, device_map="auto", 
    #     )
    # model = LLM(model_path, tensor_parallel_size=4, gpu_memory_utilization=0.15)
    
    embedding_model = FlagModel("./model/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)
    
    llm = LLMWrapper(model_path, use_vllm=False)
    rag_pipe = RAGPipeline(llm, embedding_model, None, ref_vectors, ref_docs)
    
    for idx, data in enumerate(datas):
        resp = rag_pipe(data, prompt)
        print('\n')
        print(f'原始文本: {all_data[idx]}')
        print(f'模型回复: {resp.answer}')
        print(f'参考案例: {resp.ref_examples}')
    # print(llm(['我爱你', '写一首诗']))