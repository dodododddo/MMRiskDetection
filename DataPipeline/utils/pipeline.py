import asyncio
import aiohttp

def pipeline(client, temperature=1.25):
    def generate(input_text):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=input_text,
            stream=False,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    return generate

def apipeline(aclient, temperature=1.25):
    async def generate(input_text):
        response = await aclient.chat.completions.create(
            model="deepseek-chat",
            messages=input_text,
            stream=False,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    async def batch_generate(input_texts):
        results = await asyncio.gather(*(generate(input_text) for input_text in input_texts))
        print(results)
        return results
    
    return batch_generate

def async_pipeline(model, base_url):
    async def generate(input_text, temperature=1.25):
        # 准备请求数据
        payload = {
            "model": model,
            "prompt": input_text,
            "max_tokens": 200,
            "temperature": temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0.6
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, json=payload) as response:
                response_data = await response.json()
                return response_data
    
    async def batch_generate(input_texts):
        tasks = [asyncio.create_task(generate(input_text)) for input_text in input_texts]
        results = await asyncio.gather(*tasks)
        return results
    
    return batch_generate


