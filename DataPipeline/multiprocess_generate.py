from utils.pipeline import GenPipeline
import asyncio
import time
import aiohttp
import concurrent.futures

def main(pipe):
    n = 500
    texts = ['请写五行五句诗'] * 500
    prompt = '你是一个诗人'
    
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(pipe.generate, text, prompt) for text in texts]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
        
        
    # tasks = [asyncio.create_task(pipe.async_generate(session, text, prompt)) for session, text in zip(sessions,texts)]
    # tasks = [asyncio.create_task(pipe.generate(text, prompt)) for text in texts]
    # for task in asyncio.as_completed(tasks):
    #     result = await task
    #     print(result)
        
    # resps = await asyncio.gather(*fus)
    # for resp in resps:
    #     print(resp)
    end_time = time.time()
    print(f'time: {end_time - start_time}')
    

    
if __name__ == '__main__':
    pipe = GenPipeline('sk-bbcc3b66bde14e21b1c39af3c0da74c0')
    # asyncio.run(main(pipe))
    main(pipe)