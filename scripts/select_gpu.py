import subprocess
import json

def get_gpu_memory():
    """ 使用 nvidia-smi 获取显存使用情况 """
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    lines = result.stdout.decode('utf-8').strip().split('\n')
    
    gpu_memory = []
    for line in lines:
        index, free_memory = map(int, line.split(','))
        gpu_memory.append((index, free_memory))
    
    return gpu_memory

def get_free_gpu():
    """ 返回显存使用最少的 GPU 设备编号 """
    gpu_memory = get_gpu_memory()
    
    # 按空闲显存从大到小排序
    sorted_gpus = sorted(gpu_memory, key=lambda x: x[1], reverse=True)

    # 返回显存使用最少的 GPU
    # best_gpu = sorted_gpus[0][0]
    # return best_gpu
    return sorted_gpus

if __name__ == "__main__":
    gpu_id = get_free_gpu()
    gpu_id = [x[0] for x in gpu_id]
    print(json.dumps(gpu_id))
