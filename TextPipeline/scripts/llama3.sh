/data1/home/jrchen/anaconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model model/llama3-chat-chinese \
    --served-model-name llama3 \
    --enable_prefix_caching \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.6
