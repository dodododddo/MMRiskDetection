/data1/home/jrchen/anaconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model model/qwen2-awq \
    --served-model-name qwen2 \
    --enable_prefix_caching \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --gpu_memory_utilization 0.2
