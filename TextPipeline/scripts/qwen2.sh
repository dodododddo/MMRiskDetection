CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model model/qwen2-7B-instruct \
    --served-model-name qwen2 \
    --enable_prefix_caching \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.6
