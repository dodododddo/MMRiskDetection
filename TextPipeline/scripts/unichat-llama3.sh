CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model model/unichat-llama3 \
    --served-model-name unichat \
    --enable_prefix_caching \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.4
