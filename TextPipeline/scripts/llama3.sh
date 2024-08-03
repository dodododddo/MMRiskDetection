# CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
#     --model model/llama3-chat-chinese \
#     --enable-torch-compile \
#     --dtype auto \
#     --host 0.0.0.0 \
#     --port 8000 \

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model model/llama3-chat-chinese \
    --served-model-name llama3 \
    --enable_prefix_caching \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.6
