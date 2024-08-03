ROOT_DIRECTORY=/data1/home/jrchen/MMRiskDetection
CONDA_DIRECTORY=/data1/home/jrchen/anaconda3

cd $ROOT_DIRECTORY/AudioPipeline
CUDA_VISIBLE_DEVICES=1 $CONDA_DIRECTORY/envs/To_Text/bin/python $ROOT_DIRECTORY/AudioPipeline/server_for_to_text.py > $ROOT_DIRECTORY/server/9999.log &

cd $ROOT_DIRECTORY/AudioPipeline/DeepFake-Audio-Detection-MFCC
$CONDA_DIRECTORY/envs/Audio_CD/bin/python $ROOT_DIRECTORY/AudioPipeline/DeepFake-Audio-Detection-MFCC/server_for_audioanalyzer.py > $ROOT_DIRECTORY/server/9998.log &

cd $ROOT_DIRECTORY/FilePipeline
$CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/FilePipeline/server.py > $ROOT_DIRECTORY/server/6670.log &

cd $ROOT_DIRECTORY/ImagePipeline
CUDA_VISIBLE_DEVICES=1 $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/ImagePipeline/server.py > $ROOT_DIRECTORY/server/6666.log &

cd $ROOT_DIRECTORY/TextPipeline
$CONDA_DIRECTORY/bin/python  $ROOT_DIRECTORY/TextPipeline/mm_server.py > $ROOT_DIRECTORY/server/1111.log &

cd $ROOT_DIRECTORY/TextPipeline
$CONDA_DIRECTORY/bin/python  $ROOT_DIRECTORY/TextPipeline/ext_server.py > $ROOT_DIRECTORY/server/1930.log &

cd $ROOT_DIRECTORY/VideoPipeline
CUDA_VISIBLE_DEVICES=2 $CONDA_DIRECTORY/envs/videofact/bin/python $ROOT_DIRECTORY/VideoPipeline/server.py > $ROOT_DIRECTORY/server/1927.log &

$CONDA_DIRECTORY/envs/videofact/bin/python $ROOT_DIRECTORY/VideoPipeline/send_server.py > $ROOT_DIRECTORY/server/1928.log &

cd $ROOT_DIRECTORY/WebPipeline
$CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/WebPipeline/server.py > $ROOT_DIRECTORY/server/6667.log &

CUDA_VISIBLE_DEVICES=3 $CONDA_DIRECTORY/bin/python -m vllm.entrypoints.openai.api_server \
    --model $ROOT_DIRECTORY/TextPipeline/model/llama3-chat-chinese \
    --served-model-name llama3 \
    --enable_prefix_caching \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.4 > $ROOT_DIRECTORY/server/8000.log &

cd $ROOT_DIRECTORY/DigitalHumans/CodeFormer
CUDA_VISIBLE_DEVICES=3 $CONDA_DIRECTORY/envs/codeformer/bin/python $ROOT_DIRECTORY/DigitalHumans/CodeFormer/server_for_FaceRestorer.py > $ROOT_DIRECTORY/server/9997.log &

cd $ROOT_DIRECTORY/DigitalHumans/GPT-SoVITS
CUDA_VISIBLE_DEVICES=3 $CONDA_DIRECTORY/envs/GPTSoVits/bin/python $ROOT_DIRECTORY/DigitalHumans/GPT-SoVITS/GPT_SoVITS/sever_for_SSGen.py > $ROOT_DIRECTORY/server/9995.log &

cd $ROOT_DIRECTORY/DigitalHumans/V-Express
CUDA_VISIBLE_DEVICES=3 $CONDA_DIRECTORY/envs/V-Express/bin/python $ROOT_DIRECTORY/DigitalHumans/V-Express/server_for_V-Express.py > $ROOT_DIRECTORY/server/9996.log &

cd $ROOT_DIRECTORY/DigitalHumans/Facefusion
$CONDA_DIRECTORY/envs/facefusion/bin/python $ROOT_DIRECTORY/DigitalHumans/Facefusion/server.py > $ROOT_DIRECTORY/server/1929.log &

cd $ROOT_DIRECTORY/VideoPipeline/model/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection
$CONDA_DIRECTORY/envs/videofact/bin/python $ROOT_DIRECTORY/VideoPipeline/model/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/server.py > $ROOT_DIRECTORY/server/1932.log &

cd $ROOT_DIRECTORY/Frontend
$CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/sms_server.py > $ROOT_DIRECTORY/server/1931.log &

cd $ROOT_DIRECTORY
$CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/customer_app_demo.py > $ROOT_DIRECTORY/server/7862.log &

cd $ROOT_DIRECTORY/Frontend
$CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/business_app_demo.py > $ROOT_DIRECTORY/server/7763.log &

cd $ROOT_DIRECTORY/Frontend
$CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/MMRiskDetectionApp.py > $ROOT_DIRECTORY/server/5003.log &


