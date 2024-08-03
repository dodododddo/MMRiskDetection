#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 [image path] [repaired image path] [audio path] [pth path] [output_video_path]"
    exit 1
fi

IMAGE_PATH=$1
IMAGE_REPAIRED_PATH=$2
AUDIO_PATH=$3
PTH_PATH=$4
OUTPUT_VIDEO_PATH=$5

cd ../CodeFormer
echo "当前目录是: $(pwd)"
/data1/home/jrchen/anaconda3/envs/codeformer/bin/python /data1/home/jrchen/MMRiskDetection/AudioPipeline/CodeFormer/scripts/crop_align_face.py -i ./input_folder -o ./output_floder
echo "Success for step1"
/data1/home/jrchen/anaconda3/envs/codeformer/bin/python /data1/home/jrchen/MMRiskDetection/AudioPipeline/CodeFormer/inference_codeformer.py -w 0.5 --has_aligned --input_path "./output_folder/$IMAGE_PATH"
echo "Success for step2"

cp -r /data1/home/jrchen/MMRiskDetection/AudioPipeline/CodeFormer/results/test_img_0.5/restored_faces/* /data1/home/jrchen/MMRiskDetection/AudioPipeline/V-Express/test_samples/short_case/AOC/
echo "Success for step3:cp"

cd ../V-Express
echo "当前目录是: $(pwd)"
/data1/home/jrchen/anaconda3/envs/V-Express/bin/python /data1/home/jrchen/MMRiskDetection/AudioPipeline/V-Express/inference.py \
    --reference_image_path "./test_samples/short_case/AOC/$IMAGE_REPAIRED_PATH" \
    --audio_path "./test_samples/short_case/AOC/$AUDIO_PATH" \
    --kps_path "./test_samples/short_case/AOC/$PTH_PATH" \
    --output_path "./output/short_case/$OUTPUT_VIDEO_PATH" \
    --retarget_strategy "no_retarget" \
    --num_inference_steps 25
