ROOT_DIRECTORY="/data1/home/jrchen/MMRiskDetection"
CONDA_DIRECTORY="/data1/home/jrchen/anaconda3"

declare -A SERVICES
SERVICES=(
    ["1927"]="$ROOT_DIRECTORY/VideoPipeline $CONDA_DIRECTORY/envs/videofact/bin/python $ROOT_DIRECTORY/VideoPipeline/server.py /tmp/1927.pid $ROOT_DIRECTORY/server/1927.log"
    ["8000"]="$ROOT_DIRECTORY/TextPipeline $CONDA_DIRECTORY/bin/python $ROOT_DIRECTORY/TextPipeline/scripts/start.py /tmp/8000.pid $ROOT_DIRECTORY/server/8000.log"
    ["1111"]="$ROOT_DIRECTORY/TextPipeline $CONDA_DIRECTORY/bin/python $ROOT_DIRECTORY/TextPipeline/mm_server.py /tmp/1111.pid $ROOT_DIRECTORY/server/1111.log"
    ["1928"]="$ROOT_DIRECTORY/VideoPipeline $CONDA_DIRECTORY/envs/videofact/bin/python $ROOT_DIRECTORY/VideoPipeline/send_server.py /tmp/1928.pid $ROOT_DIRECTORY/server/1928.log" 
    ["1929"]="$ROOT_DIRECTORY/DigitalHumans/Facefusion $CONDA_DIRECTORY/envs/facefusion/bin/python $ROOT_DIRECTORY/DigitalHumans/Facefusion/server.py /tmp/1929.pid $ROOT_DIRECTORY/server/1929.log"
    ["1930"]="$ROOT_DIRECTORY/TextPipeline $CONDA_DIRECTORY/bin/python $ROOT_DIRECTORY/TextPipeline/ext_server.py /tmp/1930.pid $ROOT_DIRECTORY/server/1930.log"
    ["1931"]="$ROOT_DIRECTORY/Frontend $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/sms_server.py /tmp/1931.pid $ROOT_DIRECTORY/server/1931.log"
    ["1932"]="$ROOT_DIRECTORY/VideoPipeline/model/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection $CONDA_DIRECTORY/envs/videofact/bin/python $ROOT_DIRECTORY/VideoPipeline/model/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/server.py /tmp/1932.pid $ROOT_DIRECTORY/server/1932.log"
    ["6666"]="$ROOT_DIRECTORY/ImagePipeline $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/ImagePipeline/server.py /tmp/6666.pid $ROOT_DIRECTORY/server/6666.log"
    ["6667"]="$ROOT_DIRECTORY/WebPipeline $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/WebPipeline/server.py /tmp/6667.pid $ROOT_DIRECTORY/server/6667.log"
    ["6670"]="$ROOT_DIRECTORY/FilePipeline $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/FilePipeline/server.py /tmp/6670.pid $ROOT_DIRECTORY/server/6670.log"
    ["9995"]="$ROOT_DIRECTORY/DigitalHumans/GPT-SoVITS $CONDA_DIRECTORY/envs/GPTSoVits/bin/python $ROOT_DIRECTORY/DigitalHumans/GPT-SoVITS/GPT_SoVITS/sever_for_SSGen.py /tmp/9995.pid $ROOT_DIRECTORY/server/9995.log"
    ["9996"]="$ROOT_DIRECTORY/DigitalHumans/V-Express $CONDA_DIRECTORY/envs/V-Express/bin/python $ROOT_DIRECTORY/DigitalHumans/V-Express/server_for_V-Express.py /tmp/9996.pid $ROOT_DIRECTORY/server/9996.log"
    ["9997"]="$ROOT_DIRECTORY/DigitalHumans/CodeFormer $CONDA_DIRECTORY/envs/codeformer/bin/python $ROOT_DIRECTORY/DigitalHumans/CodeFormer/server_for_FaceRestorer.py /tmp/9997.pid $ROOT_DIRECTORY/server/9997.log"
    ["9998"]="$ROOT_DIRECTORY/AudioPipeline/DeepFake-Audio-Detection-MFCC $CONDA_DIRECTORY/envs/Audio_CD/bin/python $ROOT_DIRECTORY/AudioPipeline/DeepFake-Audio-Detection-MFCC/server_for_audioanalyzer.py /tmp/9998.pid $ROOT_DIRECTORY/server/9998.log"
    ["9999"]="$ROOT_DIRECTORY/AudioPipeline $CONDA_DIRECTORY/envs/To_Text/bin/python $ROOT_DIRECTORY/AudioPipeline/server_for_to_text.py /tmp/9999.pid $ROOT_DIRECTORY/server/9999.log"
    ["7763"]="$ROOT_DIRECTORY/Frontend $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/business_app_demo.py /tmp/7763.pid $ROOT_DIRECTORY/server/7763.log"
    ["7862"]="$ROOT_DIRECTORY $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/customer_app_demo.py /tmp/7862.pid $ROOT_DIRECTORY/server/7862.log"
    ["5003"]="$ROOT_DIRECTORY/Frontend $CONDA_DIRECTORY/envs/image/bin/python $ROOT_DIRECTORY/Frontend/utils/MMRiskDetectionApp.py /tmp/5003.pid $ROOT_DIRECTORY/server/5003.log"
)

start_service() {
    local service_name=$1
    local service_info=($2)
    local root_dir=${service_info[0]}
    local python_env=${service_info[1]}
    local script_path=${service_info[2]}
    local pid_file=${service_info[3]}
    local log_file=${service_info[4]}

    local gpu_id=$(python $ROOT_DIRECTORY/scripts/select_gpu.py)

    gpu_ids=$(python $ROOT_DIRECTORY/scripts/select_gpu.py)

    gpu_id_list=($(echo $gpu_ids | jq -r '.[]'))
    
    echo "Starting $service_name..."
    cd $root_dir
    if [ "$service_name" -eq 8000 ]; then
        eval "CUDA_VISIBLE_DEVICES=${gpu_id_list[1]} $python_env $script_path > $log_file 2>&1 &"
        sleep 30
        local pid=$(lsof -ti :$service_name)
        if [ -z "$pid" ]; then
            echo "Failed to find process for port $service_name."
            return 1
        fi

        echo "Process ID for port $service_name is $pid."
        echo $pid > $pid_file
    elif [ "$service_name" -eq 9999 ]; then
        eval "CUDA_VISIBLE_DEVICES=${gpu_id_list[2]} $python_env $script_path > $log_file 2>&1 &"
        echo $! > $pid_file
    elif [ "$service_name" -eq 1927 ]; then
        eval "CUDA_VISIBLE_DEVICES=${gpu_id_list[0]} $python_env $script_path > $log_file 2>&1 &"
        echo $! > $pid_file
    elif [ "$service_name" -eq 1111 ]; then
        eval "CUDA_VISIBLE_DEVICES=${gpu_id_list[0]} $python_env $script_path > $log_file 2>&1 &"
        echo $! > $pid_file
    elif [ "$service_name" -eq 6666 ]; then
        eval "CUDA_VISIBLE_DEVICES=${gpu_id_list[1]} $python_env $script_path > $log_file 2>&1 &"
        echo $! > $pid_file
    else
        eval "CUDA_VISIBLE_DEVICES=${gpu_id_list[3]} $python_env $script_path > $log_file 2>&1 &"
        echo $! > $pid_file
    fi
}

# 初始化配置
INITIAL_DELAY=5
MAX_DELAY=60
MULTIPLIER=2

stop_service() {
    local service_name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null; then
            echo "Stopping $service_name..."
            kill $PID
            rm "$pid_file"
        else
            echo "$service_name is not running."
        fi
    else
        echo "PID file not found for $service_name."
    fi
}

# 监控服务并重启的函数
monitor_services() {
    echo $$ > /tmp/monitor_services.pid  # 记录监控进程的PID
    local delay=$INITIAL_DELAY
    while true; do
        if [ -f "$STOP_FILE" ]; then
            echo "Stop file detected. Stopping all services..."
            for service in "${!SERVICES[@]}"; do
                local service_info=(${SERVICES[$service]})
                local pid_file=${service_info[3]}
                stop_service "$service" "$pid_file"
            done
            rm "$STOP_FILE"
            rm /tmp/monitor_services.pid  # 删除PID文件
            exit 0
        fi

        for service in "${!SERVICES[@]}"; do
            local service_info=(${SERVICES[$service]})
            local root_dir=${service_info[0]}
            local python_env=${service_info[1]}
            local script_path=${service_info[2]}
            local pid_file=${service_info[3]}
            local log_file=${service_info[4]}

            if [ -f "$pid_file" ]; then
                PID=$(cat "$pid_file")
                if ! ps -p $PID > /dev/null; then
                    echo "$service has stopped. Restarting in $delay seconds..."
                    sleep "$delay"
                    start_service "$service" "${SERVICES[$service]}"
                    
                    # 更新延迟时间（指数增长）
                    delay=$((delay * MULTIPLIER))
                    if [ $delay -gt $MAX_DELAY ]; then
                        delay=$MAX_DELAY
                    fi
                fi
            else
                echo "PID file for $service not found. Starting $service..."
                start_service "$service" "${SERVICES[$service]}"
                # 重置延迟时间
                delay=$INITIAL_DELAY
            fi
        done
        
        sleep 5
    done
}


stop_all_services() {
    for service in "${!SERVICES[@]}"; do
        local service_info=(${SERVICES[$service]})
        local pid_file=${service_info[3]}
        stop_service "$service" "$pid_file"
    done
}

stop_monitor() {
    if [ -f "/tmp/monitor_services.pid" ]; then
        MONITOR_PID=$(cat /tmp/monitor_services.pid)
        if ps -p $MONITOR_PID > /dev/null; then
            echo "Stopping monitor service..."
            kill $MONITOR_PID
            rm /tmp/monitor_services.pid
        else
            echo "Monitor service is not running."
        fi
    else
        echo "Monitor PID file not found."
    fi
}
