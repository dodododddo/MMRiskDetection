import os
import logging
import pandas as pd

import torch

from videofact_main.inference_single import get_videofact_model, load_single_video, process_single_video
from videofact_main.utils import *
from rich.logging import RichHandler
from typing import *

VideoFACT_xfer = None 
VideoFACT_df = None

videofact_df_threshold = 0.33
videofact_xfer_threshold = 0.4

@torch.no_grad()
def video_forgery_detection(
    video_path: str,
    shuffle = True,
    max_num_samples = 100,
    sample_every = 5,
    batch_size = 1,
    num_workers = 8
) -> List[str]:
    '''阴影检测'''
    if video_path is None:
        raise ValueError("video_path cannot be None")

    global VideoFACT_xfer
    if VideoFACT_xfer is None:
        VideoFACT_xfer = get_videofact_model("xfer")

    dataloader = load_single_video(
        video_path,
        shuffle,
        int(max_num_samples),
        int(sample_every),
        int(batch_size),
        int(num_workers),
    )
    results = process_single_video(VideoFACT_xfer, dataloader)
    result_frame_paths, idxs, scores = list(zip(*results))
    decisions = ["Forged" if score > videofact_xfer_threshold else "Authentic" for score in scores]
    detection_graph = pd.DataFrame(
        {
            "frame": idxs,
            "score": scores,
            "decision": decisions,
        }
    )

    return (
        result_frame_paths,
        list(zip(idxs, scores)),
        f"Frame: {idxs[0]}, Score: {sum(scores) / len(scores):.5f}",
        "Forged" if sum(scores) / len(scores) > videofact_xfer_threshold else "Authentic"
    )


@torch.no_grad()
def video_deepfake_detection(
    video_path: str,
    shuffle = True,
    max_num_samples = 100,
    sample_every = 5,
    batch_size = 2,
    num_workers = 5
) -> List[str]:
    '''换脸检测'''
    if video_path is None:
        raise ValueError("video_path cannot be None")

    global VideoFACT_df
    if VideoFACT_df is None:
        VideoFACT_df = get_videofact_model("df")

    dataloader = load_single_video(
        video_path,
        shuffle,
        int(max_num_samples),
        int(sample_every),
        int(batch_size),
        int(num_workers),
    )
    results = process_single_video(VideoFACT_df, dataloader)
    result_frame_paths, idxs, scores = list(zip(*results))
    decisions = ["Deepfaked" if score > videofact_df_threshold else "Authentic" for score in scores]
    detection_graph = pd.DataFrame(
        {
            "frame": idxs,
            "score": scores,
            "decision": decisions,
        }
    )

    return (
        result_frame_paths,
        list(zip(idxs, scores)),
        f"Frame: {idxs[0]}, Score: {sum(scores) / len(scores):.5f}",
        sum(scores) / len(scores),
        "Deepfaked" if sum(scores) / len(scores) > videofact_df_threshold else "Authentic"
    )

def data_needed(filePath):
    '''获取文件列表'''
    file_name = list()        #新建列表
    for i in os.listdir(filePath):        #获取filePath路径下所有文件名
        data_collect = ''.join(i)        #文件名字符串格式
        file_name.append(filePath + data_collect)        #将文件名作为列表元素填入
    print("获取filePath中文件名列表成功")        #打印获取成功提示
    return(file_name)        #返回列表

def cal_acc(filePath, flag, choose = 'deepfake'):
    '''计算准确率'''
    res_str = 'Authentic' if flag == 1 else ('Deepfaked' if choose == 'deepfake' else 'Forged')
    file_name = data_needed(filePath)
    res = 0
    all = 0
    for p in file_name:
        all += 1
        if choose == 'deepfake':
            result = video_deepfake_detection(p)
        else:
            result = video_forgery_detection(p)
        # print('\n'+p)
        print('\n'+str(result[-2]))
        if result[-1] == res_str:
            res += 1
        print('\nall:' + str(all) + ' res:'+ str(res) + ' all/res:' + str(res / all))
    return res / len(file_name)


if __name__ == "__main__":
    video_path = './data/p_demo.mp4'
    # video_path = './videofact_main/examples/df/donald_trump_deepfake.mp4'
    # video_path = './videofact_main/examples/df/sheeran-24kgold_deepfake.mp4'
    # video_path = './data/DFMNIST+/real_dataset/selected_test/id10001#7w0IBEWc9Qw#001298#001705.mp4'
    result = video_deepfake_detection(video_path)
    print('\n'+result[-1])
    print('\n'+str(result[-2]))
    # filePath = "./data/DFMNIST+/fake_dataset/nod/"
    filePath = "./data/DFMNIST+/real_dataset/selected_test/"
    # filename = data_needed(filePath)
    # print(filename[0:5])
    print(cal_acc(filePath , 1, 'deepfake'))