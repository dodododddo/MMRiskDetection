import os
import glob
import facefusion.globals
from facefusion import my_core

def facefusion_pipeline(source, target, output='../../Frontend/demo'):
    facefusion.globals.source_paths = [source]
    facefusion.globals.target_path = target
    facefusion.globals.output_path = output
    facefusion.globals.trim_frame_end = 10
    facefusion.globals.headless = True
    # files = glob.glob(os.path.join(output, '*'))  # 获取所有文件和文件夹
    # for file in files:
    #     try:
    #         if os.path.isfile(file):
    #             os.remove(file)  # 删除文件
    #     except Exception as e:
    #         print(f"Error deleting {file}: {e}")
    my_core.cli()
    # file_paths = glob.glob(os.path.join(output, '*'))  # 获取所有文件和文件夹路径
    # return file_paths[0]
    return output + '/' + target.split('/')[-1].split('.')[0] + '_gen.mp4'

    
    