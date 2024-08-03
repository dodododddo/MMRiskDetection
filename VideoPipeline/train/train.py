import cv2
import os
import numpy as np
import torch
import decord
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from videofact_main.model.videofact_pl_wrapper import VideoFACTPLWrapper
from videofact_main.model.common.videofact import VideoFACT
from videofact_main.inference_single import get_videofact_model

decord.bridge.set_bridge("torch")
config = {
    "img_size": (1080, 1920),
    "in_chans": 3,
    "patch_size": 128,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4,
    "drop_rate": 0,
    "bb1_db_depth": 1,
    "loss_alpha": 0.4,
    "lr": 1.0e-04,
    "decay_step": 2,
    "decay_rate": 0.75,
    "fe": "mislnet",
    "fe_config": {
        "patch_size": 128,
        "num_classes": 33,
    },
    # "fe_ckpt": "/home/tai/df_models/lab04_220401_mislnet_video_v1_ep=57_vl=0.6216.ckpt",
    "fe_freeze": False,
}

class VideoDataset(Dataset):
    def __init__(self, npy_file, transform=None):
        # 加载 .npy 文件
        dataset_np = np.load(npy_file, allow_pickle=True)
        self.file_paths = dataset_np[0]  # 文件地址
        self.labels = dataset_np[1]  # 标签
        self.transform = transform  # 数据增强操作

    def __len__(self):
        return len(self.file_paths) * 10

    def __getitem__(self, index):
        file_path = self.file_paths[index // 10]
        vr = process_video(file_path)

        frame_batch = vr[index % 10]
        # frame_batch = frame_batch.permute(0, 3, 1, 2).float()
        frame_batch = frame_batch.permute(2, 0, 1)

        if self.transform:
            frame_batch = self.transform(frame_batch)

        if frame_batch.shape[1] > frame_batch.shape[2]:
            frame_batch = frame_batch.permute(0, 2, 1)
            frame_batch = torchvision.transforms.functional.vflip(frame_batch)
        if frame_batch.shape[1] != 1080 or frame_batch.shape[2] != 1920:
            frame_batch = torchvision.transforms.functional.resize(frame_batch, (1080, 1920), antialias=True)
        label = self.labels[index // 10]
        mask = torch.zeros_like(frame_batch)

        # mask = (mask.mean(3) > 0.5).float() # T, H, W
        # vid = vid.permute(0, 3, 1, 2) # T, H, W, C -> T, C, H, W
        # print(frame_batch.shape)
        return frame_batch,int(label),mask
        

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, frames_info_path, batch_size=2, transform=None):
        super().__init__()
        self.frames_info_path = frames_info_path
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.dataset = VideoDataset(self.frames_info_path,self.transform)
        # self.val_dataset = VideoDataset('val_dataset.npy')

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


def create_dataset_nparray():
    train_file_paths = []
    train_labels = []
    test_file_paths = []
    test_labels = []
    val_file_paths = []
    val_labels = []
    folds = ['blink','embarrass','left_slope','mouth','nod','right_slope','smile','surpise','up','yaw']
    for fold in folds:
        folder_path = './data/DFMNIST+/fake_dataset/' + fold
        num = 0
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and num < 800:
                train_file_paths.append(file_path)
                train_labels.append(1)
                num += 1
            elif os.path.isfile(file_path) and 800 <= num < 900:
                test_file_paths.append(file_path)
                test_labels.append(1)
                num += 1
            elif os.path.isfile(file_path) and num >= 900:
                val_file_paths.append(file_path)
                val_labels.append(1)
                num += 1
    folder_path = './data/DFMNIST+/real_dataset/selected_train'
    num = 0
    for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                train_file_paths.append(file_path)
                train_labels.append(0)
                num += 1
    folder_path = './data/DFMNIST+/real_dataset/selected_test'
    for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                test_file_paths.append(file_path)
                test_labels.append(0)
    folder_path = './data/DFMNIST+/real_dataset/selected_val'
    for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                val_file_paths.append(file_path)
                val_labels.append(0)

    # 将地址和标签转换为 NumPy 数组
    train_dataset_np = np.array([train_file_paths, train_labels])
    test_dataset_np = np.array([test_file_paths, test_labels])
    val_dataset_np = np.array([val_file_paths, val_labels])

    # 保存 NumPy 数组为 .npy 文件
    np.save('train_dataset.npy', train_dataset_np)
    np.save('test_dataset.npy', test_dataset_np)
    np.save('val_dataset.npy', val_dataset_np)

    print(f"NumPy array saved to train_dataset.npy with shape: {train_dataset_np.shape}")
    print(f"NumPy array saved to test_dataset.npy with shape: {test_dataset_np.shape}")
    print(f"NumPy array saved to val_dataset.npy with shape: {val_dataset_np.shape}")

def process_video(file_path, target_length=10):
    # 读取视频
    vid = decord.VideoReader(file_path, num_threads=8)[:] / 255.0

    # 确定当前视频的长度
    current_length = len(vid)

    if current_length == target_length:
        # 如果当前长度已经等于目标长度，直接返回
        return vid
    elif current_length < target_length:
        # 如果当前长度小于目标长度，进行填充
        last_frame = vid[-1]  # 取最后一帧
        num_frames_to_add = target_length - current_length
        padding = torch.stack([last_frame] * num_frames_to_add)
        return torch.cat([vid, padding])
    else:
        # 如果当前长度大于目标长度，进行裁剪
        return vid[:target_length]

# 使用示例  
# 数据集使用示例
if __name__ == '__main__':
    # video_path = './data/DFMNIST+/real_dataset/selected_test/id10001#7w0IBEWc9Qw#001298#001705.mp4'
    # output_folder = './data/Mydataset'
    # extract_frames(video_path, output_folder, 0)
    # folder_path = './data/DFMNIST+/fake_dataset/blink'
    create_dataset_nparray()
    # dataset_np = np.load('train_dataset.npy')
    # print(dataset_np.shape)

#     transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     ])
    
#     os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
#     data_module = VideoDataModule('train_dataset.npy', batch_size=2, transform=transform)

# #     # 创建一个Trainer来训练模型
# #     checkpoint_callback = ModelCheckpoint(
# #     monitor='val_loss',    # 监控验证集上的损失
# #     dirpath='./model/videofact_wacv/',    # 保存模型的目录
# #     filename='best-checkpoint.ckp',  # 模型文件名
# #     save_top_k=1,          # 只保存最好的模型
# #     mode='min'             # 选择最小的损失
# # )
#     trainer = pl.Trainer(max_epochs=3, limit_val_batches=0, strategy=DDPStrategy(find_unused_parameters=True))
#     # trainer = pl.Trainer(max_epochs=2, callbacks=[checkpoint_callback])
#     # 传入模型（假设已经定义了模型类），例如：
#     videofact = VideoFACT
#     model = VideoFACTPLWrapper(model=videofact, **config)
#     # if torch.cuda.is_available():
#     #     device = "cuda"
#     # else:
#     #     device = "cpu"
#     checkpoint_path = os.path.join('./model/videofact_wacv/videofact_df.ckpt')
#     # model = VideoFACTPLWrapper.load_from_checkpoint(checkpoint_path, model = videofact, map_location=device, **config)
#     checkpoint = torch.load('./model/videofact_wacv/videofact_df.ckpt')
#     # print(checkpoint['state_dict'].keys())
#     partial_state_dict = {}
#     for k in checkpoint['state_dict'].keys():
#         if k.startswith('classifier'):
#             pass
#         else:
#             partial_state_dict[k] = checkpoint['state_dict'][k]

#     # 只加载部分层的权重
#     model.load_state_dict(partial_state_dict, strict=False)
#     # model = get_videofact_model('df', train=True)
#     # [print(k) for k,_ in model.model.named_parameters()]
#     for k,v in model.model.named_parameters():
#         if k.startswith('classifier'):
#             pass
#         else:
#             v.requires_grad = False  
#         # if k.startswith('localizer'):
#         #     v.requires_grad = False   
#     # # model(model.example_input_array)
#     trainer.fit(model, datamodule=data_module)
#     trainer.save_checkpoint(filepath=os.path.join('./model/videofact_wacv/my_model_t.ckpt'))