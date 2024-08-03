#版本1

# import argparse
# import os
# import time

# import accelerate
# import cv2
# import numpy as np
# import torch
# import torchaudio.functional
# import torchvision.io
# from PIL import Image
# from diffusers import AutoencoderKL, DDIMScheduler
# from diffusers.utils.import_utils import is_xformers_available
# from insightface.app import FaceAnalysis
# from omegaconf import OmegaConf
# from transformers import Wav2Vec2Model, Wav2Vec2Processor

# from modules import UNet2DConditionModel, UNet3DConditionModel, VKpsGuider, AudioProjection
# from pipelines import VExpressPipeline
# from pipelines.utils import draw_kps_image, save_video
# from pipelines.utils import retarget_kps


# class VideoGenerator:
#     def __init__(self, args):
#         self.args = args
#         self.device = self._get_device()
#         self.dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

#         self.vae = AutoencoderKL.from_pretrained(args.vae_path).to(dtype=self.dtype, device=self.device)
#         self.audio_encoder = Wav2Vec2Model.from_pretrained(args.audio_encoder_path).to(dtype=self.dtype, device=self.device)
#         self.audio_processor = Wav2Vec2Processor.from_pretrained(args.audio_encoder_path)

#         self.scheduler = self._get_scheduler()
#         self.reference_net = self._load_reference_net()
#         self.denoising_unet = self._load_denoising_unet()
#         self.v_kps_guider = self._load_v_kps_guider()
#         self.audio_projection = self._load_audio_projection()

#         if is_xformers_available():
#             self.reference_net.enable_xformers_memory_efficient_attention()
#             self.denoising_unet.enable_xformers_memory_efficient_attention()
#         else:
#             raise ValueError("xformers is not available. Make sure it is installed correctly")

#     def _get_device(self):
#         if not self.args.do_multi_devices_inference:
#             return torch.device(f'{self.args.device}:{self.args.gpu_id}' if self.args.device == 'cuda' else self.args.device)
#         else:
#             self.accelerator = accelerate.Accelerator()
#             return torch.device(f'cuda:{self.accelerator.process_index}')

#     def _get_scheduler(self):
#         inference_config = OmegaConf.load('./inference_v2.yaml')
#         scheduler_kwargs = OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
#         return DDIMScheduler(**scheduler_kwargs)

#     def _load_reference_net(self):
#         reference_net = UNet2DConditionModel.from_config(self.args.unet_config_path).to(dtype=self.dtype, device=self.device)
#         reference_net.load_state_dict(torch.load(self.args.reference_net_path, map_location="cpu"), strict=False)
#         print(f'Loaded weights of Reference Net from {self.args.reference_net_path}.')
#         return reference_net

#     def _load_denoising_unet(self):
#         inference_config = OmegaConf.load('./inference_v2.yaml')
#         denoising_unet = UNet3DConditionModel.from_config_2d(
#             self.args.unet_config_path,
#             unet_additional_kwargs=inference_config.unet_additional_kwargs,
#         ).to(dtype=self.dtype, device=self.device)
#         denoising_unet.load_state_dict(torch.load(self.args.denoising_unet_path, map_location="cpu"), strict=False)
#         print(f'Loaded weights of Denoising U-Net from {self.args.denoising_unet_path}.')

#         denoising_unet.load_state_dict(torch.load(self.args.motion_module_path, map_location="cpu"), strict=False)
#         print(f'Loaded weights of Denoising U-Net Motion Module from {self.args.motion_module_path}.')

#         return denoising_unet

#     def _load_v_kps_guider(self):
#         v_kps_guider = VKpsGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=self.dtype, device=self.device)
#         v_kps_guider.load_state_dict(torch.load(self.args.v_kps_guider_path, map_location="cpu"))
#         print(f'Loaded weights of V-Kps Guider from {self.args.v_kps_guider_path}.')
#         return v_kps_guider

#     def _load_audio_projection(self):
#         audio_projection = AudioProjection(
#             dim=self.denoising_unet.config.cross_attention_dim,
#             depth=4,
#             dim_head=64,
#             heads=12,
#             num_queries=2 * self.args.num_pad_audio_frames + 1,
#             embedding_dim=self.denoising_unet.config.cross_attention_dim,
#             output_dim=self.denoising_unet.config.cross_attention_dim,
#             ff_mult=4,
#             max_seq_len=2 * (2 * self.args.num_pad_audio_frames + 1),
#         ).to(dtype=self.dtype, device=self.device)
#         audio_projection.load_state_dict(torch.load(self.args.audio_projection_path, map_location='cpu'))
#         print(f'Loaded weights of Audio Projection from {self.args.audio_projection_path}.')
#         return audio_projection

#     def prepare_reference_kps(self):
#         app = FaceAnalysis(
#             providers=['CUDAExecutionProvider' if self.args.device == 'cuda' else 'CPUExecutionProvider'],
#             provider_options=[{'device_id': self.args.gpu_id}] if self.args.device == 'cuda' else [],
#             root=self.args.insightface_model_path,
#         )
#         app.prepare(ctx_id=0, det_size=(self.args.image_height, self.args.image_width))

#         reference_image_for_kps = cv2.imread(self.args.reference_image_path)
#         reference_image_for_kps = cv2.resize(reference_image_for_kps, (self.args.image_width, self.args.image_height))
#         reference_kps = app.get(reference_image_for_kps)[0].kps[:3]

#         if self.args.save_gpu_memory:
#             del app
#         torch.cuda.empty_cache()

#         return reference_kps

#     def preprocess_audio(self):
#         _, audio_waveform, meta_info = torchvision.io.read_video(self.args.audio_path, pts_unit='sec')
#         audio_sampling_rate = meta_info['audio_fps']
#         print(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
#         if audio_sampling_rate != self.args.standard_audio_sampling_rate:
#             audio_waveform = torchaudio.functional.resample(
#                 audio_waveform,
#                 orig_freq=audio_sampling_rate,
#                 new_freq=self.args.standard_audio_sampling_rate,
#             )
#         audio_waveform = audio_waveform.mean(dim=0)
#         return audio_waveform

#     def compute_video_length(self, audio_waveform):
#         duration = audio_waveform.shape[0] / self.args.standard_audio_sampling_rate
#         init_video_length = int(duration * self.args.fps)
#         num_contexts = np.around((init_video_length + self.args.context_overlap) / self.args.context_frames)
#         video_length = int(num_contexts * self.args.context_frames - self.args.context_overlap)
#         fps = video_length / duration
#         print(f'The corresponding video length is {video_length}.')
#         return video_length, fps

#     def prepare_kps_sequence(self, video_length, reference_kps):
#         kps_sequence = None
#         if self.args.kps_path != "":
#             assert os.path.exists(self.args.kps_path), f'{self.args.kps_path} does not exist'
#             kps_sequence = torch.tensor(torch.load(self.args.kps_path))  # [len, 3, 2]
#             print(f'The original length of kps sequence is {kps_sequence.shape[0]}.')

#             if kps_sequence.shape[0] > video_length:
#                 kps_sequence = kps_sequence[:video_length, :, :]

#             kps_sequence = torch.nn.functional.interpolate(kps_sequence.permute(1, 2, 0), size=video_length, mode='linear')
#             kps_sequence = kps_sequence.permute(2, 0, 1)
#             print(f'The interpolated length of kps sequence is {kps_sequence.shape[0]}.')

#         retarget_strategy = self.args.retarget_strategy
#         if retarget_strategy == 'fix_face':
#             kps_sequence = torch.tensor([reference_kps] * video_length)
#         elif retarget_strategy == 'no_retarget':
#             kps_sequence = kps_sequence
#         elif retarget_strategy == 'offset_retarget':
#             kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=True)
#         elif retarget_strategy == 'naive_retarget':
#             kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=False)
#         else:
#             raise ValueError(f'The retarget strategy {retarget_strategy} is not supported.')

#         kps_images = []
#         for i in range(video_length):
#             kps_image = draw_kps_image(self.args.image_height, self.args.image_width, kps_sequence[i])
#             kps_images.append(Image.fromarray(kps_image))

#         return kps_images

#     def generate_video(self):
#         start_time = time.time()
#         reference_image = Image.open(self.args.reference_image_path).convert('RGB')
#         reference_image = reference_image.resize((self.args.image_height, self.args.image_width))

#         reference_kps = self.prepare_reference_kps()
#         audio_waveform = self.preprocess_audio()
#         video_length, fps = self.compute_video_length(audio_waveform)
#         kps_images = self.prepare_kps_sequence(video_length, reference_kps)

#         generator = torch.manual_seed(self.args.seed)
#         pipeline = VExpressPipeline(
#             vae=self.vae,
#             reference_net=self.reference_net,
#             denoising_unet=self.denoising_unet,
#             v_kps_guider=self.v_kps_guider,
#             audio_processor=self.audio_processor,
#             audio_encoder=self.audio_encoder,
#             audio_projection=self.audio_projection,
#             scheduler=self.scheduler,
#         ).to(dtype=self.dtype, device=self.device)

#         video_tensor = pipeline(
#             reference_image=reference_image,
#             kps_images=kps_images,
#             audio_waveform=audio_waveform,
#             width=self.args.image_width,
#             height=self.args.image_height,
#             video_length=video_length,
#             num_inference_steps=self.args.num_inference_steps,
#             guidance_scale=self.args.guidance_scale,
#             context_frames=self.args.context_frames,
#             context_overlap=self.args.context_overlap,
#             reference_attention_weight=self.args.reference_attention_weight,
#             audio_attention_weight=self.args.audio_attention_weight,
#             num_pad_audio_frames=self.args.num_pad_audio_frames,
#             generator=generator,
#             do_multi_devices_inference=self.args.do_multi_devices_inference,
#             save_gpu_memory=self.args.save_gpu_memory,
#         )

#         if not self.args.do_multi_devices_inference or self.accelerator.is_main_process:
#             save_video(video_tensor, self.args.audio_path, self.args.output_path, self.device, fps)
#             consumed_time = time.time() - start_time
#             generation_fps = video_tensor.shape[2] / consumed_time
#             print(f'The generated video has been saved at {self.args.output_path}. '
#                   f'The generation time is {consumed_time:.1f} seconds. '
#                   f'The generation FPS is {generation_fps:.2f}.')

# def parse_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--unet_config_path', type=str, default='./model_ckpts/stable-diffusion-v1-5/unet/config.json')
#     parser.add_argument('--vae_path', type=str, default='./model_ckpts/sd-vae-ft-mse/')
#     parser.add_argument('--audio_encoder_path', type=str, default='./model_ckpts/wav2vec2-base-960h/')
#     parser.add_argument('--insightface_model_path', type=str, default='./model_ckpts/insightface_models/')

#     parser.add_argument('--denoising_unet_path', type=str, default='./model_ckpts/v-express/denoising_unet.bin')
#     parser.add_argument('--reference_net_path', type=str, default='./model_ckpts/v-express/reference_net.bin')
#     parser.add_argument('--v_kps_guider_path', type=str, default='./model_ckpts/v-express/v_kps_guider.bin')
#     parser.add_argument('--audio_projection_path', type=str, default='./model_ckpts/v-express/audio_projection.bin')
#     parser.add_argument('--motion_module_path', type=str, default='./model_ckpts/v-express/motion_module.bin')

#     parser.add_argument('--retarget_strategy', type=str, default='fix_face',
#                         help='{fix_face, no_retarget, offset_retarget, naive_retarget}')

#     parser.add_argument('--dtype', type=str, default='fp16')
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--do_multi_devices_inference', action='store_true')
#     parser.add_argument('--save_gpu_memory', action='store_true')

#     parser.add_argument('--num_pad_audio_frames', type=int, default=2)
#     parser.add_argument('--standard_audio_sampling_rate', type=int, default=16000)

#     parser.add_argument('--reference_image_path', type=str, default='./test_samples/emo/talk_emotion/ref.jpg')
#     parser.add_argument('--audio_path', type=str, default='./test_samples/emo/talk_emotion/aud.mp3')
#     parser.add_argument('--kps_path', type=str, default='./test_samples/emo/talk_emotion/kps.pth')
#     parser.add_argument('--output_path', type=str, default='./output/emo/talk_emotion.mp4')

#     parser.add_argument('--image_width', type=int, default=512)
#     parser.add_argument('--image_height', type=int, default=512)
#     parser.add_argument('--fps', type=float, default=30.0)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--num_inference_steps', type=int, default=25)
#     parser.add_argument('--guidance_scale', type=float, default=3.5)
#     parser.add_argument('--context_frames', type=int, default=12)
#     parser.add_argument('--context_overlap', type=int, default=4)
#     parser.add_argument('--reference_attention_weight', default=0.95, type=float)
#     parser.add_argument('--audio_attention_weight', default=3., type=float)

#     args = parser.parse_args()

#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     video_generator = VideoGenerator(args)
#     video_generator.generate_video()

#版本2

# import argparse
# import os
# import time

# import accelerate
# import cv2
# import numpy as np
# import torch
# import torchaudio.functional
# import torchvision.io
# from PIL import Image
# from diffusers import AutoencoderKL, DDIMScheduler
# from diffusers.utils.import_utils import is_xformers_available
# from insightface.app import FaceAnalysis
# from omegaconf import OmegaConf
# from transformers import Wav2Vec2Model, Wav2Vec2Processor

# from modules import UNet2DConditionModel, UNet3DConditionModel, VKpsGuider, AudioProjection
# from pipelines import VExpressPipeline
# from pipelines.utils import draw_kps_image, save_video
# from pipelines.utils import retarget_kps


# class VideoGenerator:
#     def __init__(self, args):
#         self.args = args
#         self.device = self._get_device()
#         self.dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

#         self.vae = AutoencoderKL.from_pretrained(args.vae_path).to(dtype=self.dtype, device=self.device)
#         self.audio_encoder = Wav2Vec2Model.from_pretrained(args.audio_encoder_path).to(dtype=self.dtype, device=self.device)
#         self.audio_processor = Wav2Vec2Processor.from_pretrained(args.audio_encoder_path)

#         self.scheduler = self._get_scheduler()
#         self.reference_net = self._load_reference_net()
#         self.denoising_unet = self._load_denoising_unet()
#         self.v_kps_guider = self._load_v_kps_guider()
#         self.audio_projection = self._load_audio_projection()

#         if is_xformers_available():
#             self.reference_net.enable_xformers_memory_efficient_attention()
#             self.denoising_unet.enable_xformers_memory_efficient_attention()
#         else:
#             raise ValueError("xformers is not available. Make sure it is installed correctly")

#     def _get_device(self):
#         if not self.args.do_multi_devices_inference:
#             return torch.device(f'{self.args.device}:{self.args.gpu_id}' if self.args.device == 'cuda' else self.args.device)
#         else:
#             self.accelerator = accelerate.Accelerator()
#             return torch.device(f'cuda:{self.accelerator.process_index}')

#     def _get_scheduler(self):
#         inference_config = OmegaConf.load('./inference_v2.yaml')
#         scheduler_kwargs = OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
#         return DDIMScheduler(**scheduler_kwargs)

#     def _load_reference_net(self):
#         reference_net = UNet2DConditionModel.from_config(self.args.unet_config_path).to(dtype=self.dtype, device=self.device)
#         reference_net.load_state_dict(torch.load(self.args.reference_net_path, map_location="cpu"), strict=False)
#         print(f'Loaded weights of Reference Net from {self.args.reference_net_path}.')
#         return reference_net

#     def _load_denoising_unet(self):
#         inference_config = OmegaConf.load('./inference_v2.yaml')
#         denoising_unet = UNet3DConditionModel.from_config_2d(
#             self.args.unet_config_path,
#             unet_additional_kwargs=inference_config.unet_additional_kwargs,
#         ).to(dtype=self.dtype, device=self.device)
#         denoising_unet.load_state_dict(torch.load(self.args.denoising_unet_path, map_location="cpu"), strict=False)
#         print(f'Loaded weights of Denoising U-Net from {self.args.denoising_unet_path}.')

#         denoising_unet.load_state_dict(torch.load(self.args.motion_module_path, map_location="cpu"), strict=False)
#         print(f'Loaded weights of Denoising U-Net Motion Module from {self.args.motion_module_path}.')

#         return denoising_unet

#     def _load_v_kps_guider(self):
#         v_kps_guider = VKpsGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=self.dtype, device=self.device)
#         v_kps_guider.load_state_dict(torch.load(self.args.v_kps_guider_path, map_location="cpu"))
#         print(f'Loaded weights of V-Kps Guider from {self.args.v_kps_guider_path}.')
#         return v_kps_guider

#     def _load_audio_projection(self):
#         audio_projection = AudioProjection(
#             dim=self.denoising_unet.config.cross_attention_dim,
#             depth=4,
#             dim_head=64,
#             heads=12,
#             num_queries=2 * self.args.num_pad_audio_frames + 1,
#             embedding_dim=self.denoising_unet.config.cross_attention_dim,
#             output_dim=self.denoising_unet.config.cross_attention_dim,
#             ff_mult=4,
#             max_seq_len=2 * (2 * self.args.num_pad_audio_frames + 1),
#         ).to(dtype=self.dtype, device=self.device)
#         audio_projection.load_state_dict(torch.load(self.args.audio_projection_path, map_location='cpu'))
#         print(f'Loaded weights of Audio Projection from {self.args.audio_projection_path}.')
#         return audio_projection

#     def prepare_reference_kps(self):
#         app = FaceAnalysis(
#             providers=['CUDAExecutionProvider' if self.args.device == 'cuda' else 'CPUExecutionProvider'],
#             provider_options=[{'device_id': self.args.gpu_id}] if self.args.device == 'cuda' else [],
#             root=self.args.insightface_model_path,
#         )
#         app.prepare(ctx_id=0, det_size=(self.args.image_height, self.args.image_width))

#         reference_image_for_kps = cv2.imread(self.args.reference_image_path)
#         reference_image_for_kps = cv2.resize(reference_image_for_kps, (self.args.image_width, self.args.image_height))
#         reference_kps = app.get(reference_image_for_kps)[0].kps[:3]

#         if self.args.save_gpu_memory:
#             del app
#         torch.cuda.empty_cache()

#         return reference_kps

#     def preprocess_audio(self):
#         _, audio_waveform, meta_info = torchvision.io.read_video(self.args.audio_path, pts_unit='sec')
#         audio_sampling_rate = meta_info['audio_fps']
#         print(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
#         if audio_sampling_rate != self.args.standard_audio_sampling_rate:
#             audio_waveform = torchaudio.functional.resample(
#                 audio_waveform,
#                 orig_freq=audio_sampling_rate,
#                 new_freq=self.args.standard_audio_sampling_rate,
#             )
#         audio_waveform = audio_waveform.mean(dim=0)
#         return audio_waveform

#     def compute_video_length(self, audio_waveform):
#         duration = audio_waveform.shape[0] / self.args.standard_audio_sampling_rate
#         init_video_length = int(duration * self.args.fps)
#         num_contexts = np.around((init_video_length + self.args.context_overlap) / self.args.context_frames)
#         video_length = int(num_contexts * self.args.context_frames - self.args.context_overlap)
#         fps = video_length / duration
#         print(f'The corresponding video length is {video_length}.')
#         return video_length, fps

#     def prepare_kps_sequence(self, video_length, reference_kps):
#         kps_sequence = None
#         if self.args.kps_path != "":
#             assert os.path.exists(self.args.kps_path), f'{self.args.kps_path} does not exist'
#             kps_sequence = torch.tensor(torch.load(self.args.kps_path))  # [len, 3, 2]
#             print(f'The original length of kps sequence is {kps_sequence.shape[0]}.')

#             if kps_sequence.shape[0] > video_length:
#                 kps_sequence = kps_sequence[:video_length, :, :]

#             kps_sequence = torch.nn.functional.interpolate(kps_sequence.permute(1, 2, 0), size=video_length, mode='linear')
#             kps_sequence = kps_sequence.permute(2, 0, 1)
#             print(f'The interpolated length of kps sequence is {kps_sequence.shape[0]}.')

#         retarget_strategy = self.args.retarget_strategy
#         if retarget_strategy == 'fix_face':
#             kps_sequence = torch.tensor([reference_kps] * video_length)
#         elif retarget_strategy == 'no_retarget':
#             kps_sequence = kps_sequence
#         elif retarget_strategy == 'offset_retarget':
#             kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=True)
#         elif retarget_strategy == 'naive_retarget':
#             kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=False)
#         else:
#             raise ValueError(f'The retarget strategy {retarget_strategy} is not supported.')

#         kps_images = []
#         for i in range(video_length):
#             kps_image = draw_kps_image(self.args.image_height, self.args.image_width, kps_sequence[i])
#             kps_images.append(Image.fromarray(kps_image))

#         return kps_images

#     def generate_video(self):
#         start_time = time.time()
#         reference_image = Image.open(self.args.reference_image_path).convert('RGB')
#         reference_image = reference_image.resize((self.args.image_height, self.args.image_width))

#         reference_kps = self.prepare_reference_kps()
#         audio_waveform = self.preprocess_audio()
#         video_length, fps = self.compute_video_length(audio_waveform)
#         kps_images = self.prepare_kps_sequence(video_length, reference_kps)

#         generator = torch.manual_seed(self.args.seed)
#         pipeline = VExpressPipeline(
#             vae=self.vae,
#             reference_net=self.reference_net,
#             denoising_unet=self.denoising_unet,
#             v_kps_guider=self.v_kps_guider,
#             audio_processor=self.audio_processor,
#             audio_encoder=self.audio_encoder,
#             audio_projection=self.audio_projection,
#             scheduler=self.scheduler,
#         ).to(dtype=self.dtype, device=self.device)

#         video_tensor = pipeline(
#             reference_image=reference_image,
#             kps_images=kps_images,
#             audio_waveform=audio_waveform,
#             width=self.args.image_width,
#             height=self.args.image_height,
#             video_length=video_length,
#             num_inference_steps=self.args.num_inference_steps,
#             guidance_scale=self.args.guidance_scale,
#             context_frames=self.args.context_frames,
#             context_overlap=self.args.context_overlap,
#             reference_attention_weight=self.args.reference_attention_weight,
#             audio_attention_weight=self.args.audio_attention_weight,
#             num_pad_audio_frames=self.args.num_pad_audio_frames,
#             generator=generator,
#             do_multi_devices_inference=self.args.do_multi_devices_inference,
#             save_gpu_memory=self.args.save_gpu_memory,
#         )

#         if not self.args.do_multi_devices_inference or self.accelerator.is_main_process:
#             save_video(video_tensor, self.args.audio_path, self.args.output_path, self.device, fps)
#             consumed_time = time.time() - start_time
#             generation_fps = video_tensor.shape[2] / consumed_time
#             print(f'The generated video has been saved at {self.args.output_path}. '
#                   f'The generation time is {consumed_time:.1f} seconds. '
#                   f'The generation FPS is {generation_fps:.2f}.')

#     @staticmethod
#     def parse_args(unet_config_path='./model_ckpts/stable-diffusion-v1-5/unet/config.json', vae_path='./model_ckpts/sd-vae-ft-mse/',
#                    audio_encoder_path='./model_ckpts/wav2vec2-base-960h/', insightface_model_path='./model_ckpts/insightface_models/',
#                    denoising_unet_path='./model_ckpts/v-express/denoising_unet.bin', reference_net_path='./model_ckpts/v-express/reference_net.bin',
#                    v_kps_guider_path='./model_ckpts/v-express/v_kps_guider.bin', audio_projection_path='./model_ckpts/v-express/audio_projection.bin',
#                    motion_module_path='./model_ckpts/v-express/motion_module.bin', retarget_strategy='fix_face', dtype='fp16', device='cuda',
#                    gpu_id=0, do_multi_devices_inference=False, save_gpu_memory=False, num_pad_audio_frames=2, standard_audio_sampling_rate=16000,
#                    reference_image_path='./test_samples/emo/talk_emotion/ref.jpg', audio_path='./test_samples/emo/talk_emotion/aud.mp3',
#                    kps_path='./test_samples/emo/talk_emotion/kps.pth', output_path='./output/emo/talk_emotion.mp4', image_width=512,
#                    image_height=512, fps=30.0, seed=42, num_inference_steps=25, guidance_scale=3.5, context_frames=12, context_overlap=4,
#                    reference_attention_weight=0.95, audio_attention_weight=3.0):

#         parser = argparse.ArgumentParser()

#         parser.add_argument('--unet_config_path', type=str, default=unet_config_path)
#         parser.add_argument('--vae_path', type=str, default=vae_path)
#         parser.add_argument('--audio_encoder_path', type=str, default=audio_encoder_path)
#         parser.add_argument('--insightface_model_path', type=str, default=insightface_model_path)

#         parser.add_argument('--denoising_unet_path', type=str, default=denoising_unet_path)
#         parser.add_argument('--reference_net_path', type=str, default=reference_net_path)
#         parser.add_argument('--v_kps_guider_path', type=str, default=v_kps_guider_path)
#         parser.add_argument('--audio_projection_path', type=str, default=audio_projection_path)
#         parser.add_argument('--motion_module_path', type=str, default=motion_module_path)

#         parser.add_argument('--retarget_strategy', type=str, default=retarget_strategy,
#                             help='{fix_face, no_retarget, offset_retarget, naive_retarget}')

#         parser.add_argument('--dtype', type=str, default=dtype)
#         parser.add_argument('--device', type=str, default=device)
#         parser.add_argument('--gpu_id', type=int, default=gpu_id)
#         parser.add_argument('--do_multi_devices_inference', action='store_true', default=do_multi_devices_inference)
#         parser.add_argument('--save_gpu_memory', action='store_true', default=save_gpu_memory)

#         parser.add_argument('--num_pad_audio_frames', type=int, default=num_pad_audio_frames)
#         parser.add_argument('--standard_audio_sampling_rate', type=int, default=standard_audio_sampling_rate)

#         parser.add_argument('--reference_image_path', type=str, default=reference_image_path)
#         parser.add_argument('--audio_path', type=str, default=audio_path)
#         parser.add_argument('--kps_path', type=str, default=kps_path)
#         parser.add_argument('--output_path', type=str, default=output_path)

#         parser.add_argument('--image_width', type=int, default=image_width)
#         parser.add_argument('--image_height', type=int, default=image_height)
#         parser.add_argument('--fps', type=float, default=fps)
#         parser.add_argument('--seed', type=int, default=seed)
#         parser.add_argument('--num_inference_steps', type=int, default=num_inference_steps)
#         parser.add_argument('--guidance_scale', type=float, default=guidance_scale)
#         parser.add_argument('--context_frames', type=int, default=context_frames)
#         parser.add_argument('--context_overlap', type=int, default=context_overlap)
#         parser.add_argument('--reference_attention_weight', type=float, default=reference_attention_weight)
#         parser.add_argument('--audio_attention_weight', type=float, default=audio_attention_weight)

#         return parser.parse_args()

#     @classmethod
#     def run_video_generation(cls, **kwargs):
#         args = cls.parse_args(**kwargs)
#         video_generator = cls(args)
#         video_generator.generate_video()

#版本3

import argparse
import os
import time

import accelerate
import cv2
import numpy as np
import torch
import torchaudio.functional
import torchvision.io
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from insightface.app import FaceAnalysis
from omegaconf import OmegaConf
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from modules import UNet2DConditionModel, UNet3DConditionModel, VKpsGuider, AudioProjection
from pipelines import VExpressPipeline
from pipelines.utils import draw_kps_image, save_video
from pipelines.utils import retarget_kps


class VideoGenerator:
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

        self.vae = AutoencoderKL.from_pretrained(args.vae_path).to(dtype=self.dtype, device=self.device)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(args.audio_encoder_path).to(dtype=self.dtype, device=self.device)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(args.audio_encoder_path)

        self.scheduler = self._get_scheduler()
        self.reference_net = self._load_reference_net()
        self.denoising_unet = self._load_denoising_unet()
        self.v_kps_guider = self._load_v_kps_guider()
        self.audio_projection = self._load_audio_projection()

        if is_xformers_available():
            self.reference_net.enable_xformers_memory_efficient_attention()
            self.denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def _get_device(self):
        if not self.args.do_multi_devices_inference:
            return torch.device(f'{self.args.device}:{self.args.gpu_id}' if self.args.device == 'cuda' else self.args.device)
        else:
            self.accelerator = accelerate.Accelerator()
            return torch.device(f'cuda:{self.accelerator.process_index}')

    def _get_scheduler(self):
        inference_config = OmegaConf.load('./inference_v2.yaml')
        scheduler_kwargs = OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
        return DDIMScheduler(**scheduler_kwargs)

    def _load_reference_net(self):
        reference_net = UNet2DConditionModel.from_config(self.args.unet_config_path).to(dtype=self.dtype, device=self.device)
        reference_net.load_state_dict(torch.load(self.args.reference_net_path, map_location="cpu"), strict=False)
        print(f'Loaded weights of Reference Net from {self.args.reference_net_path}.')
        return reference_net

    def _load_denoising_unet(self):
        inference_config = OmegaConf.load('./inference_v2.yaml')
        denoising_unet = UNet3DConditionModel.from_config_2d(
            self.args.unet_config_path,
            unet_additional_kwargs=inference_config.unet_additional_kwargs,
        ).to(dtype=self.dtype, device=self.device)
        denoising_unet.load_state_dict(torch.load(self.args.denoising_unet_path, map_location="cpu"), strict=False)
        print(f'Loaded weights of Denoising U-Net from {self.args.denoising_unet_path}.')

        denoising_unet.load_state_dict(torch.load(self.args.motion_module_path, map_location="cpu"), strict=False)
        print(f'Loaded weights of Denoising U-Net Motion Module from {self.args.motion_module_path}.')

        return denoising_unet

    def _load_v_kps_guider(self):
        v_kps_guider = VKpsGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=self.dtype, device=self.device)
        v_kps_guider.load_state_dict(torch.load(self.args.v_kps_guider_path, map_location="cpu"))
        print(f'Loaded weights of V-Kps Guider from {self.args.v_kps_guider_path}.')
        return v_kps_guider

    def _load_audio_projection(self):
        audio_projection = AudioProjection(
            dim=self.denoising_unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=2 * self.args.num_pad_audio_frames + 1,
            embedding_dim=self.denoising_unet.config.cross_attention_dim,
            output_dim=self.denoising_unet.config.cross_attention_dim,
            ff_mult=4,
            max_seq_len=2 * (2 * self.args.num_pad_audio_frames + 1),
        ).to(dtype=self.dtype, device=self.device)
        audio_projection.load_state_dict(torch.load(self.args.audio_projection_path, map_location='cpu'))
        print(f'Loaded weights of Audio Projection from {self.args.audio_projection_path}.')
        return audio_projection

    def prepare_reference_kps(self):
        app = FaceAnalysis(
            providers=['CUDAExecutionProvider' if self.args.device == 'cuda' else 'CPUExecutionProvider'],
            provider_options=[{'device_id': self.args.gpu_id}] if self.args.device == 'cuda' else [],
            root=self.args.insightface_model_path,
        )
        app.prepare(ctx_id=0, det_size=(self.args.image_height, self.args.image_width))

        reference_image_for_kps = cv2.imread(self.args.reference_image_path)
        reference_image_for_kps = cv2.resize(reference_image_for_kps, (self.args.image_width, self.args.image_height))
        reference_kps = app.get(reference_image_for_kps)[0].kps[:3]

        if self.args.save_gpu_memory:
            del app
        torch.cuda.empty_cache()

        return reference_kps

    def preprocess_audio(self):
        _, audio_waveform, meta_info = torchvision.io.read_video(self.args.audio_path, pts_unit='sec')
        audio_sampling_rate = meta_info['audio_fps']
        print(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
        if audio_sampling_rate != self.args.standard_audio_sampling_rate:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform,
                orig_freq=audio_sampling_rate,
                new_freq=self.args.standard_audio_sampling_rate,
            )
        audio_waveform = audio_waveform.mean(dim=0)
        return audio_waveform

    def compute_video_length(self, audio_waveform):
        duration = audio_waveform.shape[0] / self.args.standard_audio_sampling_rate
        init_video_length = int(duration * self.args.fps)
        num_contexts = np.around((init_video_length + self.args.context_overlap) / self.args.context_frames)
        video_length = int(num_contexts * self.args.context_frames - self.args.context_overlap)
        fps = video_length / duration
        print(f'The corresponding video length is {video_length}.')
        return video_length, fps

    def prepare_kps_sequence(self, video_length, reference_kps):
        kps_sequence = None
        if self.args.kps_path != "":
            assert os.path.exists(self.args.kps_path), f'{self.args.kps_path} does not exist'
            kps_sequence = torch.tensor(torch.load(self.args.kps_path))  # [len, 3, 2]
            print(f'The original length of kps sequence is {kps_sequence.shape[0]}.')

            if kps_sequence.shape[0] > video_length:
                kps_sequence = kps_sequence[:video_length, :, :]

            kps_sequence = torch.nn.functional.interpolate(kps_sequence.permute(1, 2, 0), size=video_length, mode='linear')
            kps_sequence = kps_sequence.permute(2, 0, 1)
            print(f'The interpolated length of kps sequence is {kps_sequence.shape[0]}.')

        retarget_strategy = self.args.retarget_strategy
        if retarget_strategy == 'fix_face':
            kps_sequence = torch.tensor([reference_kps] * video_length)
        elif retarget_strategy == 'no_retarget':
            kps_sequence = kps_sequence
        elif retarget_strategy == 'offset_retarget':
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=True)
        elif retarget_strategy == 'naive_retarget':
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=False)
        else:
            raise ValueError(f'The retarget strategy {retarget_strategy} is not supported.')

        kps_images = []
        for i in range(video_length):
            kps_image = draw_kps_image(self.args.image_height, self.args.image_width, kps_sequence[i])
            kps_images.append(Image.fromarray(kps_image))

        return kps_images

    def generate_video(self):
        start_time = time.time()
        reference_image = Image.open(self.args.reference_image_path).convert('RGB')
        reference_image = reference_image.resize((self.args.image_height, self.args.image_width))

        reference_kps = self.prepare_reference_kps()
        audio_waveform = self.preprocess_audio()
        video_length, fps = self.compute_video_length(audio_waveform)
        kps_images = self.prepare_kps_sequence(video_length, reference_kps)

        generator = torch.manual_seed(self.args.seed)
        pipeline = VExpressPipeline(
            vae=self.vae,
            reference_net=self.reference_net,
            denoising_unet=self.denoising_unet,
            v_kps_guider=self.v_kps_guider,
            audio_processor=self.audio_processor,
            audio_encoder=self.audio_encoder,
            audio_projection=self.audio_projection,
            scheduler=self.scheduler,
        ).to(dtype=self.dtype, device=self.device)

        video_tensor = pipeline(
            reference_image=reference_image,
            kps_images=kps_images,
            audio_waveform=audio_waveform,
            width=self.args.image_width,
            height=self.args.image_height,
            video_length=video_length,
            num_inference_steps=self.args.num_inference_steps,
            guidance_scale=self.args.guidance_scale,
            context_frames=self.args.context_frames,
            context_overlap=self.args.context_overlap,
            reference_attention_weight=self.args.reference_attention_weight,
            audio_attention_weight=self.args.audio_attention_weight,
            num_pad_audio_frames=self.args.num_pad_audio_frames,
            generator=generator,
            do_multi_devices_inference=self.args.do_multi_devices_inference,
            save_gpu_memory=self.args.save_gpu_memory,
        )

        output_path = self.args.output_path
        if not self.args.do_multi_devices_inference or self.accelerator.is_main_process:
            save_video(video_tensor, self.args.audio_path, output_path, self.device, fps)
            consumed_time = time.time() - start_time
            generation_fps = video_tensor.shape[2] / consumed_time
            print(f'The generated video has been saved at {output_path}. '
                  f'The generation time is {consumed_time:.1f} seconds. '
                  f'The generation FPS is {generation_fps:.2f}.')
        
        return output_path

    @staticmethod
    def parse_args(unet_config_path='./model_ckpts/stable-diffusion-v1-5/unet/config.json', vae_path='./model_ckpts/sd-vae-ft-mse/',
                   audio_encoder_path='./model_ckpts/wav2vec2-base-960h/', insightface_model_path='./model_ckpts/insightface_models/',
                   denoising_unet_path='./model_ckpts/v-express/denoising_unet.bin', reference_net_path='./model_ckpts/v-express/reference_net.bin',
                   v_kps_guider_path='./model_ckpts/v-express/v_kps_guider.bin', audio_projection_path='./model_ckpts/v-express/audio_projection.bin',
                   motion_module_path='./model_ckpts/v-express/motion_module.bin', retarget_strategy='fix_face', dtype='fp16', device='cuda',
                   gpu_id=0, do_multi_devices_inference=False, save_gpu_memory=False, num_pad_audio_frames=2, standard_audio_sampling_rate=16000,
                   reference_image_path='./test_samples/emo/talk_emotion/ref.jpg', audio_path='./test_samples/emo/talk_emotion/aud.mp3',
                   kps_path='./test_samples/emo/talk_emotion/kps.pth', output_path='./output/emo/talk_emotion.mp4', image_width=512,
                   image_height=512, fps=30.0, seed=42, num_inference_steps=25, guidance_scale=3.5, context_frames=12, context_overlap=4,
                   reference_attention_weight=0.95, audio_attention_weight=3.0):

        parser = argparse.ArgumentParser()

        parser.add_argument('--unet_config_path', type=str, default=unet_config_path)
        parser.add_argument('--vae_path', type=str, default=vae_path)
        parser.add_argument('--audio_encoder_path', type=str, default=audio_encoder_path)
        parser.add_argument('--insightface_model_path', type=str, default=insightface_model_path)

        parser.add_argument('--denoising_unet_path', type=str, default=denoising_unet_path)
        parser.add_argument('--reference_net_path', type=str, default=reference_net_path)
        parser.add_argument('--v_kps_guider_path', type=str, default=v_kps_guider_path)
        parser.add_argument('--audio_projection_path', type=str, default=audio_projection_path)
        parser.add_argument('--motion_module_path', type=str, default=motion_module_path)

        parser.add_argument('--retarget_strategy', type=str, default=retarget_strategy,
                            help='{fix_face, no_retarget, offset_retarget, naive_retarget}')

        parser.add_argument('--dtype', type=str, default=dtype)
        parser.add_argument('--device', type=str, default=device)
        parser.add_argument('--gpu_id', type=int, default=gpu_id)
        parser.add_argument('--do_multi_devices_inference', action='store_true', default=do_multi_devices_inference)
        parser.add_argument('--save_gpu_memory', action='store_true', default=save_gpu_memory)

        parser.add_argument('--num_pad_audio_frames', type=int, default=num_pad_audio_frames)
        parser.add_argument('--standard_audio_sampling_rate', type=int, default=standard_audio_sampling_rate)

        parser.add_argument('--reference_image_path', type=str, default=reference_image_path)
        parser.add_argument('--audio_path', type=str, default=audio_path)
        parser.add_argument('--kps_path', type=str, default=kps_path)
        parser.add_argument('--output_path', type=str, default=output_path)

        parser.add_argument('--image_width', type=int, default=image_width)
        parser.add_argument('--image_height', type=int, default=image_height)
        parser.add_argument('--fps', type=float, default=fps)
        parser.add_argument('--seed', type=int, default=seed)
        parser.add_argument('--num_inference_steps', type=int, default=num_inference_steps)
        parser.add_argument('--guidance_scale', type=float, default=guidance_scale)
        parser.add_argument('--context_frames', type=int, default=context_frames)
        parser.add_argument('--context_overlap', type=int, default=context_overlap)
        parser.add_argument('--reference_attention_weight', type=float, default=reference_attention_weight)
        parser.add_argument('--audio_attention_weight', type=float, default=audio_attention_weight)

        return parser.parse_args()

    @classmethod
    def run_video_generation(cls, **kwargs):
        args = cls.parse_args(**kwargs)
        video_generator = cls(args)
        result_path = video_generator.generate_video()
        return result_path



