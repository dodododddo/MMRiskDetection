#版本1

# import os
# import cv2
# import argparse
# import glob
# import torch
# from torchvision.transforms.functional import normalize
# from basicsr.utils import imwrite, img2tensor, tensor2img
# from basicsr.utils.download_util import load_file_from_url
# from basicsr.utils.misc import gpu_is_available, get_device
# from facelib.utils.face_restoration_helper import FaceRestoreHelper
# from facelib.utils.misc import is_gray
# from basicsr.utils.registry import ARCH_REGISTRY

# pretrain_model_url = {
#     'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
# }

# class FaceRestorer:
#     def __init__(self, args):
#         self.args = args
#         self.device = get_device()
#         self.w = args.fidelity_weight
#         self.input_video = False
#         self.input_img_list, self.result_root = self._prepare_input_output()

#         self.bg_upsampler = self._set_bg_upsampler() if args.bg_upsampler == 'realesrgan' else None
#         self.face_upsampler = self.bg_upsampler if args.face_upsample else None

#         self.net = self._set_codeformer_restorer()
#         self.face_helper = self._set_face_helper()

#     def _prepare_input_output(self):
#         args = self.args
#         w = self.w
#         if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):  # input single img path
#             input_img_list = [args.input_path]
#             result_root = f'results/test_img_{w}'
#         elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):  # input video path
#             from basicsr.utils.video_util import VideoReader
#             input_img_list = []
#             vidreader = VideoReader(args.input_path)
#             image = vidreader.get_frame()
#             while image is not None:
#                 input_img_list.append(image)
#                 image = vidreader.get_frame()
#             self.audio = vidreader.get_audio()
#             self.fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
#             self.video_name = os.path.basename(args.input_path)[:-4]
#             result_root = f'results/{self.video_name}_{w}'
#             self.input_video = True
#             vidreader.close()
#         else:  # input img folder
#             if args.input_path.endswith('/'):  # solve when path ends with /
#                 args.input_path = args.input_path[:-1]
#             input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
#             result_root = f'results/{os.path.basename(args.input_path)}_{w}'
#         if not args.output_path is None:  # set output path
#             result_root = args.output_path
#         if len(input_img_list) == 0:
#             raise FileNotFoundError('No input image/video is found...')
#         return input_img_list, result_root

#     def _set_bg_upsampler(self):
#         from basicsr.archs.rrdbnet_arch import RRDBNet
#         from basicsr.utils.realesrgan_utils import RealESRGANer

#         use_half = False
#         if torch.cuda.is_available():  # set False in CPU/MPS mode
#             no_half_gpu_list = ['1650', '1660']  # set False for GPUs that don't support f16
#             if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
#                 use_half = True

#         model = RRDBNet(
#             num_in_ch=3,
#             num_out_ch=3,
#             num_feat=64,
#             num_block=23,
#             num_grow_ch=32,
#             scale=2,
#         )
#         upsampler = RealESRGANer(
#             scale=2,
#             model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
#             model=model,
#             tile=self.args.bg_tile,
#             tile_pad=40,
#             pre_pad=0,
#             half=use_half
#         )

#         if not gpu_is_available():  # CPU
#             import warnings
#             warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
#                           'The unoptimized RealESRGAN is slow on CPU. '
#                           'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
#                           category=RuntimeWarning)
#         return upsampler

#     def _set_codeformer_restorer(self):
#         net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
#                                               connect_list=['32', '64', '128', '256']).to(self.device)
#         ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
#                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
#         checkpoint = torch.load(ckpt_path)['params_ema']
#         net.load_state_dict(checkpoint)
#         net.eval()
#         return net

#     def _set_face_helper(self):
#         face_helper = FaceRestoreHelper(
#             self.args.upscale,
#             face_size=512,
#             crop_ratio=(1, 1),
#             det_model=self.args.detection_model,
#             save_ext='png',
#             use_parse=True,
#             device=self.device)
#         return face_helper

#     def process_images(self):
#         for i, img_path in enumerate(self.input_img_list):
#             self.face_helper.clean_all()
#             img_name, img = self._load_image(img_path, i)
#             if self.args.has_aligned:
#                 img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
#                 self.face_helper.is_gray = is_gray(img, threshold=10)
#                 if self.face_helper.is_gray:
#                     print('Grayscale input: True')
#                 self.face_helper.cropped_faces = [img]
#             else:
#                 self.face_helper.read_image(img)
#                 num_det_faces = self.face_helper.get_face_landmarks_5(
#                     only_center_face=self.args.only_center_face, resize=640, eye_dist_threshold=5)
#                 print(f'\tdetect {num_det_faces} faces')
#                 self.face_helper.align_warp_face()

#             self._restore_faces()
#             self._paste_back_faces(img, img_name)

#     def _load_image(self, img_path, idx):
#         if isinstance(img_path, str):
#             img_name = os.path.basename(img_path)
#             basename, ext = os.path.splitext(img_name)
#             print(f'[{idx + 1}/{len(self.input_img_list)}] Processing: {img_name}')
#             img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         else:
#             basename = str(idx).zfill(6)
#             img_name = f'{self.video_name}_{basename}' if self.input_video else basename
#             print(f'[{idx + 1}/{len(self.input_img_list)}] Processing: {img_name}')
#             img = img_path
#         return img_name, img

#     def _restore_faces(self):
#         for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
#             cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
#             normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
#             cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

#             try:
#                 with torch.no_grad():
#                     output = self.net(cropped_face_t, w=self.w, adain=True)[0]
#                     restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
#                 del output
#                 torch.cuda.empty_cache()
#             except Exception as error:
#                 print(f'\tFailed inference for CodeFormer: {error}')
#                 restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

#             restored_face = restored_face.astype('uint8')
#             self.face_helper.add_restored_face(restored_face, cropped_face)

#     def _paste_back_faces(self, img, img_name):
#         if not self.args.has_aligned:
#             bg_img = self.bg_upsampler.enhance(img, outscale=self.args.upscale)[0] if self.bg_upsampler else None
#             self.face_helper.get_inverse_affine(None)
#             if self.args.face_upsample and self.face_upsampler:
#                 restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.args.draw_box, face_upsampler=self.face_upsampler)
#             else:
#                 restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.args.draw_box)

#         self._save_images(img_name, restored_img)

#     def _save_images(self, img_name, restored_img):
#         for idx, (cropped_face, restored_face) in enumerate(zip(self.face_helper.cropped_faces, self.face_helper.restored_faces)):
#             if not self.args.has_aligned:
#                 save_crop_path = os.path.join(self.result_root, 'cropped_faces', f'{img_name}_{idx:02d}.png')
#                 imwrite(cropped_face, save_crop_path)
#             save_face_name = f'{img_name}.png' if self.args.has_aligned else f'{img_name}_{idx:02d}.png'
#             if self.args.suffix is not None:
#                 save_face_name = f'{save_face_name[:-4]}_{self.args.suffix}.png'
#             save_restore_path = os.path.join(self.result_root, 'restored_faces', save_face_name)
#             imwrite(restored_face, save_restore_path)

#         if not self.args.has_aligned and restored_img is not None:
#             if self.args.suffix is not None:
#                 img_name = f'{img_name}_{self.args.suffix}'
#             save_restore_path = os.path.join(self.result_root, 'final_results', f'{img_name}.png')
#             imwrite(restored_img, save_restore_path)

#     def save_video(self):
#         if self.input_video:
#             from basicsr.utils.video_util import VideoWriter
#             print('Video Saving...')
#             video_frames = []
#             img_list = sorted(glob.glob(os.path.join(self.result_root, 'final_results', '*.[jp][pn]g')))
#             for img_path in img_list:
#                 img = cv2.imread(img_path)
#                 video_frames.append(img)
#             height, width = video_frames[0].shape[:2]
#             if self.args.suffix is not None:
#                 video_name = f'{self.video_name}_{self.args.suffix}.png'
#             save_restore_path = os.path.join(self.result_root, f'{self.video_name}.mp4')
#             vidwriter = VideoWriter(save_restore_path, height, width, self.fps, self.audio)
#             for f in video_frames:
#                 vidwriter.write_frame(f)
#             vidwriter.close()

#     def run(self):
#         self.process_images()
#         self.save_video()
#         print(f'\nAll results are saved in {self.result_root}')


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input_path', type=str, default='./inputs/whole_imgs',
#                         help='Input image, video or folder. Default: inputs/whole_imgs')
#     parser.add_argument('-o', '--output_path', type=str, default=None,
#                         help='Output folder. Default: results/<input_name>_<w>')
#     parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5,
#                         help='Balance the quality and fidelity. Default: 0.5')
#     parser.add_argument('-s', '--upscale', type=int, default=2,
#                         help='The final upsampling scale of the image. Default: 2')
#     parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
#     parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
#     parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
#     parser.add_argument('--detection_model', type=str, default='retinaface_resnet50',
#                         help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
#                         Default: retinaface_resnet50')
#     parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
#     parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
#     parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
#     parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
#     parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')
#     return parser.parse_args()


# if __name__ == '__main__':
#     args = parse_args()
#     face_restorer = FaceRestorer(args)
#     face_restorer.run()

#版本2
# import os
# import cv2
# import argparse
# import glob
# import torch
# from torchvision.transforms.functional import normalize
# from basicsr.utils import imwrite, img2tensor, tensor2img
# from basicsr.utils.download_util import load_file_from_url
# from basicsr.utils.misc import gpu_is_available, get_device
# from facelib.utils.face_restoration_helper import FaceRestoreHelper
# from facelib.utils.misc import is_gray
# from basicsr.utils.registry import ARCH_REGISTRY

# pretrain_model_url = {
#     'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
# }

# class FaceRestorer:
#     def __init__(self, args):
#         self.args = args
#         self.device = get_device()
#         self.w = args.fidelity_weight
#         self.input_video = False
#         self.input_img_list, self.result_root = self._prepare_input_output()

#         self.bg_upsampler = self._set_bg_upsampler() if args.bg_upsampler == 'realesrgan' else None
#         self.face_upsampler = self.bg_upsampler if args.face_upsample else None

#         self.net = self._set_codeformer_restorer()
#         self.face_helper = self._set_face_helper()

#     def _prepare_input_output(self):
#         args = self.args
#         w = self.w
#         if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
#             input_img_list = [args.input_path]
#             result_root = f'results/test_img_{w}'
#         elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
#             from basicsr.utils.video_util import VideoReader
#             input_img_list = []
#             vidreader = VideoReader(args.input_path)
#             image = vidreader.get_frame()
#             while image is not None:
#                 input_img_list.append(image)
#                 image = vidreader.get_frame()
#             self.audio = vidreader.get_audio()
#             self.fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
#             self.video_name = os.path.basename(args.input_path)[:-4]
#             result_root = f'results/{self.video_name}_{w}'
#             self.input_video = True
#             vidreader.close()
#         else:
#             if args.input_path.endswith('/'):
#                 args.input_path = args.input_path[:-1]
#             input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
#             result_root = f'results/{os.path.basename(args.input_path)}_{w}'
#         if not args.output_path is None:
#             result_root = args.output_path
#         if len(input_img_list) == 0:
#             raise FileNotFoundError('No input image/video is found...')
#         return input_img_list, result_root

#     def _set_bg_upsampler(self):
#         from basicsr.archs.rrdbnet_arch import RRDBNet
#         from basicsr.utils.realesrgan_utils import RealESRGANer

#         use_half = False
#         if torch.cuda.is_available():
#             no_half_gpu_list = ['1650', '1660']
#             if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
#                 use_half = True

#         model = RRDBNet(
#             num_in_ch=3,
#             num_out_ch=3,
#             num_feat=64,
#             num_block=23,
#             num_grow_ch=32,
#             scale=2,
#         )
#         upsampler = RealESRGANer(
#             scale=2,
#             model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
#             model=model,
#             tile=self.args.bg_tile,
#             tile_pad=40,
#             pre_pad=0,
#             half=use_half
#         )

#         if not gpu_is_available():
#             import warnings
#             warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
#                           'The unoptimized RealESRGAN is slow on CPU. '
#                           'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
#                           category=RuntimeWarning)
#         return upsampler

#     def _set_codeformer_restorer(self):
#         net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
#                                               connect_list=['32', '64', '128', '256']).to(self.device)
#         ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
#                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
#         checkpoint = torch.load(ckpt_path)['params_ema']
#         net.load_state_dict(checkpoint)
#         net.eval()
#         return net

#     def _set_face_helper(self):
#         face_helper = FaceRestoreHelper(
#             self.args.upscale,
#             face_size=512,
#             crop_ratio=(1, 1),
#             det_model=self.args.detection_model,
#             save_ext='png',
#             use_parse=True,
#             device=self.device)
#         return face_helper

#     def process_images(self):
#         for i, img_path in enumerate(self.input_img_list):
#             self.face_helper.clean_all()
#             img_name, img = self._load_image(img_path, i)
#             if self.args.has_aligned:
#                 img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
#                 self.face_helper.is_gray = is_gray(img, threshold=10)
#                 if self.face_helper.is_gray:
#                     print('Grayscale input: True')
#                 self.face_helper.cropped_faces = [img]
#             else:
#                 self.face_helper.read_image(img)
#                 num_det_faces = self.face_helper.get_face_landmarks_5(
#                     only_center_face=self.args.only_center_face, resize=640, eye_dist_threshold=5)
#                 print(f'\tdetect {num_det_faces} faces')
#                 self.face_helper.align_warp_face()

#             self._restore_faces()
#             self._paste_back_faces(img, img_name)

#     def _load_image(self, img_path, idx):
#         if isinstance(img_path, str):
#             img_name = os.path.basename(img_path)
#             basename, ext = os.path.splitext(img_name)
#             print(f'[{idx + 1}/{len(self.input_img_list)}] Processing: {img_name}')
#             img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         else:
#             basename = str(idx).zfill(6)
#             img_name = f'{self.video_name}_{basename}' if self.input_video else basename
#             print(f'[{idx + 1}/{len(self.input_img_list)}] Processing: {img_name}')
#             img = img_path
#         return img_name, img

#     def _restore_faces(self):
#         for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
#             cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
#             normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
#             cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

#             try:
#                 with torch.no_grad():
#                     output = self.net(cropped_face_t, w=self.w, adain=True)[0]
#                     restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
#                 del output
#                 torch.cuda.empty_cache()
#             except Exception as error:
#                 print(f'\tFailed inference for CodeFormer: {error}')
#                 restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

#             restored_face = restored_face.astype('uint8')
#             self.face_helper.add_restored_face(restored_face, cropped_face)

#     def _paste_back_faces(self, img, img_name):
#         if not self.args.has_aligned:
#             bg_img = self.bg_upsampler.enhance(img, outscale=self.args.upscale)[0] if self.bg_upsampler else None
#             self.face_helper.get_inverse_affine(None)
#             if self.args.face_upsample and self.face_upsampler:
#                 restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.args.draw_box, face_upsampler=self.face_upsampler)
#             else:
#                 restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.args.draw_box)

#         return self._save_images(img_name, restored_img)

#     def _save_images(self, img_name, restored_img):
#         save_paths = []
#         for idx, (cropped_face, restored_face) in enumerate(zip(self.face_helper.cropped_faces, self.face_helper.restored_faces)):
#             if not self.args.has_aligned:
#                 save_crop_path = os.path.join(self.result_root, 'cropped_faces', f'{img_name}_{idx:02d}.png')
#                 imwrite(cropped_face, save_crop_path)
#                 save_paths.append(save_crop_path)
#             save_face_name = f'{img_name}.png' if self.args.has_aligned else f'{img_name}_{idx:02d}.png'
#             if self.args.suffix is not None:
#                 save_face_name = f'{save_face_name[:-4]}_{self.args.suffix}.png'
#             save_restore_path = os.path.join(self.result_root, 'restored_faces', save_face_name)
#             imwrite(restored_face, save_restore_path)
#             save_paths.append(save_restore_path)

#         if not self.args.has_aligned and restored_img is not None:
#             if self.args.suffix is not None:
#                 img_name = f'{img_name}_{self.args.suffix}'
#             save_restore_path = os.path.join(self.result_root, 'final_results', f'{img_name}.png')
#             imwrite(restored_img, save_restore_path)
#             save_paths.append(save_restore_path)

#         return save_paths

#     def save_video(self):
#         if self.input_video:
#             from basicsr.utils.video_util import VideoWriter
#             print('Video Saving...')
#             video_frames = []
#             img_list = sorted(glob.glob(os.path.join(self.result_root, 'final_results', '*.[jp][pn]g')))
#             for img_path in img_list:
#                 img = cv2.imread(img_path)
#                 video_frames.append(img)
#             height, width = video_frames[0].shape[:2]
#             if self.args.suffix is not None:
#                 video_name = f'{self.video_name}_{self.args.suffix}.png'
#             save_restore_path = os.path.join(self.result_root, f'{self.video_name}.mp4')
#             vidwriter = VideoWriter(save_restore_path, height, width, self.fps, self.audio)
#             for f in video_frames:
#                 vidwriter.write_frame(f)
#             vidwriter.close()
#             return save_restore_path
#         return None

#     def run(self):
#         image_paths = []
#         for img_path in self.process_images():
#             image_paths.extend(img_path)
#         video_path = self.save_video()
#         print(f'\nAll results are saved in {self.result_root}')
#         return image_paths, video_path


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input_path', type=str, default='./inputs/whole_imgs',
#                         help='Input image, video or folder. Default: inputs/whole_imgs')
#     parser.add_argument('-o', '--output_path', type=str, default=None,
#                         help='Output folder. Default: results/<input_name>_<w>')
#     parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5,
#                         help='Balance the quality and fidelity. Default: 0.5')
#     parser.add_argument('-s', '--upscale', type=int, default=2,
#                         help='The final upsampling scale of the image. Default: 2')
#     parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
#     parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
#     parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
#     parser.add_argument('--detection_model', type=str, default='retinaface_resnet50',
#                         help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
#                         Default: retinaface_resnet50')
#     parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
#     parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
#     parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
#     parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
#     parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')
#     return parser.parse_args()


# if __name__ == '__main__':
#     args = parse_args()
#     face_restorer = FaceRestorer(args)
#     image_paths, video_path = face_restorer.run()
#     print(f'Restored image paths: {image_paths}')
#     if video_path:
#         print(f'Restored video path: {video_path}')

#版本3

import os
import cv2
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.registry import ARCH_REGISTRY
import argparse

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

class FaceRestorer:
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        self.w = args.fidelity_weight
        self.input_video = False
        self.input_img_list, self.result_root = self._prepare_input_output()

        self.bg_upsampler = self._set_bg_upsampler() if args.bg_upsampler == 'realesrgan' else None
        self.face_upsampler = self.bg_upsampler if args.face_upsample else None

        self.net = self._set_codeformer_restorer()
        self.face_helper = self._set_face_helper()

    def _prepare_input_output(self):
        args = self.args
        w = self.w
        if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
            input_img_list = [args.input_path]
            result_root = f'../../DataBuffer/DigitalBuffer'
        elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
            from basicsr.utils.video_util import VideoReader
            input_img_list = []
            vidreader = VideoReader(args.input_path)
            image = vidreader.get_frame()
            while image is not None:
                input_img_list.append(image)
                image = vidreader.get_frame()
            self.audio = vidreader.get_audio()
            self.fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
            self.video_name = os.path.basename(args.input_path)[:-4]
            result_root = f'../../DataBuffer/DigitalBuffer/{self.video_name}'
            self.input_video = True
            vidreader.close()
        else:
            if args.input_path.endswith('/'):
                args.input_path = args.input_path[:-1]
            input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
            result_root = f'results/{os.path.basename(args.input_path)}_{w}'
        if not args.output_path is None:
            result_root = args.output_path
        if len(input_img_list) == 0:
            raise FileNotFoundError('No input image/video is found...')
        return input_img_list, result_root

    def _set_bg_upsampler(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer

        use_half = False
        if torch.cuda.is_available():
            no_half_gpu_list = ['1650', '1660']
            if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
                use_half = True

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
            model=model,
            tile=self.args.bg_tile,
            tile_pad=40,
            pre_pad=0,
            half=use_half
        )

        if not gpu_is_available():
            import warnings
            warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                          'The unoptimized RealESRGAN is slow on CPU. '
                          'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                          category=RuntimeWarning)
        return upsampler

    def _set_codeformer_restorer(self):
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                              connect_list=['32', '64', '128', '256']).to(self.device)
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                       model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()
        return net

    def _set_face_helper(self):
        face_helper = FaceRestoreHelper(
            self.args.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=self.args.detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device)
        return face_helper

    def process_images(self):
        for i, img_path in enumerate(self.input_img_list):
            self.face_helper.clean_all()
            img_name, img = self._load_image(img_path, i)
            if self.args.has_aligned:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                self.face_helper.is_gray = is_gray(img, threshold=10)
                if self.face_helper.is_gray:
                    print('Grayscale input: True')
                self.face_helper.cropped_faces = [img]
            else:
                self.face_helper.read_image(img)
                num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=self.args.only_center_face, resize=640, eye_dist_threshold=5)
                print(f'\tdetect {num_det_faces} faces')
                self.face_helper.align_warp_face()

            self._restore_faces()
            self._paste_back_faces(img, img_name)

    def _load_image(self, img_path, idx):
        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f'[{idx + 1}/{len(self.input_img_list)}] Processing: {img_name}')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else:
            basename = str(idx).zfill(6)
            img_name = f'{self.video_name}_{basename}' if self.input_video else basename
            print(f'[{idx + 1}/{len(self.input_img_list)}] Processing: {img_name}')
            img = img_path
        return img_name, img

    def _restore_faces(self):
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=self.w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face, cropped_face)

    def _paste_back_faces(self, img, img_name):
        if not self.args.has_aligned:
            bg_img = self.bg_upsampler.enhance(img, outscale=self.args.upscale)[0] if self.bg_upsampler else None
            self.face_helper.get_inverse_affine(None)
            if self.args.face_upsample and self.face_upsampler:
                restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.args.draw_box, face_upsampler=self.face_upsampler)
            else:
                restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.args.draw_box)

        self._save_images(img_name, restored_img)

    def _save_images(self, img_name, restored_img):
        for idx, (cropped_face, restored_face) in enumerate(zip(self.face_helper.cropped_faces, self.face_helper.restored_faces)):
            if not self.args.has_aligned:
                save_crop_path = os.path.join(self.result_root, 'cropped_faces', f'{img_name}.jpg')
                print('--------------------------------------------------')
                print(save_crop_path)
                print('--------------------------------------------------')
                imwrite(cropped_face, save_crop_path)
            save_face_name = f'{img_name}.jpg' if self.args.has_aligned else f'{img_name}.jpg'
            if self.args.suffix is not None:
                save_face_name = f'{save_face_name[:-4]}_{self.args.suffix}.jpg'
            save_restore_path = os.path.join(self.result_root, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)

        if not self.args.has_aligned and restored_img is not None:
            if self.args.suffix is not None:
                img_name = f'{img_name}_{self.args.suffix}'
            save_restore_path = os.path.join(self.result_root, 'final_results', f'{img_name}.jpg')
            imwrite(restored_img, save_restore_path)

    def save_video(self):
        print('------------------------------------------------')
        print(1)
        if self.input_video:
            from basicsr.utils.video_util import VideoWriter
            print('Video Saving...')
            video_frames = []
            img_list = sorted(glob.glob(os.path.join(self.result_root, 'final_results', '*.[jp][pn]g')))
            for img_path in img_list:
                img = cv2.imread(img_path)
                video_frames.append(img)
            height, width = video_frames[0].shape[:2]
            if self.args.suffix is not None:
                video_name = f'{self.video_name}_{self.args.suffix}.jpg'
            save_restore_path = os.path.join('../../Frontend/demo', f'{self.video_name}.mp4')
            vidwriter = VideoWriter(save_restore_path, height, width, self.fps, self.audio)
            for f in video_frames:
                vidwriter.write_frame(f)
            vidwriter.close()

    def run(self):
        self.process_images()
        self.save_video()
        print(f'\nAll results are saved in {self.result_root}')
        return self.result_root

    @staticmethod
    def parse_args(input_path, output_path=None, fidelity_weight=0.5, upscale=2, has_aligned=False,
                   only_center_face=False, draw_box=False, detection_model='retinaface_resnet50',
                   bg_upsampler='None', face_upsample=False, bg_tile=400, suffix=None, save_video_fps=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input_path', type=str, default=input_path,
                            help='Input image, video or folder. Default: inputs/whole_imgs')
        parser.add_argument('-o', '--output_path', type=str, default=output_path,
                            help='Output folder. Default: results/<input_name>_<w>')
        parser.add_argument('-w', '--fidelity_weight', type=float, default=fidelity_weight,
                            help='Balance the quality and fidelity. Default: 0.5')
        parser.add_argument('-s', '--upscale', type=int, default=upscale,
                            help='The final upsampling scale of the image. Default: 2')
        parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
        parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
        parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
        parser.add_argument('--detection_model', type=str, default=detection_model,
                            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                            Default: retinaface_resnet50')
        parser.add_argument('--bg_upsampler', type=str, default=bg_upsampler, help='Background upsampler. Optional: realesrgan')
        parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
        parser.add_argument('--bg_tile', type=int, default=bg_tile, help='Tile size for background sampler. Default: 400')
        parser.add_argument('--suffix', type=str, default=suffix, help='Suffix of the restored faces. Default: None')
        parser.add_argument('--save_video_fps', type=float, default=save_video_fps, help='Frame rate for saving video. Default: None')
        return parser.parse_args()

    @classmethod
    def run_face_restoration(cls, input_path, output_path=None, fidelity_weight=0.5, upscale=2, has_aligned=False,
                             only_center_face=False, draw_box=False, detection_model='retinaface_resnet50',
                             bg_upsampler='None', face_upsample=False, bg_tile=400, suffix=None, save_video_fps=None):
        args = cls.parse_args(input_path, output_path, fidelity_weight, upscale, has_aligned, only_center_face,
                              draw_box, detection_model, bg_upsampler, face_upsample, bg_tile, suffix, save_video_fps)
        face_restorer = cls(args)
        result_path = face_restorer.run()
        return result_path

# Example usage:
# result = FaceRestorer.run_face_restoration(input_path='./inputs/whole_imgs', output_path='./results', fidelity_weight=0.5)
# print(result)


    


# Example usage:
# result = run_face_restoration(input_path='./inputs/whole_imgs', output_path='./results', fidelity_weight=0.5)
# print(result)
