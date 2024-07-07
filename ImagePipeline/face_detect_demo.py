from utils import detect_model, predict, recursively_read

arch = 'CLIP:ViT-L/14'
face_detect_model = detect_model(arch, "model/UniversalFakeDetect/checkpoints/clip_vitl14/ffhq_dffd_best.pth")
# img_paths = recursively_read('data/DFFD/faceapp/test')
img_path = '../VideoPipeline/data/DFMINST+_image/true/1/output_0010.png'
# img_path = 'data/DFFD/ffhq/train/R_FFHQ_00000.png'


result = predict(img_path, face_detect_model, arch)
print(result)