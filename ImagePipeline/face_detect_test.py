from utils import auto_model, predict, recursively_read

arch = 'CLIP:ViT-L/14'
detect_model = auto_model(arch)
img_paths = recursively_read('data/DFFD/faceapp/test')

predict(img_paths, detect_model, arch)