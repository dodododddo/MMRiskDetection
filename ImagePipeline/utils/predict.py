import torchvision.transforms as transforms
from PIL import Image
import torch
from model import get_model
import requests
from io import BytesIO

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def predict(img_paths:str|list, model, arch, thres=0.5):
    stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from]),
    ])
    if isinstance(img_paths, str):
        img_paths = [img_paths]
    imgs = [Image.open(img_path).resize((299,299), Image.LANCZOS).convert("RGB") for img_path in img_paths]
    imgs = [transform(img) for img in imgs]
    in_tens = torch.stack([img.cuda() for img in imgs], dim=0)
    preds = model(in_tens).sigmoid().flatten()
    preds = (preds > thres).tolist()
    return True in preds


def detect_model(arch="CLIP:ViT-L/14", weight="model/UniversalFakeDetect/pretrained_weights/fc_weights.pth"):
    model = get_model(arch)
    state_dict = torch.load(weight, map_location='cpu')
    if weight.split('/')[-1] == 'fc_weights.pth':
        model.fc.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict['model'])
    print ("Model loaded..")
    model.eval()
    model.cuda()
    return model






