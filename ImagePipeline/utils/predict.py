import torchvision.transforms as transforms
from PIL import Image
import torch
from ImagePipeline.model import get_model

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def predict(img_path, model, arch, max_number = 50):
    stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
    ])
    # num = 0;
    # if type(img_paths).__name__ != 'list':
    #     img_paths = [img_paths]
    with open('result.log', 'w') as f:
        # for img_path in img_paths:
            # if num == max_number: 
            #     break
            # num += 1
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        in_tens = img.cuda().unsqueeze(0)
        pred = model(in_tens).sigmoid().flatten().tolist()[0]
        if (pred >= 0.5):
            # print(f'fake {pred:.5f}')
            # f.write('fake ' + str(round(pred, 2)) + '\n')
            return True
        else:
            # print(f'real {pred:.5f}')
            # f.write('real ' + str(round(pred, 2)) + '\n')
            return False

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






