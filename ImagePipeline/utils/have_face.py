import face_recognition
from PIL import Image
import numpy as np
import os

def have_face(image_path:str|list, output='../DataBuffer/FaceCropBuffer') -> list:
    if isinstance(image_path, str):
        image_path = [image_path]
    output_paths = []
    for img_path in image_path:
        origin_image = Image.open(img_path)
        image = np.array(origin_image.convert('RGB'))
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image) 
        print(face_locations)
        for i, face_location in enumerate(face_locations):
            crop_box = (face_location[3] * 0.9, face_location[0] * 0.9, face_location[1] * 1.1, face_location[2] * 1.1)
            cropped_img = origin_image.crop(crop_box)
            print(image_path)
            output_path = output + '/' + str(i) + '_' + img_path.split('/')[-1]
            output_paths.append(output_path)
            cropped_img.save(output_path)
    return output_paths

def recursively_read(rootdir, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts):
                out.append(os.path.join(r, file))
    return out
