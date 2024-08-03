import numpy as np
import cv2
from io import BytesIO
from cnocr import CnOcr

def ocr(image_path:str, model:CnOcr, save_image=False, result_path=None):
    if isinstance(image_path, BytesIO):
        return [False, '']
    outs_list = model.ocr(image_path)
    out = ""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    if result_path == None:
        result_path = "data/ocr/result/" + image_path.split('/')[-1]
    if (not outs_list) and (not image_path.startswith(('http://', 'https://'))):
        # cv2.imencode(result_path, image)[1].tofile(result_path)
        return [False, out]
    for out_dict in outs_list:
        out += out_dict["text"]
        out += "\n"
        if save_image == True:
            position = np.int0(out_dict["position"])
            position = position.reshape((-1, 1, 2))
            if abs(position[0][0][0] - 989) >= 50 and [331, 538] not in position[0]:
                print(position)
                cv2.drawContours(image, [position], 0, (0, 0, 255), 2)
                cv2.imencode(result_path, image)[1].tofile(result_path)
    return [True, out]

if __name__ == '__main__':
    print(ocr('/data1/home/jrchen/MMRiskDetection/ImagePipeline/data/ocr/诈骗聊天记录.jpg', CnOcr()))