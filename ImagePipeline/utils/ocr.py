from cnocr import CnOcr
import numpy as np
import cv2

def ocr(image_path, model, save_image=True, result_path=None):
    outs_list = model.ocr(image_path)
    out = ""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    if result_path == None:
        result_path = "data/ocr/result/" + image_path.split('/')[-1]
    if not outs_list:
        cv2.imencode(result_path, image)[1].tofile(result_path)
        return [False, out]
    for out_dict in outs_list:
        out += out_dict["text"]
        if format:
            out += "\n"
        if save_image == True:
            position = np.int0(out_dict["position"])
            position = position.reshape((-1, 1, 2))
            cv2.drawContours(image, [position], 0, (0, 0, 255), 2)
            cv2.imencode(result_path, image)[1].tofile(result_path)
    return [True, out]