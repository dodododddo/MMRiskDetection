import requests

def image_reply(image_path):
    try:
        resp = requests.post(url = "http://127.0.0.1:6666/image", json={'image_path': image_path})
        print(resp.status_code)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {'synthesis':None, 'fakeface': None, 'have_characters':None, 'ocr_content':'', 'image_content':'', 'risk':'', 'sex': None}
    except:
        return {'synthesis':None, 'fakeface': None, 'have_characters':None, 'ocr_content':'', 'image_content':'', 'risk':'', 'sex': None}
    
if __name__ == '__main__':
    image_path = 'data/ocr/诈骗聊天记录.jpg'
    print(image_reply(image_path))