import gradio as gr
import requests

def reply(image_path):
    resp = requests.post(url = "http://127.0.0.1:6666/image", json={'image_path': image_path}).json()
    return resp

if __name__ == '__main__':    
    demo = gr.Interface(fn=reply, 
                    inputs=[gr.Image(label='image', type='filepath')], 
                    outputs=[gr.Markdown(label="ImageData")], 
                    title="网页风险识别", 
                    description="使用大模型进行网页风险识别", 
                    allow_flagging="never", 
                    examples=['ImagePipeline/data/ocr/test/test1.png'])                
    demo.launch()




