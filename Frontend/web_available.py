import sys
sys.path.append('.')

import gradio as gr
from WebPipeline.utils import WebPipeline, WebData

webPipeline = WebPipeline()

def test(url):
    web_data:WebData = webPipeline(url)
    return web_data
    
demo = gr.Interface(fn=test, 
                    inputs=[gr.Textbox(label='url')], 
                    outputs=[gr.Textbox(label="ImageData")], 
                    title="网页风险识别", 
                    description="使用大模型进行网页风险识别", 
                    allow_flagging="never", 
                    examples=['https://www.hitsz.edu.cn/index.html', 
                              'https://www.dkxs.net/', 
                              'https://18mh.org/'])
demo.launch(share=False)