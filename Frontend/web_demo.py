import gradio as gr
import requests
 
def reply(web_url):
    resp = requests.post(url = "http://127.0.0.1:6667/web", json={'web_url': web_url}).json()
    return resp

demo = gr.Interface(fn=reply, 
                    inputs=[gr.Textbox(label='url')], 
                    outputs=[gr.Textbox(label="ImageData")], 
                    title="网页风险识别", 
                    description="使用大模型进行网页风险识别", 
                    allow_flagging="never", 
                    examples=['https://www.hitsz.edu.cn/index.html', 
                              'https://www.dkxs.net/', 
                              'https://18mh.org/'])
demo.launch()