import gradio as gr
import requests

def reply(video_path):
    resp = requests.post(url = "http://127.0.0.1:1927/video", json={'video_path': video_path}).json()
    return str(resp)


if __name__ == '__main__':    
    demo = gr.Interface(fn=reply, 
                    inputs=[gr.Video(label='video', type='filepath')], 
                    outputs=[gr.Textbox(label="VideoData")], 
                    title="网页风险识别", 
                    description="使用大模型进行网页风险识别", 
                    allow_flagging="never", 
                    examples=['./data/p_demo.mp4'], )
    demo.launch()