import gradio as gr
import os
from typing import *
from business_pipeline import *

app_name = '多模态风险识别'
with gr.Blocks(title=app_name) as demo:
    gr.Markdown(app_name, elem_id="page-title")

    # video
    with gr.Tab("企业端"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label='视频',visible=False)
                output_text = gr.Textbox(label='结果',lines=20)
            with gr.Column():
                output_count_text =  gr.Textbox(label='结果',lines=6)
        vid_submit_btn = gr.Button("分析")
        vid_submit_btn.click(fn=pipeline,
                            #  inputs=vid_input, 
                             outputs=[output_text, output_count_text],queue=True)

if __name__ == "__main__":
    demo.launch(server_port=7763)
