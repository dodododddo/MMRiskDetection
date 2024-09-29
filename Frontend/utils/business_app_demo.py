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
                gr.Markdown("**企业端主要功能对批量内容进行多模态风险分类和风险摘要生成，并对各类风险进行统计。本界面是对企业端后台监控界面的一个简单演示，点击“分析”后，将选取服务器上的小部分数据进行风险识别和摘要生成，并对各类风险进行简单统计。在真实环境下，该过程将持续进行。考虑到真实的企业端后台监控界面只需要能定位有风险数据而无需对数据本身进行展示，我们仅仅记录了数据的路径。该部分数据将在个人端界面中作为示例展示，可切换到功能更丰富的个人端界面体验。**")
                vid_input = gr.Video(label='视频',visible=False)
                output_text = gr.Textbox(label='结果',lines=20)
            with gr.Column():
                output_count_text =  gr.Textbox(label='结果',lines=6)
                # output_bar_chart = gr.Image(label='结果')
        vid_submit_btn = gr.Button("分析")
        vid_submit_btn.click(fn=pipeline,
                            #  inputs=vid_input,
                             outputs=[output_text, output_count_text],queue=True)

if __name__ == "__main__":
    demo.launch(server_port=7763)
