import gradio as gr
import os
from typing import *
from customer_pipeline import *

def list_dir(
    path,
    prefix: Union[str, Tuple] = "",
    suffix: Union[str, Tuple] = "",
    contains="",
):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    return [
        f"{path}/{name}"
        for name in os.listdir(path)
        if name.startswith(prefix) and name.endswith(suffix) and contains in name
    ]


app_name = '多模态风险识别'
with gr.Blocks(title=app_name) as demo:
    gr.Markdown(app_name, elem_id="page-title")

    # video
    with gr.Tab("视频"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**本项目已对延迟进行优化，但由于本服务器被多个科研项目使用，项目可用显卡有限，可用于KVCache的显存不足，延迟波动较大，视频较长时推理最长约需70s，请耐心等待。带来不便，万分抱歉**")
                vid_input = gr.Video(label='视频')
            with gr.Column():
                video_text = gr.Textbox(label='风险识别')
                image_sex_detection = gr.Textbox(label='色情检测')
                video_df_detection = gr.Textbox(label='视频人脸伪造检测')
                audio_df_detection = gr.Textbox(label='声音伪造检测')
                syn_df_detection = gr.Textbox(label='综合伪造检测')
        vid_submit_btn = gr.Button("分析")
        with gr.Row():
            vid_examples = gr.Examples(
                ['./Frontend/demo/wzy.mp4',
                 './Frontend/demo/test.mp4'],
                inputs=[vid_input],
            )
        vid_submit_btn.click(fn=video_pipeline,
                             inputs=vid_input, 
                             outputs=[image_sex_detection,
                                      video_df_detection,
                                      audio_df_detection,
                                      syn_df_detection,
                                      video_text],queue=True)

    # image
    with gr.Tab("图片"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label='图片',type='filepath')
            with gr.Column():
                imagetext_output = gr.Textbox(label='风险识别')
                imagesynthesis_output = gr.Textbox(label='合成检测')
                imagesex_output = gr.Textbox(label='色情检测')
        image_submit_btn = gr.Button("分析")
        with gr.Row():
            image_examples = gr.Examples(
                ['./Frontend/demo/F_PGN2_00003.png',
                 './Frontend/demo/chat.jpg'],
                inputs=[image_input],
            )
        image_submit_btn.click(
            fn=image_pipeline,
            inputs=[image_input],
            outputs=[
                imagesynthesis_output,
                imagesex_output,
                imagetext_output
            ]
        )

    # text
    with gr.Tab("短信"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Text(label='短信')
                text_md_output = gr.Markdown(label='风险检测')
            with gr.Column():
                text_output1 = gr.Textbox(label='风险识别')
                text_output2 = gr.Textbox(label='案例分析')
        audio_submit_btn = gr.Button("分析")
        with gr.Row():
            audio_examples = gr.Examples(
                ['您好，我是金源贷款的客服，请问您是否有贷款需求？如果有，可以加我的QQ（QQ号：123456789）详细了解。我们提供快速便捷的贷款服务，您只需下载我们的APP并填写相关信息即可。'],
                inputs=[text_input],
            )
        audio_submit_btn.click(
            fn=text_pipeline,
            inputs=[text_input],
            outputs=[
                text_output1,
                text_output2,
                text_md_output,
            ],
            queue=True
        )


    # audio
    with gr.Tab("声音"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label='声音', type='filepath')
            with gr.Column():
                audiotext_output = gr.Textbox(label='风险检测')
                audiodetect_output = gr.Textbox(label='伪造检测')
        audio_submit_btn = gr.Button("分析")
        with gr.Row():
            audio_examples = gr.Examples(
                ['./Frontend/demo/demo.wav'],
                inputs=[audio_input],
            )
        audio_submit_btn.click(
            fn=audio_pipeline,
            inputs=[audio_input],
            outputs=[
                audiodetect_output,
                audiotext_output
            ]
        )
    
    # 网页
    with gr.Tab("网页"):
        with gr.Row():
            with gr.Column():
                web_input = gr.Text(label='网址')
            with gr.Column():
                webtext_output = gr.Textbox(label='风险识别') 
                webimage_output = gr.Textbox(label='图片识别')
        web_submit_btn = gr.Button("分析")
        with gr.Row():
            web_examples = gr.Examples(
                ['https://www.kunnu.com/doupo/'],
                inputs=[web_input],
            )
        web_submit_btn.click(
            fn=web_pipeline,
            inputs=[web_input],
            outputs=[
                webimage_output,
                webtext_output
            ]
        )

    # PDF文件
    with gr.Tab("文件"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label='文件')
            with gr.Column():
                filetext_output = gr.Textbox(label='文字风险识别')
                fileimage_output = gr.Textbox(label='图像色情检测') 
        file_submit_btn = gr.Button("分析")
        with gr.Row():
            file_examples = gr.Examples(
                ['./Frontend/demo/template.pdf'],
                inputs=[file_input],
            )
        file_submit_btn.click(
            fn=file_pipeline,
            inputs=[file_input],
            outputs=[
                filetext_output,
                fileimage_output
            ]
        )

    # 数字人
    with gr.Tab("数字人"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**本项目已对延迟进行优化，但由于本服务器被多个科研项目使用，项目可用显卡有限，可用于KVCache的显存不足，延迟波动较大，生成最长约需300s，请耐心等待。带来不便，万分抱歉**")
                dh_image_input = gr.Image(label='图片', type='filepath')
                dh_audio_input = gr.Audio(label='声音', type='filepath')
                dh_text_input = gr.Text(label='文本')
            with gr.Column():
                dh_videopath_output = gr.Video(label='输出视频')
        dh_submit_btn = gr.Button("开始")
        with gr.Row():
            with gr.Column():
                image_examples = gr.Examples(
                    ['./Frontend/demo/wzy.jpg'],
                    inputs=[dh_image_input],
                )
            with gr.Column():
                audio_examples = gr.Examples(
                    ['./Frontend/demo/wzy.wav'],
                    inputs=[dh_audio_input],
                )
            with gr.Column():
                text_examples = gr.Examples(
                    ['账户异常，请确认'],
                    inputs=[dh_text_input],
                )
        
        dh_submit_btn.click(
            fn=digital_humans_pipeline,
            inputs=[
                dh_image_input,
                dh_audio_input,
                dh_text_input
                ],
            outputs=[
                dh_videopath_output
            ]
        )

    # facefusion
    with gr.Tab("换脸"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**本项目已对延迟进行优化，但由于本服务器被多个科研项目使用，项目可用显卡有限，可用于KVCache的显存不足，延迟波动较大，生成最长约需70s，请耐心等待。带来不便，万分抱歉**")
                ff_video_input = gr.Video(label='目标视频')
                ff_image_input = gr.Image(label='原始照片', type='filepath')
            with gr.Column():
                ff_videopath_output = gr.Video(label='输出视频')
        ff_submit_btn = gr.Button("开始")
        with gr.Row():
            ff_video_examples = gr.Examples(
                ['./Frontend/demo/jr.mp4'],
                inputs=[ff_video_input],
            )
        with gr.Row():
            ff_image_examples = gr.Examples(
                ['./Frontend/demo/rj.jpg'],
                inputs=[ff_image_input],
            )
        ff_submit_btn.click(
            fn=facefusion_pipeline,
            inputs=[
                ff_image_input,
                ff_video_input
                ],
            outputs=[
                ff_videopath_output
            ]
        )

if __name__ == "__main__":
    demo.launch(server_port=7862)