import gradio as gr

def web_structure(text_input, image_input):
    # 定义网页结构，包括文本框和图像框
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gradio Web Structure</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 20px;
            }}
            .text-container {{
                margin-bottom: 20px;
            }}
            .image-container {{
                border: 1px solid #ccc;
                padding: 10px;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <h1>Gradio Web Structure Example</h1>
        <div class="text-container">
            <p>{text_input}</p>
        </div>
        <div class="image-container">
            <img src="{image_input}" alt="Input Image" style="max-width: 400px; max-height: 400px;">
        </div>
    </body>
    </html>
    """
    return html_content

def text_structure():
    # 输入文本框
    return gr.Textbox(label="Enter Text", placeholder="Type something here")

def image_structure():
    # 输入图像框
    return gr.Image(label="Upload Image", type="filepath")

# 创建Gradio界面
gr.Interface(
    web_structure, 
    inputs=[text_structure(), image_structure()],
    outputs="html",
    live=True,
    title="Gradio Web Structure Example",
    description="This interface demonstrates a simple web structure with text and image display."
).launch()
