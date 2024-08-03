def web_structure(text_input, image_input):
    # 生成HTML内容
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gradio Web Structure</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 20px;
            }
            .text-container {
                margin-bottom: 20px;
            }
            .image-container {
                border: 1px solid #ccc;
                padding: 10px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h1>Gradio Web Structure Example</h1>
        <div class="text-container">
            <p>{}</p>
        </div>
        <div class="image-container">
            <img src="{}" alt="Input Image" style="max-width: 400px; max-height: 400px;">
        </div>
    </body>
    </html>
    """.format(text_input, image_input)

    # 将HTML内容保存到文件
    with open('output.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
