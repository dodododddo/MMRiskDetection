import gradio as gr

def tooltip(text: str, tooltip_text: str) -> str:
    html_code = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tooltip Example</title>
            <style>
                .tooltip {{
                    position: relative;
                    display: inline-block;
                    border-bottom: 1px dotted black;
                }}
                .tooltip .tooltiptext {{
                    visibility: hidden;
                    width: 200px;
                    background-color: black;
                    color: #fff;
                    text-align: center;
                    border-radius: 6px;
                    padding: 5px;
                    position: absolute;
                    z-index: 1;
                    bottom: 150%; /* Position the tooltip above the text */
                    left: 50%;
                    margin-left: -100px;
                    opacity: 0;
                    transition: opacity 0.3s;
                }}
                .tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
            </style>
        </head>
        <body>
            <div>
                <h2>Hover over the text below to see the tooltip</h2>
                <span class="tooltip"> {text} <span class="tooltiptext"> {tooltip_text} </span></span>
            </div>
        </body>
        </html>
        """
    return html_code


if __name__ == '__main__':
    def example_function(input_text):
        return "This function doesn't do much"

    html_code = tooltip('hover me', "This function doesn't do much")

    with gr.Blocks() as demo:
        with gr.Row():
            gr.HTML(html_code)
        
        with gr.Row():
            input_text = gr.Textbox(label="input_text")
            output_text = gr.Textbox(label="output")
            submit_button = gr.Button("Submit")

        submit_button.click(example_function, inputs=input_text, outputs=output_text)

    demo.launch()