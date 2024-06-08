import os
from openai import OpenAI
import gradio as gr

def stream_chatbot(client, history, max_history_len=32):
    
    def chat(input_text):
        nonlocal history
        if len(history) > max_history_len:
            history = history[0:2] + history[2 - max_history_len :]
        history.append({"role": "user", "content": input_text})
        reply = ""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=history,
            stream=True
        )
        for resp in response:
            reply += resp.choices[0].delta.content
            yield reply
        history.append({"role": "assistant", "content": reply})
    
    return chat
    

def demo():
    api_key = os.getenv('DEEPSEEK_API_KEY')
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    history = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ]
    # 使用Gradio创建一个简单的界面
    with gr.Blocks() as demo:
        chatbot_output = gr.Textbox(label="Chatbot Output", lines=10)
        user_input = gr.Textbox(label="User Input", lines=5)
        submit_button = gr.Button("Submit")
        chat = stream_chatbot(client, history)
        submit_button.click(
            fn=chat,
            inputs=user_input,
            outputs=chatbot_output,
        )
    demo.launch()


if __name__ == "__main__":
    demo()
    

