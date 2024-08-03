import gradio as gr
import requests

def reply(message, history):
    resp = requests.post(url = "http://127.0.0.1:2222/generate", json={'message': message}).json()
    return resp

def structure(resp):
    return resp['answer'] + '\n' + '\n' + '参考案例：' + resp['ref_examples']

if __name__ == '__main__':
    demo = gr.ChatInterface(reply)
    demo.launch()