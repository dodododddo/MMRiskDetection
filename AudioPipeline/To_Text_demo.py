import gradio as gr
import requests

def reply(audio_path):
    resp = requests.post(url = "http://127.0.0.1:9999/audio", json={'audio_path': audio_path}).json()
    print(resp)
    return resp["text"]

if __name__ == '__main__':    
    demo = gr.Interface(fn=reply, 
                    inputs=[gr.Audio(label='audio', type='filepath')], 
                    outputs=[gr.Markdown(label="AudioData")], 
                    title="语音转文字", 
                    description="使用ChatTTS进行语音转文字", 
                    allow_flagging="never", 
                    examples=['./test_api/SSB00090187.wav'])                
    demo.launch()




