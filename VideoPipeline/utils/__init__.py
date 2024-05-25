import av
import numpy as np

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def pipeline(model, processor):
    def process(prompt:str, video_path:str)->str:
        container = av.open(video_path)
        # sample uniformly 8 frames from the video
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)

        inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
        
        # Generate
        generate_ids = model.generate(**inputs, max_length=512)
        reply = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return reply[reply.find("ASSISTANT:"):]
    
    return process