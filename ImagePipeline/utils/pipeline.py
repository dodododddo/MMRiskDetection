from PIL import Image

'''MiniCPM-V-2'''
def ocr_pipeline(image_paths, model, tokenizer):
    if type(image_paths).__name__ != 'list':
            image_paths = [image_paths]
    def pipe(question):
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            msgs = [{'role': 'user', 'content': question}]
            res, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7, 
                max_length=500,
                use_cache=True
            )
            print(res)
    return pipe


def risk_pineline(model, processor):
    #  if type(image_paths).__name__ != 'list':
    #         image_paths = [image_paths]
     def pipe(image_path, prompt):
            # for image_path in image_paths:
            image = Image.open(image_path)
            inputs = processor(prompt, image, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=100)
            answer = processor.decode(output[0], skip_special_tokens=True)
            return answer.split('[/INST] ')[-1]
     return pipe