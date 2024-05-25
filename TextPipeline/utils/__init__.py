def pipeline(model, tokenizer):
    def process(input_text, max_new_token=8192, temperature=0.6):

        messages = [
            {"role": "user", "content": input_text},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_token,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)
    return process