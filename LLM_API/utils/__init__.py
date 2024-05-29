def pipeline(client, temperature=1.25):
    def generate(input_text):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=input_text,
            stream=False,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    return generate