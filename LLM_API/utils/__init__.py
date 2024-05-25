def pipeline(client, seed_template, temperature=1.25):
    
    def generate():
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=seed_template,
            stream=False,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    return generate