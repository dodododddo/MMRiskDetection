import opennsfw2 as n2

def sex_detect_(image_path):
    nsfw_probability = n2.predict_image(image_path)
    return nsfw_probability > 0.5
