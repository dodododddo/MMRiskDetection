from PIL import Image, ImageDraw

# Load the image
image_path = "/data1/home/jrchen/MMRiskDetection/Digital_humans/CodeFormer/Temp/jrzy.png"
image = Image.open(image_path)

# Define the coordinates for the red box (assuming leftmost person's face)
# Coordinates are (left, top, right, bottom)
# These coordinates need to be estimated or provided. 
# For the purpose of this example, we will use a generic estimate:
left = 450
top = 380
right = 780
bottom = 750

# Draw the red box
draw = ImageDraw.Draw(image)
draw.rectangle([left, top, right, bottom], outline="red", width=5)

# Save the modified image
output_path = "/data1/home/jrchen/MMRiskDetection/Digital_humans/CodeFormer/Temp/jrzy_K.png"
image.save(output_path)

output_path
