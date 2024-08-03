import fitz 
import io
from PIL import Image
from docx import Document

def get_pdf_image(file_path, output_folder='../DataBuffer/FileImageBuffer'):
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))

            image_path = f"{output_folder}/image_page{page_num+1}_img{img_index+1}.{image_ext}"
            image.save(image_path)
            print(f"Image saved at: {image_path}")
    return output_folder

def get_word_image(file_path, output_folder='../DataBuffer/FileImageBuffer'):
    doc = Document(file_path)

    for i, rel in enumerate(doc.part.rels):
        if "image" in doc.part.rels[rel].target_ref:
            img = doc.part.rels[rel].target_part
            img_data = img.blob
            with open(f"{output_folder}/image_{i+1}.jpg", "wb") as f:
                f.write(img_data)
    return output_folder

