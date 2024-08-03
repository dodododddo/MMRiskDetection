from PyPDF2 import PdfReader
from docx import Document

def get_pdf_text(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        num_pages = len(reader.pages)
        text = ''
        for page in range(num_pages):
            page_obj = reader.pages[page]
            text += page_obj.extract_text()
    return text

def get_word_text(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text 

if __name__ == '__main__':
    print(get_word_text('format2023.doc'))




