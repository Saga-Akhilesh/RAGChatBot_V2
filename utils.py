import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os

def load_pdf_text(file_path):
    # Try to extract text normally
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        if text.strip():
            return text.strip()
    except:
        pass

    # Fallback to OCR
    text = ""
    images = convert_from_path(file_path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text.strip()
