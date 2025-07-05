import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os

def load_pdf_text(file_path, use_ocr=False):
    """
    Extract text from a PDF using either direct extraction or OCR fallback.

    Args:
        file_path (str): Path to the PDF file.
        use_ocr (bool): Force use of OCR even if text is extractable.

    Returns:
        str: Extracted text content.
    """
    text = ""

    if not use_ocr:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"⚠️ Error during pdfplumber extraction: {e}")

    # Fallback or forced OCR
    try:
        images = convert_from_path(file_path)
        for img in images:
            text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return ""

    return text.strip()
