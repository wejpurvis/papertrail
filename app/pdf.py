"""
Utilit module to extract text from a PDF file
"""

import io
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )
